"""Causal offline tests for the opt-in W4 per-style writer ablation.

Run with::

    PYTHONPATH=. python scripts/test_w4_style_split.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx

from app.models import validate_results


_FLAG = "W4_STYLE_SPLIT"
_TECH_SELECTION_FLAG = "HTECH_CANDIDATE_SELECTION"
_STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
_CONTROLLED_ENV = {
    "ENSEMBLE_OBSERVERS": "test/observer",
    "ENSEMBLE_WRITER": "test/writer",
    "ENSEMBLE_CONCISE": "0",
    "STYLE_EXEMPLARS": "1",
    "STRICT_GROUNDING": "0",
    "CREATIVE_DISCIPLINE": "0",
    "WRITER_LENGTH_HINT": "",
    "WRITER_TEMP": "0.5",
    "HTECH_CANDIDATE_SELECTION": "0",
    "VIDEO_OBSERVER": "",
    "OPENROUTER_API_KEY": "w4-offline-test-secret",
}
_V8_DOCKER_WRITER_SHA256 = (
    "bdb2501c938f6a0d9368e8bad25e3a37907432f2f4587ad7b33333708066bf92"
)


@contextmanager
def _loaded_ensemble(
    flag: str | None, tech_candidate_selection: str = "0"
) -> Iterator[object]:
    names = (*_CONTROLLED_ENV, _FLAG)
    before = {name: os.environ.get(name) for name in names}
    existed = {name: name in os.environ for name in names}
    try:
        os.environ.update(_CONTROLLED_ENV)
        if flag is None:
            os.environ.pop(_FLAG, None)
        else:
            os.environ[_FLAG] = flag
        os.environ[_TECH_SELECTION_FLAG] = tech_candidate_selection

        from app import ensemble

        yield importlib.reload(ensemble)
    finally:
        for name in names:
            if existed[name]:
                assert before[name] is not None
                os.environ[name] = before[name]
            else:
                os.environ.pop(name, None)


def _run_with_frame(ensemble) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(b"offline-frame")
        return asyncio.run(ensemble.caption_ensemble_frames([frame], list(_STYLES)))


def _writer_caption(prefix: str, style: str) -> str:
    """Produce a valid synthetic caption for each split-writer test."""
    if style == "humorous_tech":
        return f"{prefix} humorous_tech runtime"
    return f"{prefix} {style}"


def test_default_off_preserves_the_v38_common_writer() -> None:
    with _loaded_ensemble(None) as ensemble:
        assert ensemble.W4_STYLE_SPLIT is False
        base_system = ensemble._writer_system()
        assert hashlib.sha256(base_system.encode("utf-8")).hexdigest() == (
            _V8_DOCKER_WRITER_SHA256
        )
        calls: list[dict] = []

        async def fake_call(
            client, model, system, content, max_tokens, temperature=0.5
        ) -> str:
            calls.append(
                {
                    "model": model,
                    "system": system,
                    "content": content,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            if system == ensemble.OBSERVE_SYSTEM:
                return '["A verified subject is visible.", "The subject moves."]'
            return json.dumps({style: f"legacy {style}" for style in _STYLES})

        ensemble._call = fake_call
        result = _run_with_frame(ensemble)

        writer_calls = [call for call in calls if call["system"] != ensemble.OBSERVE_SYSTEM]
        assert len(writer_calls) == 1
        writer = writer_calls[0]
        assert writer["model"] == "test/writer"
        assert writer["system"].encode("utf-8") == base_system.encode("utf-8")
        assert writer["max_tokens"] == 3000
        assert writer["temperature"] == 0.5
        assert result == {style: f"legacy {style}" for style in _STYLES}


def test_empty_observer_content_is_skipped() -> None:
    with _loaded_ensemble(None) as ensemble:
        assert ensemble._parse_list(None) == []
        assert ensemble._parse_list("") == []


def test_on_runs_four_style_writers_concurrently_with_one_observation_spine() -> None:
    with _loaded_ensemble("1") as ensemble:
        assert ensemble.W4_STYLE_SPLIT is True
        base_system = ensemble._writer_system()
        assert hashlib.sha256(base_system.encode("utf-8")).hexdigest() == (
            _V8_DOCKER_WRITER_SHA256
        )

        observer_calls = 0
        writer_calls: list[dict] = []
        active = 0
        max_active = 0
        all_started = asyncio.Event()

        async def fake_call(
            client, model, system, content, max_tokens, temperature=0.5
        ) -> str:
            nonlocal observer_calls, active, max_active
            if system == ensemble.OBSERVE_SYSTEM:
                observer_calls += 1
                return '["A verified subject is visible.", "The subject moves."]'

            assert system.startswith(base_system)
            suffix = system[len(base_system):]
            matches = [style for style in _STYLES if f'"{style}"' in suffix]
            assert len(matches) == 1, suffix
            style = matches[0]
            call = {
                "style": style,
                "model": model,
                "system": system,
                "content": content,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            writer_calls.append(call)
            active += 1
            max_active = max(max_active, active)
            if len(writer_calls) == 4:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1.0)
            active -= 1
            return json.dumps({"caption": _writer_caption("split", style)})

        ensemble._call = fake_call
        result = _run_with_frame(ensemble)

        assert observer_calls == 1
        assert len(writer_calls) == 4
        assert max_active == 4
        assert {call["style"] for call in writer_calls} == set(_STYLES)
        assert {call["model"] for call in writer_calls} == {"test/writer"}
        assert {call["temperature"] for call in writer_calls} == {0.5}
        assert {call["max_tokens"] for call in writer_calls} == {750}
        assert sum(call["max_tokens"] for call in writer_calls) == 3000
        first_content = writer_calls[0]["content"]
        assert all(call["content"] == first_content for call in writer_calls)
        assert first_content.startswith(
            "Independent observation lists from several vision models for ONE clip. "
            "Cross-reference and write the four captions.\n\n"
        )
        expected = {style: _writer_caption("split", style) for style in _STYLES}
        assert result == expected
        validated = validate_results([{"task_id": "w4", "captions": result}])
        assert validated[0]["captions"] == expected


def test_each_style_keeps_the_existing_single_retry_contract() -> None:
    with _loaded_ensemble("true") as ensemble:
        base_system = ensemble._writer_system()
        attempts = {style: 0 for style in _STYLES}
        original_sleep = ensemble.asyncio.sleep

        async def no_sleep(_seconds: float) -> None:
            return None

        async def fake_call(
            client, model, system, content, max_tokens, temperature=0.5
        ) -> str:
            if system == ensemble.OBSERVE_SYSTEM:
                return '["A verified subject is visible."]'
            suffix = system[len(base_system):]
            style = next(style for style in _STYLES if f'"{style}"' in suffix)
            attempts[style] += 1
            if style == "formal" and attempts[style] == 1:
                raise httpx.TransportError("synthetic transient writer failure")
            return json.dumps({"caption": _writer_caption("retry-safe", style)})

        ensemble.asyncio.sleep = no_sleep
        ensemble._call = fake_call
        try:
            result = _run_with_frame(ensemble)
        finally:
            ensemble.asyncio.sleep = original_sleep

        assert attempts == {
            "formal": 2,
            "sarcastic": 1,
            "humorous_tech": 1,
            "humorous_non_tech": 1,
        }
        assert list(result) == list(_STYLES)
        assert result == {
            style: _writer_caption("retry-safe", style) for style in _STYLES
        }


def test_each_style_retries_malformed_writer_json() -> None:
    with _loaded_ensemble("1") as ensemble:
        base_system = ensemble._writer_system()
        attempts = {style: 0 for style in _STYLES}
        original_sleep = ensemble.asyncio.sleep

        async def no_sleep(_seconds: float) -> None:
            return None

        async def fake_call(
            client, model, system, content, max_tokens, temperature=0.5
        ) -> str:
            if system == ensemble.OBSERVE_SYSTEM:
                return '["A verified subject is visible."]'
            suffix = system[len(base_system):]
            style = next(style for style in _STYLES if f'"{style}"' in suffix)
            attempts[style] += 1
            if style == "formal" and attempts[style] == 1:
                return "not-json"
            return json.dumps({"caption": _writer_caption("parse-safe", style)})

        ensemble.asyncio.sleep = no_sleep
        ensemble._call = fake_call
        try:
            result = _run_with_frame(ensemble)
        finally:
            ensemble.asyncio.sleep = original_sleep

        assert attempts == {
            "formal": 2,
            "sarcastic": 1,
            "humorous_tech": 1,
            "humorous_non_tech": 1,
        }
        assert result == {
            style: _writer_caption("parse-safe", style) for style in _STYLES
        }


def test_humorous_tech_gets_one_grounded_repair_before_outer_fallback() -> None:
    """A missing tech marker must not immediately lose the detailed candidate."""
    with _loaded_ensemble("1") as ensemble:
        base_system = ensemble._writer_system()
        tech_calls = 0

        async def fake_call(
            client, model, system, content, max_tokens, temperature=0.5
        ) -> str:
            nonlocal tech_calls
            if system == ensemble.OBSERVE_SYSTEM:
                return '["An orange kitten walks through green foliage."]'
            suffix = system[len(base_system):]
            style = next(style for style in _STYLES if f'"{style}"' in suffix)
            if style != "humorous_tech":
                return json.dumps({"caption": _writer_caption("valid", style)})
            tech_calls += 1
            if system.endswith(ensemble._TECH_STYLE_REPAIR_RULE):
                return json.dumps({
                    "caption": "An orange kitten threads through green foliage like a cache "
                    "finally serving the right page."
                })
            return json.dumps({
                "caption": "An orange kitten walks through green foliage, apparently very busy."
            })

        ensemble._call = fake_call
        result = _run_with_frame(ensemble)

        assert tech_calls == 2
        assert result["humorous_tech"] == (
            "An orange kitten threads through green foliage like a cache finally serving "
            "the right page."
        )


def test_humorous_tech_selector_keeps_the_best_grounded_candidate() -> None:
    with _loaded_ensemble("1", "1") as ensemble:
        base_system = ensemble._writer_system()
        tech_writer_calls = 0
        selector_calls = 0

        async def fake_call(
            client, model, system, content, max_tokens, temperature=0.5
        ) -> str:
            nonlocal tech_writer_calls, selector_calls
            if system == ensemble.OBSERVE_SYSTEM:
                return '["An orange kitten walks through green foliage toward the camera."]'
            if system == ensemble._TECH_SELECTOR_SYSTEM:
                selector_calls += 1
                assert "CANDIDATE A:" in content
                assert "CANDIDATE B:" in content
                return '{"winner":"B"}'

            suffix = system[len(base_system):]
            style = next(style for style in _STYLES if f'"{style}"' in suffix)
            if style != "humorous_tech":
                return json.dumps({"caption": _writer_caption("valid", style)})
            tech_writer_calls += 1
            if system.endswith(ensemble._TECH_ALTERNATE_CANDIDATE_RULE):
                return json.dumps({
                    "caption": "An orange kitten walks toward the camera through green foliage, "
                    "its tail acting like a runtime meter finally reaching full strength."
                })
            return json.dumps({
                "caption": "An orange kitten walks through green foliage toward the camera like "
                "a server doing its job."
            })

        ensemble._call = fake_call
        result = _run_with_frame(ensemble)

        assert tech_writer_calls == 2
        assert selector_calls == 1
        assert result["humorous_tech"] == (
            "An orange kitten walks toward the camera through green foliage, its tail acting "
            "like a runtime meter finally reaching full strength."
        )


def test_flag_is_strict_and_off_unless_explicitly_enabled() -> None:
    for value in (None, "", "0", "false", "False", "off", "no", " 0 "):
        with _loaded_ensemble(value) as ensemble:
            assert ensemble.W4_STYLE_SPLIT is False, value

    for value in ("1", "true", "True", "on", "yes", " 1 "):
        with _loaded_ensemble(value) as ensemble:
            assert ensemble.W4_STYLE_SPLIT is True, value

    try:
        with _loaded_ensemble("enable-maybe"):
            pass
    except ValueError as error:
        assert _FLAG in str(error)
    else:
        raise AssertionError("invalid W4 style-split flag was accepted")


def main() -> None:
    test_empty_observer_content_is_skipped()
    test_default_off_preserves_the_v38_common_writer()
    test_on_runs_four_style_writers_concurrently_with_one_observation_spine()
    test_each_style_keeps_the_existing_single_retry_contract()
    test_each_style_retries_malformed_writer_json()
    test_humorous_tech_gets_one_grounded_repair_before_outer_fallback()
    test_humorous_tech_selector_keeps_the_best_grounded_candidate()
    test_flag_is_strict_and_off_unless_explicitly_enabled()
    print("w4_style_split_ok")


if __name__ == "__main__":
    main()
