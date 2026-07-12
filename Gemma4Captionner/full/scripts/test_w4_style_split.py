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
    "VIDEO_OBSERVER": "",
    "OPENROUTER_API_KEY": "w4-offline-test-secret",
}
_V38_DOCKER_WRITER_SHA256 = (
    "863a76829b72e10b8d842c3fe0c38edae1f63fecb7f0149a2528b4885ad825b6"
)


@contextmanager
def _loaded_ensemble(flag: str | None) -> Iterator[object]:
    names = (*_CONTROLLED_ENV, _FLAG)
    before = {name: os.environ.get(name) for name in names}
    existed = {name: name in os.environ for name in names}
    try:
        os.environ.update(_CONTROLLED_ENV)
        if flag is None:
            os.environ.pop(_FLAG, None)
        else:
            os.environ[_FLAG] = flag

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


def test_default_off_preserves_the_v38_common_writer() -> None:
    with _loaded_ensemble(None) as ensemble:
        assert ensemble.W4_STYLE_SPLIT is False
        base_system = ensemble._writer_system()
        assert hashlib.sha256(base_system.encode("utf-8")).hexdigest() == (
            _V38_DOCKER_WRITER_SHA256
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


def test_on_runs_four_style_writers_concurrently_with_one_observation_spine() -> None:
    with _loaded_ensemble("1") as ensemble:
        assert ensemble.W4_STYLE_SPLIT is True
        base_system = ensemble._writer_system()
        assert hashlib.sha256(base_system.encode("utf-8")).hexdigest() == (
            _V38_DOCKER_WRITER_SHA256
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
            return json.dumps({"caption": f"split {style}"})

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
        assert result == {style: f"split {style}" for style in _STYLES}
        validated = validate_results([{"task_id": "w4", "captions": result}])
        assert validated[0]["captions"] == result


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
            return json.dumps({"caption": f"retry-safe {style}"})

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
        assert all(result[style] == f"retry-safe {style}" for style in _STYLES)


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
            return json.dumps({"caption": f"parse-safe {style}"})

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
        assert result == {style: f"parse-safe {style}" for style in _STYLES}


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
    test_default_off_preserves_the_v38_common_writer()
    test_on_runs_four_style_writers_concurrently_with_one_observation_spine()
    test_each_style_keeps_the_existing_single_retry_contract()
    test_each_style_retries_malformed_writer_json()
    test_flag_is_strict_and_off_unless_explicitly_enabled()
    print("w4_style_split_ok")


if __name__ == "__main__":
    main()
