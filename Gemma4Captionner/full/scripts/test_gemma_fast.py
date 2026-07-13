"""Offline regression tests for the V19 single-call engine."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import httpx

from app import gemma_fast


def test_local_json_and_validation() -> None:
    raw = """```json
    {"formal":"A runner crosses a red track in a stadium.",
     "sarcastic":"A runner crosses the track, a sweeping epic of athletic administration.",
     "humorous_tech":"A runner crosses the track like a packet racing through a low-latency network.",
     "humorous_non_tech":"A runner crosses the track like someone who just remembered the oven."}
    ```"""
    parsed = gemma_fast._parse_local(raw)
    assert set(parsed) == set(gemma_fast.REQUIRED_STYLES)
    checked = gemma_fast._post_validate(parsed, list(gemma_fast.REQUIRED_STYLES))
    assert "sweeping epic" not in checked["sarcastic"].lower()
    assert gemma_fast._has_tech_reference(checked["humorous_tech"])
    assert len({caption.casefold() for caption in checked.values()}) == 4

    risky = dict(parsed)
    risky["formal"] = "A high-angle time-lapse captures cars moving along a city street."
    risky["sarcastic"] = "Someone is having a thrilling time typing at an office desk."
    repaired = gemma_fast._post_validate(risky, list(gemma_fast.REQUIRED_STYLES))
    assert "time-lapse" not in repaired["formal"].lower()
    assert "thrilling time" not in repaired["sarcastic"].lower()
    fallback_a = gemma_fast._fallbacks("A dog runs across a grassy field under a bright sky.")
    fallback_b = gemma_fast._fallbacks("A woman types on a keyboard in an office setting.")
    assert fallback_a["humorous_tech"] != fallback_b["humorous_tech"]
    assert not fallback_a["humorous_tech"].endswith("in, running like.")
    assert gemma_fast._contradicts_formal(
        "Hands type on a laptop in a brightly lit room.",
        "Someone sends another email from a dimly lit room.",
    )
    assert gemma_fast._contradicts_formal(
        "Three people sit around a table with drinks.",
        "They hold glasses of beer.",
    )
    assert gemma_fast._contradicts_formal(
        "A nighttime view from a vehicle shows a wet street.",
        "It is a pleasant midnight stroll.",
    )
    musical = dict(parsed)
    musical["humorous_non_tech"] = "The intersection looks like a giant game of musical chairs."
    assert "musical chairs" not in gemma_fast._post_validate(
        musical, list(gemma_fast.REQUIRED_STYLES)
    )["humorous_non_tech"].lower()

    meta = gemma_fast._post_validate({
        "formal": "A close-up shows a hand typing on a laptop keyboard and using the trackpad.",
        "sarcastic": "The keyboard receives the attention normally reserved for a national emergency.",
        "humorous_tech": "The visual runtime keeps objects in a queue while QA asks for one cleaner punchline.",
        "humorous_non_tech": "The fingers dance across the keys like a pianist playing for one screen.",
    }, list(gemma_fast.REQUIRED_STYLES))
    assert "qa" not in meta["humorous_tech"].lower()
    assert "punchline" not in meta["humorous_tech"].lower()
    assert "here," not in meta["humorous_tech"].lower()

    racing = gemma_fast._post_validate({
        "formal": "A woman presents a racing event, speaks with a man beside a red race car, sits in a race car, and the car drives onto the circuit.",
        "sarcastic": "A woman spends her day near expensive red cars, because that is a very demanding way to spend an afternoon.",
        "humorous_tech": "The race sequence runs like software promoting its fastest process into production.",
        "humorous_non_tech": "She looks great, but the red cars are stealing the spotlight while the driver just wants to go fast.",
    }, list(gemma_fast.REQUIRED_STYLES))
    assert "demanding way" not in racing["sarcastic"].lower()
    assert "looks great" not in racing["humorous_non_tech"].lower()
    assert "stealing the spotlight" not in racing["humorous_non_tech"].lower()
    assert "window-shopping" in racing["humorous_non_tech"].lower()

    malformed = (
        'formal: "A cat walks through leaves.", '
        '"sarcastic":"A cat conducts a very serious garden inspection.", '
        '"humorous_tech":"A cat navigates the leaves like a tiny robot mapping terrain.", '
        '"humorous_non_tech":"A cat walks through the leaves like a landlord checking the garden."'
    )
    assert gemma_fast._parse_local(malformed)["formal"] == "A cat walks through leaves."


async def test_exactly_one_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert len(body["messages"][0]["content"]) == 25  # prompt + 24 frames
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"formal":"ok"}'}}]
        })

    old_key = gemma_fast.API_KEY
    gemma_fast.API_KEY = "test-key"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            for index in range(24):
                frame = Path(tmp) / f"{index}.jpg"
                frame.write_bytes(b"jpeg")
                frames.append(frame)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await gemma_fast._one_gemma_call(client, frames)
    finally:
        gemma_fast.API_KEY = old_key
    assert calls == 1


def main() -> None:
    test_local_json_and_validation()
    asyncio.run(test_exactly_one_request())
    print("Gemma Fast V19 regressions: ok")


if __name__ == "__main__":
    main()
