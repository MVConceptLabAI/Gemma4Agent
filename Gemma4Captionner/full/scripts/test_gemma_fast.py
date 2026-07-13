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

    microphone = gemma_fast._post_validate({
        "formal": "A man wearing a blue polo shirt speaks into a vintage-style silver microphone in an indoor setting.",
        "sarcastic": "The visible action receives the gravity of an event the history books nearly missed.",
        "humorous_tech": "His setup is like a legacy hardware driver on a modern OS; it is outdated but still functioning.",
        "humorous_non_tech": "He treats the microphone like a fancy milkshake, leaning in for a sip of fame.",
    }, list(gemma_fast.REQUIRED_STYLES))
    assert "national broadcast" in microphone["sarcastic"].lower()
    assert "legacy" not in microphone["humorous_tech"].lower()
    assert "outdated" not in microphone["humorous_tech"].lower()
    assert "milkshake" in microphone["humorous_non_tech"].lower()

    malformed = (
        'formal: "A cat walks through leaves.", '
        '"sarcastic":"A cat conducts a very serious garden inspection.", '
        '"humorous_tech":"A cat navigates the leaves like a tiny robot mapping terrain.", '
        '"humorous_non_tech":"A cat walks through the leaves like a landlord checking the garden."'
    )
    assert gemma_fast._parse_local(malformed)["formal"] == "A cat walks through leaves."

    battery = gemma_fast._post_validate({
        "formal": "An animation shows lithium ions moving between battery electrodes as a light bulb illuminates.",
        "sarcastic": "The battery moves ions around with all the ceremony required to illuminate one bulb.",
        "humorous_tech": "The ions move like packets of data in a flood of packets through a server.",
        "humorous_non_tech": "The small visible moment carries on like it is delighted to have an audience.",
    }, list(gemma_fast.REQUIRED_STYLES))
    assert "packets" not in battery["humorous_tech"].lower()
    assert "ions" in battery["humorous_tech"].lower()
    assert "software thread" not in battery["humorous_tech"].lower()
    assert "commuters" in battery["humorous_non_tech"].lower()
    assert "burn out" not in gemma_fast._post_validate({
        **battery,
        "sarcastic": "The battery powers a bulb that still manages to burn out in the end.",
    }, list(gemma_fast.REQUIRED_STYLES))["sarcastic"].lower()

    sports = gemma_fast._fallbacks(
        "An aerial view shows players in athletic wear on a grassy field with two goals."
    )
    assert "field" in sports["humorous_tech"].lower()
    assert "track" not in sports["humorous_tech"].lower()
    assert "software" in sports["humorous_tech"].lower()

    dance = gemma_fast._post_validate({
        "formal": "A man performs a dance routine on a wooden floor in a large room.",
        "sarcastic": "The dancer gives the empty room all the ceremony it requested.",
        "humorous_tech": "The dancer moves like software executing expressive code.",
        "humorous_non_tech": "The small visible moment carries on like it is delighted to have an audience.",
    }, list(gemma_fast.REQUIRED_STYLES))
    assert "dancer" in dance["humorous_non_tech"].lower()
    assert "small visible moment" not in dance["humorous_non_tech"].lower()
    assert "software thread" not in dance["humorous_tech"].lower()

    repeated = gemma_fast._post_validate({
        "formal": "A dancer performs alone in a large room.",
        "sarcastic": "The dancer uses the room with admirable confidence.",
        "humorous_tech": "The dancer moves like a single thread executing code without server support.",
        "humorous_non_tech": "He dances like a man who left the oven on at home.",
    }, list(gemma_fast.REQUIRED_STYLES))
    assert "single thread" not in repeated["humorous_tech"].lower()
    assert "oven on at home" not in repeated["humorous_non_tech"].lower()

    assert gemma_fast._contradicts_formal(
        "A black car drives along a city road and through a tunnel.",
        "A high-speed chase races through the tunnel.",
    )
    assert gemma_fast._contradicts_formal(
        "Vehicles travel on a road through a tunnel and under overpasses.",
        "The cars move like confused tourists searching for a hidden landmark.",
    )


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
