"""Offline regressions for the V20 hybrid routing and batch guard."""

from __future__ import annotations

import asyncio

from app import gemma_hybrid as hybrid


def captions(subject: str = "A dog runs across a grassy field") -> dict[str, str]:
    return {
        "formal": f"{subject} during a clear daytime outdoor scene.",
        "sarcastic": f"{subject}, delivering the solemn importance this ordinary outing clearly required.",
        "humorous_tech": f"{subject} like software clearing a processor queue before the server notices.",
        "humorous_non_tech": f"{subject} like a late guest who finally spotted the picnic table.",
    }


async def test_fast_acceptance_and_recovery() -> None:
    original_fast = hybrid.caption_gemma_fast
    original_demo = hybrid.caption_demo
    original_timeout = hybrid.FAST_TIMEOUT_S
    calls = {"fast": 0, "demo": 0}

    async def good_fast(video_url: str, styles: list[str]) -> dict[str, str]:
        calls["fast"] += 1
        return captions()

    async def good_demo(video_url: str, styles: list[str]) -> dict[str, str]:
        calls["demo"] += 1
        return captions("A black dog crosses a field")

    try:
        hybrid.caption_gemma_fast = good_fast
        hybrid.caption_demo = good_demo
        result = await hybrid.caption_gemma_hybrid("https://example.test/video.mp4", list(hybrid.REQUIRED_STYLES))
        assert result["formal"].startswith("A dog runs")
        assert calls == {"fast": 1, "demo": 0}

        async def leaked_fast(video_url: str, styles: list[str]) -> dict[str, str]:
            value = captions()
            value["humorous_tech"] = "{'fact': 'A dog runs across a field'}, like software clearing a server queue."
            return value

        hybrid.caption_gemma_fast = leaked_fast
        result = await hybrid.caption_gemma_hybrid("https://example.test/video.mp4", list(hybrid.REQUIRED_STYLES))
        assert result["formal"].startswith("A black dog")
        assert calls["demo"] == 1

        async def failed_demo(video_url: str, styles: list[str]) -> dict[str, str]:
            calls["demo"] += 1
            raise asyncio.TimeoutError("simulated recovery timeout")

        async def pricey_fast(video_url: str, styles: list[str]) -> dict[str, str]:
            value = captions("A black car drives along a multi-lane city road")
            value["sarcastic"] = "The car proves that spending a fortune still earns the same concrete view."
            return value

        hybrid.caption_gemma_fast = pricey_fast
        hybrid.caption_demo = failed_demo
        result = await hybrid.caption_gemma_hybrid("https://example.test/video.mp4", list(hybrid.REQUIRED_STYLES))
        assert "spending a fortune" not in result["sarcastic"]
        assert "black car drives along a multi-lane city road" in result["sarcastic"].lower()
        assert calls["demo"] == 2
        hybrid.caption_demo = good_demo

        async def slow_fast(video_url: str, styles: list[str]) -> dict[str, str]:
            await asyncio.sleep(0.05)
            return captions()

        hybrid.caption_gemma_fast = slow_fast
        hybrid.FAST_TIMEOUT_S = 0.01
        result = await hybrid.caption_gemma_hybrid("https://example.test/video.mp4", list(hybrid.REQUIRED_STYLES))
        assert result["formal"].startswith("A black dog")
        assert calls["demo"] == 3
    finally:
        hybrid.caption_gemma_fast = original_fast
        hybrid.caption_demo = original_demo
        hybrid.FAST_TIMEOUT_S = original_timeout


async def test_recovery_keeps_evidence_first_creative_copy() -> None:
    original_demo = hybrid.caption_demo

    async def rich_demo(video_url: str, styles: list[str]) -> dict[str, str]:
        return {
            "formal": "Cars and motorcycles drive beneath concrete overpasses and through a tunnel.",
            "sarcastic": "Cars and motorcycles pass beneath concrete overpasses, because apparently a tunnel commute deserved this much architectural suspense.",
            "humorous_tech": "Cars and motorcycles route through concrete tunnels like data packets following a very determined network path.",
            "humorous_non_tech": "Cars and motorcycles squeeze through the concrete tunnel like guests leaving a party through one determined doorway.",
        }

    try:
        hybrid.caption_demo = rich_demo
        result = await hybrid._recover(
            "https://example.test/video.mp4",
            list(hybrid.REQUIRED_STYLES),
            10,
            "test",
        )
        assert "architectural suspense" in result["sarcastic"]
        assert "data packets" in result["humorous_tech"]
        assert "one determined doorway" in result["humorous_non_tech"]
    finally:
        hybrid.caption_demo = original_demo


async def test_batch_repetition_recovery() -> None:
    first = captions("A dog runs across a grassy field")
    second = captions("Players run across a football field")
    second["humorous_tech"] = first["humorous_tech"]
    results = [
        {"task_id": "animals", "captions": first},
        {"task_id": "sports", "captions": second},
    ]
    assert hybrid.repeated_task_ids(results) == ["sports"]

    original_recover = hybrid._recover

    async def fake_recover(video_url: str, styles: list[str], timeout: float, reason: str) -> dict[str, str]:
        assert reason == "batch repetition"
        return captions("Football players chase a ball on a field")

    try:
        hybrid._recover = fake_recover
        repaired = await hybrid.recover_batch_repetitions(
            [
                {"task_id": "animals", "video_url": "https://example.test/a.mp4", "styles": list(hybrid.REQUIRED_STYLES)},
                {"task_id": "sports", "video_url": "https://example.test/b.mp4", "styles": list(hybrid.REQUIRED_STYLES)},
            ],
            results,
            lambda: 300.0,
        )
        assert repaired[1]["captions"]["formal"].startswith("Football players")
    finally:
        hybrid._recover = original_recover


def test_quality_signals() -> None:
    risky = captions("Food cooks in sealed bags in a water bath")
    risky["humorous_tech"] = "A distant skyline behaves like software processing city data through a server."
    reasons = hybrid.caption_risks(risky, list(hybrid.REQUIRED_STYLES))
    assert "humorous_tech:unsupported-scene-noun" in reasons

    urban = captions("A black car drives along a multi-lane city road")
    urban["sarcastic"] = "The car proves that spending a fortune still earns the same concrete view."
    reasons = hybrid.caption_risks(urban, list(hybrid.REQUIRED_STYLES))
    assert "sarcastic:unsupported-value-claim" in reasons

    race = captions("A red race car drives onto a circuit")
    race["sarcastic"] = "The expensive racing machine finally performs the radical act of entering the circuit."
    reasons = hybrid.caption_risks(race, list(hybrid.REQUIRED_STYLES))
    assert "sarcastic:unsupported-value-claim" not in reasons

    generic = captions("A black car drives along a multi-lane city road, under an overpass and through a tunnel")
    generic["humorous_tech"] = (
        "The traffic flow behaves like a poorly managed server, where every car is a request waiting for the processor."
    )
    generic["humorous_non_tech"] = (
        "This commute is like a slow-motion parade where the only prize is another traffic jam."
    )
    reasons = hybrid.caption_risks(generic, list(hybrid.REQUIRED_STYLES))
    assert "humorous_tech:weak-formal-grounding" in reasons
    assert "humorous_non_tech:weak-formal-grounding" in reasons
    repaired = hybrid._repair_fast_after_failed_recovery(generic, list(hybrid.REQUIRED_STYLES), reasons)
    assert "black car drives along a multi-lane city road" in repaired["humorous_tech"].lower()
    assert "black car drives along a multi-lane city road" in repaired["humorous_non_tech"].lower()


def main() -> None:
    test_quality_signals()
    asyncio.run(test_fast_acceptance_and_recovery())
    asyncio.run(test_recovery_keeps_evidence_first_creative_copy())
    asyncio.run(test_batch_repetition_recovery())
    print("Gemma Hybrid V20 regressions: ok")


if __name__ == "__main__":
    main()
