"""Offline contract tests for the opt-in direct Qwen caption engine.

Run:
    PYTHONPATH=. python scripts/test_qwen_direct.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sys
import tempfile
import types
from pathlib import Path

import httpx

sys.path.insert(0, ".")

from app import main as M  # noqa: E402
from app import pipeline as P  # noqa: E402
from app import qwen_direct as Q  # noqa: E402


STYLES = [
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
]


def _frames(workdir: Path) -> list[Path]:
    frames: list[Path] = []
    for index in range(1, 5):
        frame = workdir / f"frame_{index}.jpg"
        frame.write_bytes(f"jpeg-{index}".encode("ascii"))
        frames.append(frame)
    return frames


def test_profile_matches_the_predeclared_direct_candidate() -> None:
    assert Q.FRAME_COUNT == 4
    assert Q.FRAME_MAX_EDGE == 1024
    assert Q.MODEL == "accounts/fireworks/models/qwen3p7-plus"
    assert Q.TEMPERATURE == 0.7
    assert Q.MAX_TOKENS == 400
    assert Q.REASONING_EFFORT == "none"
    assert Q.MAX_ATTEMPTS == 3
    assert Q.STYLE_CONCURRENCY == 1


def test_build_request_contains_four_images_and_one_style_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frames = _frames(Path(tmp))
        payload = Q.build_request("formal", frames)

    assert payload["model"] == Q.MODEL
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 400
    assert payload["reasoning_effort"] == "none"
    assert len(payload["messages"]) == 2
    system = payload["messages"][0]["content"]
    assert "professional" in system.lower()
    assert "objective" in system.lower()
    assert "sarcastic" not in system.lower()
    assert "cross-reference" not in system.lower()
    assert "observation list" not in system.lower()

    content = payload["messages"][1]["content"]
    image_blocks = [part for part in content if part["type"] == "image_url"]
    text_blocks = [part for part in content if part["type"] == "text"]
    assert len(image_blocks) == 4
    assert len(text_blocks) == 1
    assert "one caption" in text_blocks[0]["text"].lower()
    for index, part in enumerate(image_blocks, start=1):
        url = part["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        encoded = url.split(",", 1)[1]
        assert base64.b64decode(encoded) == f"jpeg-{index}".encode("ascii")


def test_v1_prompt_profile_remains_byte_stable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frames = _frames(Path(tmp))
        system = Q.build_request(
            "formal", frames, prompt_profile="v1"
        )["messages"][0]["content"]
    assert hashlib.sha256(system.encode("utf-8")).hexdigest() == (
        "f6f6f4e96070165c817335a2cbfc3713d8a142b44cb94940f8379cba6208ec51"
    )


def test_strong_v2_has_short_safe_core_and_distinct_style_signals() -> None:
    systems: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        frames = _frames(Path(tmp))
        for style in STYLES:
            payload = Q.build_request(
                style, frames, prompt_profile="strong_v2"
            )
            systems[style] = payload["messages"][0]["content"].lower()
            assert payload["temperature"] == Q.TEMPERATURE
            assert payload["max_tokens"] == Q.MAX_TOKENS
            assert payload["reasoning_effort"] == Q.REASONING_EFFORT

    for system in systems.values():
        assert "exactly one complete sentence" in system
        assert "18-32 words" in system
        assert "main subject" in system
        assert "main directly visible action" in system
        assert "safe general setting" in system
        assert "chronology" in system
        assert "clothing" in system
        assert "colors" in system
        assert "unless unmistakably" in system

    assert "professional" in systems["formal"]
    assert "objective" in systems["formal"]
    assert "no humor" in systems["formal"]
    assert "unmistakable dry irony" in systems["sarcastic"]
    assert "not merely factual" in systems["sarcastic"]
    assert "no technology jargon" in systems["sarcastic"]
    assert "one explicit technology or programming metaphor" in systems[
        "humorous_tech"
    ]
    assert "obvious everyday joke" in systems["humorous_non_tech"]
    assert "no technical jargon" in systems["humorous_non_tech"]


def test_global_prompt_profile_selects_strong_v2_without_changing_v1() -> None:
    original = Q.PROMPT_PROFILE
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = _frames(Path(tmp))
            v1 = Q.build_request("sarcastic", frames, prompt_profile="v1")
            Q.PROMPT_PROFILE = "strong_v2"
            selected = Q.build_request("sarcastic", frames)
    finally:
        Q.PROMPT_PROFILE = original

    assert selected["messages"][0]["content"] != v1["messages"][0]["content"]
    assert "unmistakable dry irony" in selected["messages"][0]["content"].lower()


def test_unknown_prompt_profile_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frames = _frames(Path(tmp))
        try:
            Q.build_request("formal", frames, prompt_profile="typo")
        except ValueError as error:
            assert "unknown qwen direct prompt profile" in str(error).lower()
        else:
            raise AssertionError("unknown prompt profile was silently accepted")


def test_four_independent_style_calls_preserve_requested_keys() -> None:
    requests: list[dict] = []

    async def requester(payload: dict) -> str:
        requests.append(payload)
        return f"Caption response {len(requests)}."

    with tempfile.TemporaryDirectory() as tmp:
        captions = asyncio.run(
            Q.caption_styles_from_frames(
                _frames(Path(tmp)),
                STYLES,
                requester=requester,
                retry_delay_s=0,
            )
        )

    assert list(captions) == STYLES
    assert len(requests) == 4
    assert all(captions[style] for style in STYLES)
    assert len({request["messages"][0]["content"] for request in requests}) == 4
    assert all(
        len(
            [
                part
                for part in request["messages"][1]["content"]
                if part["type"] == "image_url"
            ]
        )
        == 4
        for request in requests
    )


def test_style_calls_share_one_default_inflight_slot() -> None:
    current = 0
    peak = 0

    async def requester(_payload: dict) -> str:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return "A grounded caption."

    original_concurrency = Q.STYLE_CONCURRENCY
    try:
        Q.STYLE_CONCURRENCY = 1
        with tempfile.TemporaryDirectory() as tmp:
            captions = asyncio.run(
                Q.caption_styles_from_frames(
                    _frames(Path(tmp)),
                    STYLES,
                    requester=requester,
                    retry_delay_s=0,
                )
            )
    finally:
        Q.STYLE_CONCURRENCY = original_concurrency

    assert peak == 1
    assert list(captions) == STYLES


def test_style_limit_is_shared_across_concurrent_videos() -> None:
    current = 0
    peak = 0

    async def requester(_payload: dict) -> str:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return "A grounded caption."

    async def run_two(frames: list[Path]) -> None:
        await asyncio.gather(
            Q.caption_styles_from_frames(
                frames,
                ["formal"],
                requester=requester,
                retry_delay_s=0,
            ),
            Q.caption_styles_from_frames(
                frames,
                ["sarcastic"],
                requester=requester,
                retry_delay_s=0,
            ),
        )

    original_concurrency = Q.STYLE_CONCURRENCY
    try:
        Q.STYLE_CONCURRENCY = 1
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(run_two(_frames(Path(tmp))))
    finally:
        Q.STYLE_CONCURRENCY = original_concurrency

    assert peak == 1


def _http_429(retry_after: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.test/chat/completions")
    response = httpx.Response(
        429,
        headers={"Retry-After": retry_after},
        request=request,
    )
    return httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=response,
    )


def test_429_retries_three_times_and_honors_retry_after() -> None:
    attempts = 0
    sleeps: list[float] = []
    original_sleep = Q.asyncio.sleep
    original_uniform = Q.random.uniform
    try:
        async def requester(_payload: dict) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise _http_429("0.5")
            return "A grounded caption."

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        Q.asyncio.sleep = fake_sleep
        Q.random.uniform = lambda lower, upper: upper
        with tempfile.TemporaryDirectory() as tmp:
            captions = asyncio.run(
                Q.caption_styles_from_frames(
                    _frames(Path(tmp)),
                    ["formal"],
                    requester=requester,
                    retry_delay_s=0.1,
                )
            )
    finally:
        Q.asyncio.sleep = original_sleep
        Q.random.uniform = original_uniform

    assert attempts == 3
    assert captions == {"formal": "A grounded caption."}
    assert len(sleeps) == 2
    assert all(0.5 < delay <= Q.RETRY_MAX_DELAY_S for delay in sleeps)


def test_long_retry_after_falls_back_without_retrying_too_early() -> None:
    attempts = 0
    sleeps: list[float] = []
    original_sleep = Q.asyncio.sleep
    try:
        async def requester(_payload: dict) -> str:
            nonlocal attempts
            attempts += 1
            raise _http_429("600")

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        Q.asyncio.sleep = fake_sleep
        with tempfile.TemporaryDirectory() as tmp:
            captions = asyncio.run(
                Q.caption_styles_from_frames(
                    _frames(Path(tmp)),
                    ["formal"],
                    requester=requester,
                )
            )
    finally:
        Q.asyncio.sleep = original_sleep

    assert attempts == 1
    assert sleeps == []
    assert captions == {"formal": ""}


def test_exponential_jitter_is_capped() -> None:
    original_uniform = Q.random.uniform
    try:
        Q.random.uniform = lambda lower, upper: upper
        delay = Q._retry_delay_seconds(
            httpx.TransportError("temporary"),
            attempt=20,
            base_delay_s=0.75,
        )
    finally:
        Q.random.uniform = original_uniform
    assert 0 < delay <= Q.RETRY_MAX_DELAY_S


def test_transient_failure_retries_only_the_failed_style() -> None:
    attempts: dict[str, int] = {}

    async def requester(payload: dict) -> str:
        system = payload["messages"][0]["content"]
        attempts[system] = attempts.get(system, 0) + 1
        if "dry" in system.lower() and attempts[system] == 1:
            raise httpx.TransportError("temporary transport failure")
        return "A grounded caption."

    with tempfile.TemporaryDirectory() as tmp:
        captions = asyncio.run(
            Q.caption_styles_from_frames(
                _frames(Path(tmp)),
                STYLES,
                requester=requester,
                retry_delay_s=0,
            )
        )

    assert captions["sarcastic"] == "A grounded caption."
    assert sorted(attempts.values()) == [1, 1, 1, 2]


def test_exhausted_style_returns_empty_for_pipeline_fallback() -> None:
    calls = 0

    async def requester(_payload: dict) -> str:
        nonlocal calls
        calls += 1
        raise httpx.TransportError("still unavailable")

    with tempfile.TemporaryDirectory() as tmp:
        captions = asyncio.run(
            Q.caption_styles_from_frames(
                _frames(Path(tmp)),
                ["formal"],
                requester=requester,
                retry_delay_s=0,
            )
        )

    assert captions == {"formal": ""}
    assert calls == 3


def test_extract_profile_uses_the_confirmed_fps_geometry() -> None:
    captured: dict = {}
    original_probe = P._ffprobe_duration
    original_extract = getattr(Q, "_extract_fps_frames", None)
    try:
        P._ffprobe_duration = lambda _video: 80.0

        def fake_extract(**kwargs):
            captured.update(kwargs)
            return _frames(kwargs["workdir"])

        Q._extract_fps_frames = fake_extract
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            video = workdir / "clip.mp4"
            video.write_bytes(b"video")
            frames = Q.extract_frames(video, workdir)
    finally:
        P._ffprobe_duration = original_probe
        if original_extract is None:
            delattr(Q, "_extract_fps_frames")
        else:
            Q._extract_fps_frames = original_extract

    assert len(frames) == 4
    assert captured == {
        "video": video,
        "workdir": workdir,
        "duration": 80.0,
    }


def test_fps_extractor_runs_one_four_frame_1024px_ffmpeg_filter() -> None:
    calls: list[tuple[list[str], dict]] = []
    original_run = Q.subprocess.run
    try:
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            pattern = Path(command[-1])
            for index in range(1, 5):
                Path(str(pattern).replace("%02d", f"{index:02d}")).write_bytes(
                    f"jpeg-{index}".encode("ascii")
                )

        Q.subprocess.run = fake_run
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            video = workdir / "clip.mp4"
            video.write_bytes(b"video")
            frames = Q._extract_fps_frames(video, workdir, duration=80.0)
    finally:
        Q.subprocess.run = original_run

    assert len(frames) == 4
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:6] == [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i"
    ]
    assert command[6] == str(video)
    vf = command[command.index("-vf") + 1]
    assert vf.startswith("fps=0.05000000,")
    assert "min(1024,iw)" in vf
    assert "min(1024,ih)" in vf
    assert "force_original_aspect_ratio=decrease" in vf
    assert command[command.index("-vframes") + 1] == "4"
    assert command.index("-vf") < command.index("-vframes")
    assert kwargs == {"check": True, "timeout": 12.0}


def test_missing_direct_caption_uses_existing_pipeline_once() -> None:
    original_key = Q.FIREWORKS_API_KEY
    original_download = P._download
    original_extract = Q.extract_frames
    original_caption_styles = Q.caption_styles_from_frames
    original_pipeline = P.caption_one_video
    fallback_calls: list[list[str]] = []
    try:
        Q.FIREWORKS_API_KEY = "test-key"

        async def fake_download(_url: str, dst: Path) -> Path:
            dst.write_bytes(b"video")
            return dst

        def fake_extract(_video: Path, workdir: Path) -> list[Path]:
            return _frames(workdir)

        async def fake_direct(_frames_arg, styles, **_kwargs):
            return {
                "formal": "A person walks through a quiet park.",
                "sarcastic": "",
                "humorous_tech": "The walking pipeline completes its tiny runtime.",
                "humorous_non_tech": "The walking API deploys a cheerful stroll.",
            }

        async def fake_pipeline(video_url: str, styles: list[str]):
            assert video_url == "https://example.test/v.mp4"
            fallback_calls.append(styles)
            return {
                "sarcastic": "Clearly, this walk has reached historic importance.",
                "humorous_tech": "The walking pipeline completes its tiny runtime.",
                "humorous_non_tech": "A stroll enters like it booked the whole path.",
            }

        P._download = fake_download
        Q.extract_frames = fake_extract
        Q.caption_styles_from_frames = fake_direct
        P.caption_one_video = fake_pipeline
        captions = asyncio.run(Q.caption_qwen_direct("https://example.test/v.mp4", STYLES))
    finally:
        Q.FIREWORKS_API_KEY = original_key
        P._download = original_download
        Q.extract_frames = original_extract
        Q.caption_styles_from_frames = original_caption_styles
        P.caption_one_video = original_pipeline

    assert fallback_calls == [["sarcastic", "humorous_non_tech"]]
    assert list(captions) == STYLES
    assert all(captions.values())


def test_main_dispatches_only_when_explicitly_selected() -> None:
    original_engine = M.CAPTION_ENGINE
    original_pipeline = M.caption_one_video
    original_module = sys.modules.get("app.qwen_direct")
    called: list[tuple[str, list[str]]] = []
    fake_module = types.ModuleType("app.qwen_direct")

    async def fake_direct(video_url: str, styles: list[str]) -> dict[str, str]:
        called.append((video_url, styles))
        return {
            "formal": "A person walks through a quiet park.",
            "sarcastic": "Clearly, this walk has reached historic importance.",
            "humorous_tech": "The walking pipeline completes its tiny runtime.",
            "humorous_non_tech": "A stroll enters like it booked the whole path.",
        }

    async def forbidden_pipeline(**_kwargs):
        raise AssertionError("legacy pipeline selected for qwen_direct")

    fake_module.caption_qwen_direct = fake_direct
    try:
        sys.modules["app.qwen_direct"] = fake_module
        M.CAPTION_ENGINE = "qwen_direct"
        M.caption_one_video = forbidden_pipeline
        result = asyncio.run(
            M._run_one(
                asyncio.Semaphore(1),
                {
                    "task_id": "v1",
                    "video_url": "https://example.test/v.mp4",
                    "styles": STYLES,
                },
            )
        )
    finally:
        M.CAPTION_ENGINE = original_engine
        M.caption_one_video = original_pipeline
        if original_module is None:
            sys.modules.pop("app.qwen_direct", None)
        else:
            sys.modules["app.qwen_direct"] = original_module

    assert called == [("https://example.test/v.mp4", STYLES)]
    assert result["task_id"] == "v1"
    assert list(result["captions"]) == STYLES


def test_docker_default_is_untouched() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "CAPTION_ENGINE=ensemble" in dockerfile
    assert "CAPTION_ENGINE=qwen_direct" not in dockerfile


def main() -> None:
    test_profile_matches_the_predeclared_direct_candidate()
    test_build_request_contains_four_images_and_one_style_only()
    test_v1_prompt_profile_remains_byte_stable()
    test_strong_v2_has_short_safe_core_and_distinct_style_signals()
    test_global_prompt_profile_selects_strong_v2_without_changing_v1()
    test_unknown_prompt_profile_fails_closed()
    test_four_independent_style_calls_preserve_requested_keys()
    test_style_calls_share_one_default_inflight_slot()
    test_style_limit_is_shared_across_concurrent_videos()
    test_429_retries_three_times_and_honors_retry_after()
    test_long_retry_after_falls_back_without_retrying_too_early()
    test_exponential_jitter_is_capped()
    test_transient_failure_retries_only_the_failed_style()
    test_exhausted_style_returns_empty_for_pipeline_fallback()
    test_extract_profile_uses_the_confirmed_fps_geometry()
    test_fps_extractor_runs_one_four_frame_1024px_ffmpeg_filter()
    test_missing_direct_caption_uses_existing_pipeline_once()
    test_main_dispatches_only_when_explicitly_selected()
    test_docker_default_is_untouched()
    print("qwen_direct_ok")


if __name__ == "__main__":
    main()
