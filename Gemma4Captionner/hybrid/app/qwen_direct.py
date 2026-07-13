"""Opt-in direct Qwen vision caption engine.

The engine deliberately keeps the visual path shallow: four chronological
frames are sent directly to one independent multimodal request per requested
style. Select it with ``CAPTION_ENGINE=qwen_direct``. The container default and
the legacy pipeline/ensemble paths are not changed.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import subprocess
import tempfile
import weakref
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from app import pipeline as P
from app.models import (
    REQUIRED_STYLES,
    caption_passes_style_filter,
    fallback_caption,
    normalize_captions,
)

log = logging.getLogger("track2.qwen_direct")

FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
MODEL = os.environ.get(
    "QWEN_DIRECT_MODEL", "accounts/fireworks/models/qwen3p7-plus"
)
FRAME_COUNT = 4
FRAME_MAX_EDGE = 1024
TEMPERATURE = 0.7
MAX_TOKENS = 400
REASONING_EFFORT = "none"
MAX_ATTEMPTS = 3
HTTP_TIMEOUT_S = float(os.environ.get("QWEN_DIRECT_HTTP_TIMEOUT_S", "45"))
STYLE_CONCURRENCY = int(os.environ.get("QWEN_DIRECT_STYLE_CONCURRENCY", "1"))
RETRY_MAX_DELAY_S = float(os.environ.get("QWEN_DIRECT_RETRY_MAX_DELAY_S", "4"))
RETRY_JITTER_CAP_S = float(
    os.environ.get("QWEN_DIRECT_RETRY_JITTER_CAP_S", "0.35")
)
PROMPT_PROFILE = (
    os.environ.get("QWEN_DIRECT_PROMPT_PROFILE", "v1").strip().casefold()
    or "v1"
)

_STYLE_PERSONAS_V1 = {
    "formal": (
        "You are a professional video captioner. Write in an objective, factual, "
        "neutral tone with precise visible details and no humor."
    ),
    "sarcastic": (
        "You write video captions with dry, lightly ironic wit. Keep every scene "
        "claim literal and visible; the irony may frame the facts but never replace them. "
        "Use no technology jargon and no exclamation marks."
    ),
    "humorous_tech": (
        "You write concise technology-flavored comedy. Tie one natural programming "
        "or computing metaphor to the visible action while preserving the literal scene."
    ),
    "humorous_non_tech": (
        "You write warm everyday comedy with no technical jargon. Use one playful, "
        "ordinary-life comparison while keeping the visible scene accurate."
    ),
}

_GROUNDING_RULES_V1 = (
    " The four images are chronological frames from one video. Mention the main "
    "subject, visible action, setting, and meaningful change only when the frames "
    "support them. Do not infer identity, intent, dialogue, unseen events, brands, "
    "or exact counts that are unclear. Write one compact English caption of one or "
    "two complete sentences, normally 25-55 words. Return only the caption: no label, "
    "preamble, analysis, markdown, or JSON."
)

_STYLE_PERSONAS_STRONG_V2 = {
    "formal": (
        "Write a professional, objective, factual video caption with no humor "
        "or editorial judgment."
    ),
    "sarcastic": (
        "Write a clearly sarcastic caption with unmistakable dry irony. The wit "
        "must be obvious, not merely factual, while every scene claim stays literal. "
        "Use no technology jargon or exclamation marks."
    ),
    "humorous_tech": (
        "Write a clearly funny caption using one explicit technology or programming "
        "metaphor tied directly to the main visible action. Do not add a second gag."
    ),
    "humorous_non_tech": (
        "Write an obvious everyday joke or familiar-life comparison tied directly "
        "to the main visible action. Use no technical jargon."
    ),
}

_GROUNDING_RULES_STRONG_V2 = (
    " Use exactly one complete sentence of 18-32 words. Select only the main "
    "subject, main directly visible action, and a safe general setting. Omit "
    "chronology, camera changes, clothing, colors, counts, directions, and minor "
    "objects unless unmistakably supported by the four frames. Do not turn sampled "
    "differences into a temporal story or infer identity, intent, dialogue, or unseen "
    "events. Return only the caption with no label, preamble, analysis, markdown, or JSON."
)

Requester = Callable[[dict[str, Any]], Awaitable[str]]
_STYLE_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    tuple[int, asyncio.Semaphore],
] = weakref.WeakKeyDictionary()

if STYLE_CONCURRENCY < 1:
    raise ValueError("QWEN_DIRECT_STYLE_CONCURRENCY must be at least 1")
if RETRY_MAX_DELAY_S <= 0:
    raise ValueError("QWEN_DIRECT_RETRY_MAX_DELAY_S must be positive")
if RETRY_JITTER_CAP_S < 0:
    raise ValueError("QWEN_DIRECT_RETRY_JITTER_CAP_S must not be negative")


def _validate_styles(styles: list[str]) -> None:
    if not styles or len(styles) != len(set(styles)):
        raise ValueError("styles must be a non-empty unique list")
    unknown = set(styles) - set(REQUIRED_STYLES)
    if unknown:
        raise ValueError(f"unsupported styles: {sorted(unknown)}")


def _validate_frames(frames: list[Path]) -> None:
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"qwen_direct requires exactly {FRAME_COUNT} frames")
    if any(not frame.is_file() or frame.stat().st_size <= 0 for frame in frames):
        raise ValueError("every qwen_direct frame must be a non-empty file")


def _shared_style_semaphore() -> asyncio.Semaphore:
    """Share one provider limit across every video on the current event loop."""
    loop = asyncio.get_running_loop()
    existing = _STYLE_SEMAPHORES.get(loop)
    if existing is None or existing[0] != STYLE_CONCURRENCY:
        semaphore = asyncio.Semaphore(STYLE_CONCURRENCY)
        _STYLE_SEMAPHORES[loop] = (STYLE_CONCURRENCY, semaphore)
        return semaphore
    return existing[1]


def _system_prompt(style: str, prompt_profile: str) -> str:
    if not isinstance(prompt_profile, str):
        raise ValueError("unknown qwen direct prompt profile")
    profile = prompt_profile.strip().casefold()
    if profile == "v1":
        return _STYLE_PERSONAS_V1[style] + _GROUNDING_RULES_V1
    if profile == "strong_v2":
        return (
            _STYLE_PERSONAS_STRONG_V2[style]
            + _GROUNDING_RULES_STRONG_V2
        )
    raise ValueError(f"unknown qwen direct prompt profile: {prompt_profile}")


def build_request(
    style: str,
    frames: list[Path],
    *,
    prompt_profile: str | None = None,
) -> dict[str, Any]:
    """Build one style-specific Fireworks multimodal request."""
    _validate_styles([style])
    _validate_frames(frames)
    profile = PROMPT_PROFILE if prompt_profile is None else prompt_profile
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Watch these four chronological frames from one clip and write one "
                "caption in the requested voice."
            ),
        }
    ]
    for frame in frames:
        encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt(style, profile),
            },
            {"role": "user", "content": content},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }


def _message_text(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Fireworks response has no message content") from error
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {None, "text"}
        )
    else:
        raise ValueError("Fireworks message content is not text")
    text = P._extract_final_caption(text).strip().strip('"').strip("'").strip()
    if not text:
        raise ValueError("Fireworks returned an empty caption")
    return text


async def _fireworks_request(payload: dict[str, Any]) -> str:
    if not FIREWORKS_API_KEY:
        raise ValueError("FIREWORKS_API_KEY is not configured")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        response = await client.post(
            f"{FIREWORKS_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {FIREWORKS_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return _message_text(response.json())


def _retry_after_seconds(error: Exception) -> float | None:
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    if error.response.status_code != 429:
        return None
    raw = error.response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            value = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, value)


def _retry_delay_seconds(
    error: Exception,
    *,
    attempt: int,
    base_delay_s: float,
) -> float:
    """Return a bounded backoff while respecting short server retry hints."""
    if attempt < 0:
        raise ValueError("retry attempt must not be negative")
    base = max(0.0, float(base_delay_s))
    exponential = min(RETRY_MAX_DELAY_S, base * (2**attempt))
    jitter_ceiling = min(RETRY_JITTER_CAP_S, exponential)
    jitter = random.uniform(0.0, jitter_ceiling) if jitter_ceiling else 0.0
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return min(RETRY_MAX_DELAY_S, retry_after + jitter)
    return min(RETRY_MAX_DELAY_S, exponential + jitter)


async def _caption_style_from_frames(
    frames: list[Path],
    style: str,
    requester: Requester,
    retry_delay_s: float,
) -> str:
    payload = build_request(style, frames)
    for attempt in range(MAX_ATTEMPTS):
        try:
            text = (await requester(payload)).strip()
            if not text:
                raise ValueError("caption response is empty")
            return P._extract_final_caption(text).strip().strip('"').strip("'").strip()
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            retry_after = _retry_after_seconds(error)
            if (
                retry_after is not None
                and retry_after > RETRY_MAX_DELAY_S
            ):
                log.warning(
                    "direct caption defers [%s]: Retry-After %.2fs exceeds "
                    "the %.2fs retry budget",
                    style,
                    retry_after,
                    RETRY_MAX_DELAY_S,
                )
                return ""
            if attempt + 1 == MAX_ATTEMPTS:
                log.warning(
                    "direct caption failed [%s] after %d attempts: %s",
                    style,
                    MAX_ATTEMPTS,
                    error,
                )
                return ""
            delay = _retry_delay_seconds(
                error,
                attempt=attempt,
                base_delay_s=retry_delay_s,
            )
            log.warning(
                "direct caption retry [%s] in %.2fs: %s",
                style,
                delay,
                error,
            )
            if delay > 0:
                await asyncio.sleep(delay)
    return ""


async def caption_styles_from_frames(
    frames: list[Path],
    styles: list[str],
    *,
    requester: Requester | None = None,
    retry_delay_s: float = 0.75,
) -> dict[str, str]:
    """Run one independent multimodal call for every requested style."""
    _validate_styles(styles)
    _validate_frames(frames)
    request = requester or _fireworks_request
    semaphore = _shared_style_semaphore()

    async def limited_request(payload: dict[str, Any]) -> str:
        async with semaphore:
            return await request(payload)

    results = await asyncio.gather(
        *(
            _caption_style_from_frames(
                frames,
                style,
                limited_request,
                retry_delay_s,
            )
            for style in styles
        )
    )
    return dict(zip(styles, results, strict=True))


def _extract_fps_frames(video: Path, workdir: Path, duration: float) -> list[Path]:
    """Mirror the confirmed four-frame source geometry with one FFmpeg pass."""
    duration_value = P._positive_finite(duration, "video duration")
    output_dir = workdir / "qwen_direct"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("frame_*.jpg"):
        stale.unlink()
    output_pattern = output_dir / "frame_%02d.jpg"
    fps = FRAME_COUNT / duration_value
    scale = (
        f"scale='min({FRAME_MAX_EDGE},iw)':'min({FRAME_MAX_EDGE},ih)':"
        "force_original_aspect_ratio=decrease"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"fps={fps:.8f},{scale}",
            "-vframes", str(FRAME_COUNT),
            "-pix_fmt", "yuvj420p",
            "-q:v", "4",
            str(output_pattern),
        ],
        check=True,
        timeout=12.0,
    )
    frames = sorted(
        frame
        for frame in output_dir.glob("frame_*.jpg")
        if frame.is_file() and frame.stat().st_size > 0
    )
    _validate_frames(frames)
    return frames


def extract_frames(video: Path, workdir: Path) -> list[Path]:
    """Extract the confirmed four-frame, quarter-interval 1024px profile."""
    duration = P._ffprobe_duration(video)
    if duration <= 0:
        duration = 60.0
    frames = _extract_fps_frames(video=video, workdir=workdir, duration=duration)
    _validate_frames(frames)
    return frames


async def _pipeline_fallback(video_url: str, styles: list[str]) -> dict[str, str]:
    try:
        return await P.caption_one_video(video_url=video_url, styles=styles)
    except Exception as error:  # noqa: BLE001 - the outer contract must stay valid
        log.warning("legacy pipeline fallback failed: %s", error)
        return {style: fallback_caption(style) for style in styles}


async def caption_qwen_direct(video_url: str, styles: list[str]) -> dict[str, str]:
    """Caption one video with the opt-in direct profile and safe fallback."""
    _validate_styles(styles)
    if not FIREWORKS_API_KEY:
        log.warning("qwen_direct has no Fireworks key; using legacy pipeline")
        captions = await _pipeline_fallback(video_url, styles)
        return normalize_captions(captions, styles)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            video = await P._download(video_url, workdir / "clip.mp4")
            frames = extract_frames(video, workdir)
            captions = await caption_styles_from_frames(frames, styles)
    except Exception as error:  # noqa: BLE001 - extraction/API setup can degrade
        log.warning("qwen_direct primary path failed: %s", error)
        captions = {style: "" for style in styles}

    missing = [
        style
        for style in styles
        if not str(captions.get(style, "")).strip()
        or not caption_passes_style_filter(style, str(captions[style]))
    ]
    if missing:
        fallback = await _pipeline_fallback(video_url, missing)
        for style in missing:
            captions[style] = fallback.get(style, fallback_caption(style))
    return normalize_captions(captions, styles)
