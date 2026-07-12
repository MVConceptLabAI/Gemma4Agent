"""
Two-stage pipeline:

    1. DESCRIBE — VLM reads 8 sampled frames + audio-transcript hint → scene facts JSON.
    2. STYLE    — 4 parallel LLM calls (one per requested style), grounded in the facts.

This split lets you optimise the two scoring axes independently:
    - Caption accuracy is set by stage 1 (what's in the video).
    - Style match is set by stage 2 (tone, per-style system prompts + few-shots).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from app.models import caption_passes_style_filter, fallback_caption, normalize_captions
from app.prompts import DESCRIBE_SYSTEM, DESCRIBE_USER, STYLE_PROMPTS

load_dotenv()

log = logging.getLogger("track2.pipeline")


# =============================================================================
# Config — every knob is env-driven so the same image serves dev + prod.
# =============================================================================
FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PROVIDER_ORDER = os.environ.get("PROVIDER_ORDER", "groq,fireworks,openrouter")
DESCRIBE_PROVIDER_ORDER = os.environ.get("DESCRIBE_PROVIDER_ORDER", PROVIDER_ORDER)
STYLE_PROVIDER_ORDER = os.environ.get("STYLE_PROVIDER_ORDER", PROVIDER_ORDER)

# Vision model (multimodal). Qwen2.5-VL 7B on Fireworks is the recommended default.
VLM_MODEL = os.environ.get(
    "VLM_MODEL", "accounts/fireworks/models/qwen2p5-vl-7b-instruct"
)
VLM_FALLBACK_MODELS = os.environ.get("VLM_FALLBACK_MODELS", "")
DIRECT_VIDEO_MODEL = os.environ.get("DIRECT_VIDEO_MODEL", "")
GROQ_VISION_MODEL = os.environ.get(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
GROQ_MAX_IMAGES = int(os.environ.get("GROQ_MAX_IMAGES", "5"))
GROQ_FRAME_MAX_EDGE = int(os.environ.get("GROQ_FRAME_MAX_EDGE", "512"))
OPENROUTER_VLM_MODEL = os.environ.get(
    "OPENROUTER_VLM_MODEL", "qwen/qwen3-vl-8b-instruct"
)
# Text model for the 4 style rewrites. Gemma 3 27B → eligible for the Gemma bonus.
STYLE_MODEL = os.environ.get(
    "STYLE_MODEL", "accounts/fireworks/models/gemma-3-27b-it"
)
STYLE_LORA = os.environ.get("STYLE_LORA", "")
STYLE_FALLBACK_MODELS = os.environ.get("STYLE_FALLBACK_MODELS", "")
GROQ_STYLE_MODEL = os.environ.get("GROQ_STYLE_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_STYLE_MODEL = os.environ.get("OPENROUTER_STYLE_MODEL", "qwen/qwen3-vl-8b-instruct")

NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "8"))
FRAME_MAX_EDGE = int(os.environ.get("FRAME_MAX_EDGE", "720"))
FRAME_PROFILES = {
    "describex_oci_hypothesis": tuple(
        0.05 + index * (0.90 / 8) for index in range(8)
    ),
    "endpoint_aware": tuple(
        0.05 + index * (0.90 / 7) for index in range(8)
    ),
}
DIRECT_VIDEO_MAX_SECONDS = int(os.environ.get("DIRECT_VIDEO_MAX_SECONDS", "60"))
SCENE_DETECT_ENABLED = os.environ.get("SCENE_DETECT_ENABLED", "1") != "0"
SCENE_THRESHOLD = float(os.environ.get("SCENE_THRESHOLD", "0.35"))
AUDIO_TRANSCRIBE_ENABLED = os.environ.get("AUDIO_TRANSCRIBE_ENABLED", "1") != "0"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "20"))
DESCRIBE_MAX_TOKENS = int(os.environ.get("DESCRIBE_MAX_TOKENS", "700"))
STYLE_MAX_TOKENS = int(os.environ.get("STYLE_MAX_TOKENS", "140"))
EVIDENCE_LOCK_ENABLED = os.environ.get("EVIDENCE_LOCK_ENABLED", "0") != "0"
DETERMINISTIC_FORMAL = os.environ.get("DETERMINISTIC_FORMAL", "1") != "0"
STYLE_CANDIDATES = max(1, int(os.environ.get("STYLE_CANDIDATES", "2")))
STYLE_REPAIR_ENABLED = os.environ.get("STYLE_REPAIR_ENABLED", "1") != "0"


# =============================================================================
# Public entry
# =============================================================================
async def caption_one_video(video_url: str, styles: list[str]) -> dict[str, str]:
    """Full pipeline for one clip. Returns {style: caption_text}."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        video_path = await _download(video_url, workdir / "clip.mp4")
        facts = await _direct_video_facts(video_path, workdir)
        if facts:
            captions = await _style_all(facts, styles)
            normalized = normalize_captions(captions, styles, facts)
            if EVIDENCE_LOCK_ENABLED:
                normalized = await _repair_with_sibling_context(normalized, facts, styles)
            return normalize_captions(normalized, styles, facts)
        frames = _extract_keyframes(video_path, workdir, NUM_FRAMES, FRAME_MAX_EDGE)
        # Audio transcript is optional but strongly boosts humorous_tech / non_tech
        # scores (jokes live in the voice track). Left as a hook — enable Whisper
        # if you ship it in the image.
        transcript_hint = await _transcript_hint(video_path, workdir)

        facts = await _describe(frames, transcript_hint)
        facts = _neutralize_risky_colors(facts)
        captions = await _style_all(facts, styles)
        normalized = normalize_captions(captions, styles, facts)
        if EVIDENCE_LOCK_ENABLED:
            normalized = await _repair_with_sibling_context(normalized, facts, styles)
        return normalize_captions(normalized, styles, facts)


# =============================================================================
# Step 0 — fetch the video (harness may point at gs:// public URLs)
# =============================================================================
async def _download(url: str, dst: Path) -> Path:
    return await _download_with_retry(url, dst)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=3.0),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _download_with_retry(url: str, dst: Path) -> Path:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as c:
        async with c.stream("GET", url) as r:
            r.raise_for_status()
            with dst.open("wb") as f:
                async for chunk in r.aiter_bytes():
                    f.write(chunk)
    return dst


def _model_candidates(primary: str, fallbacks: str) -> list[str]:
    models = [primary, *(m.strip() for m in fallbacks.split(",") if m.strip())]
    return list(dict.fromkeys(models))


def _provider_order(kind: str = "all") -> list[str]:
    order = PROVIDER_ORDER
    if kind == "describe":
        order = DESCRIBE_PROVIDER_ORDER
    elif kind == "style":
        order = STYLE_PROVIDER_ORDER
    return [p.strip().lower() for p in order.split(",") if p.strip()]


def _provider_endpoint(provider: str, *, style: bool = False) -> tuple[str, str, list[str]]:
    if provider == "groq" and GROQ_API_KEY:
        model = GROQ_STYLE_MODEL if style else GROQ_VISION_MODEL
        return GROQ_BASE_URL, GROQ_API_KEY, [model]
    if provider == "fireworks" and FIREWORKS_API_KEY:
        if style:
            primary = STYLE_LORA or STYLE_MODEL
            models = _model_candidates(primary, STYLE_FALLBACK_MODELS)
        else:
            models = _model_candidates(VLM_MODEL, VLM_FALLBACK_MODELS)
        return FIREWORKS_BASE_URL, FIREWORKS_API_KEY, models
    if provider == "openrouter" and OPENROUTER_API_KEY:
        model = OPENROUTER_STYLE_MODEL if style else OPENROUTER_VLM_MODEL
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY, [model]
    return "", "", []


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) and _facts_useful(obj) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                candidates.append(obj)
        except json.JSONDecodeError:
            continue
    useful = [obj for obj in candidates if _facts_useful(obj)]
    if useful:
        return max(useful, key=_facts_score)
    return max(candidates, key=_facts_score) if candidates else None


def _facts_score(facts: dict[str, Any]) -> int:
    score = 0
    for key in ("summary", "setting", "camera", "temporal_progression"):
        value = facts.get(key)
        if isinstance(value, str) and value.strip():
            score += len(value.split())
    for key in (
        "subjects",
        "actions",
        "visual_details",
        "fine_grained_observations",
        "salient_objects",
        "spatial_relations",
    ):
        value = facts.get(key)
        if isinstance(value, list):
            score += sum(1 for item in value if str(item).strip())
    return score


def _facts_useful(facts: dict[str, Any]) -> bool:
    return _facts_score(facts) >= 8 and bool(str(facts.get("summary", "")).strip())


def _extract_final_caption(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    for marker in ("Final caption:", "Caption:", "Final:"):
        if marker.lower() in cleaned.lower():
            idx = cleaned.lower().rfind(marker.lower())
            cleaned = cleaned[idx + len(marker) :].strip()
            break
    # Drop obvious preamble/reasoning lines, then JOIN the caption body.
    # Taking only the last line here used to silently delete sentence 1 of a
    # two-sentence caption when the model put sentences on separate lines.
    preamble = ("here's", "here is", "sure", "certainly", "okay", "of course", "caption:")
    lines = [
        line.strip(" -*\t")
        for line in cleaned.splitlines()
        if line.strip() and not line.strip().lower().startswith(preamble)
    ]
    if lines:
        cleaned = " ".join(lines)
    # Strip a leaked leading reasoning token (some models prefix "thought ...").
    cleaned = re.sub(r"^\s*(thought|thinking|reasoning|answer)\b[:\s-]*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


async def _direct_video_facts(video_path: Path, workdir: Path) -> dict[str, Any] | None:
    if not DIRECT_VIDEO_MODEL or not FIREWORKS_API_KEY:
        return None
    try:
        video_b64, audio_b64 = _preprocess_direct_video(video_path, workdir)
        content: list[dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/ogg;base64,{audio_b64}"}},
            {"type": "text", "text": DESCRIBE_USER.format(transcript_hint="(audio attached)")},
        ]
        payload = {
            "model": DIRECT_VIDEO_MODEL,
            "messages": [
                {"role": "system", "content": DESCRIBE_SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": DESCRIBE_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        text = await _chat_content(payload)
        return json.loads(text)
    except Exception as e:  # noqa: BLE001
        log.warning("direct video describe failed: %s", e)
        return None


def _preprocess_direct_video(video_path: Path, workdir: Path) -> tuple[str, str]:
    processed_video = workdir / "direct_video.mp4"
    processed_audio = workdir / "direct_audio.ogg"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(video_path),
            "-t", str(DIRECT_VIDEO_MAX_SECONDS),
            "-vf", "fps=1,scale=-1:360",
            "-c:v", "libx264", "-preset", "fast",
            "-an", str(processed_video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(video_path),
            "-t", str(DIRECT_VIDEO_MAX_SECONDS),
            "-vn", "-c:a", "libopus", "-b:a", "24k",
            "-ar", "16000", "-ac", "1",
            str(processed_audio),
        ],
        check=True,
    )
    total_size = processed_video.stat().st_size + processed_audio.stat().st_size
    if total_size > 7_500_000:
        # ponytail: Fireworks recommends <10MB base64 payload; 7.5MB raw keeps margin.
        raise ValueError(f"direct video payload too large before base64: {total_size} bytes")
    return (
        base64.b64encode(processed_video.read_bytes()).decode("ascii"),
        base64.b64encode(processed_audio.read_bytes()).decode("ascii"),
    )


async def _transcript_hint(video_path: Path, workdir: Path) -> str:
    if not AUDIO_TRANSCRIBE_ENABLED or not GROQ_API_KEY:
        return ""
    try:
        if not _has_audio_stream(video_path):
            return ""
        audio_path = _extract_audio(video_path, workdir / "audio.wav")
        return await _transcribe_audio(audio_path)
    except Exception as e:  # noqa: BLE001
        log.warning("audio transcription failed: %s", e)
        return ""


def _has_audio_stream(video_path: Path) -> bool:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video_path),
            ],
            text=True,
        ).strip()
        return bool(out)
    except Exception:  # noqa: BLE001
        return False


def _extract_audio(video_path: Path, out_path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(out_path),
        ],
        check=True,
    )
    return out_path


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential_jitter(initial=0.5, max=3.0),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _transcribe_audio(audio_path: Path) -> str:
    data = {"model": WHISPER_MODEL, "response_format": "json"}
    files = {"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        r = await c.post(
            f"{GROQ_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            data=data,
            files=files,
        )
        r.raise_for_status()
        text = r.json().get("text", "")
    return text.strip()[:1200]


# =============================================================================
# Step 1 — extract N frames uniformly from the clip
# =============================================================================
def _extract_keyframes(video: Path, workdir: Path, n: int, max_edge: int) -> list[Path]:
    """Uniform sampling via ffmpeg — cheap, deterministic, works for 30s-2min clips."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    duration = _ffprobe_duration(video)
    if duration <= 0:
        duration = 60.0  # fallback assumption

    hybrid_paths: list[Path] = []
    if SCENE_DETECT_ENABLED:
        try:
            hybrid_paths.extend(
                _extract_scene_frames(
                    video=video,
                    workdir=workdir,
                    max_count=max(1, n // 2),
                    max_edge=max_edge,
                    threshold=SCENE_THRESHOLD,
                )
            )
        except Exception as e:  # noqa: BLE001
            log.warning("scene-detect frame extraction failed: %s", e)

    remaining = n - len(hybrid_paths)
    if remaining > 0:
        hybrid_paths.extend(
            _extract_uniform_frames(
                video=video,
                workdir=workdir,
                n=remaining,
                max_edge=max_edge,
                duration=duration,
                prefix="u",
            )
        )
    return hybrid_paths[:n]


def _extract_scene_frames(
    video: Path,
    workdir: Path,
    max_count: int,
    max_edge: int,
    threshold: float,
) -> list[Path]:
    scene_dir = workdir / "scene"
    scene_dir.mkdir(exist_ok=True)
    out_pattern = scene_dir / "s%03d.jpg"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"select='gt(scene\\,{threshold})',scale='min({max_edge},iw)':-2",
            "-vsync", "vfr",
            "-q:v", "4",
            str(out_pattern),
        ],
        check=False,
        capture_output=True,
    )
    return sorted(scene_dir.glob("s*.jpg"))[:max_count]


def _extract_uniform_frames(
    video: Path,
    workdir: Path,
    n: int,
    max_edge: int,
    duration: float,
    prefix: str,
) -> list[Path]:
    step = max(duration / (n + 1), 0.1)
    out_paths: list[Path] = []
    # Burn frame#/timestamp/duration into each frame (TIMESTAMP_FRAMES=1):
    # observers gain temporal grounding ("at 0:15 the bus enters") - the
    # leaderboard leader's single biggest measured lever.
    stamp = os.environ.get("TIMESTAMP_FRAMES", "0") != "0"
    for i in range(1, n + 1):
        t = round(i * step, 3)
        out = workdir / f"{prefix}{i:02d}.jpg"
        vf = f"scale='min({max_edge},iw)':-2"
        if stamp:
            label = f"frame {i}/{n}  t={int(t//60)}\\:{int(t%60):02d}  dur={int(duration//60)}\\:{int(duration%60):02d}"
            font = os.environ.get("TIMESTAMP_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            vf += (f",drawtext=fontfile='{font}':text='{label}':x=8:y=8:fontsize=20:"
                   "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", str(t), "-i", str(video),
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "4",
                str(out),
            ],
            check=True,
        )
        if out.exists():
            out_paths.append(out)
    return out_paths


def _ffprobe_duration(video: Path) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(video),
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return 0.0


def _ffprobe_video_timestamps(video: Path, timeout_s: float) -> list[float]:
    timeout_value = _positive_finite(timeout_s, "ffprobe timeout")
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            str(video),
        ],
        text=True,
        timeout=timeout_value,
    )
    timestamps: list[float] = []
    for token in re.split(r"[\s,]+", out.strip()):
        try:
            timestamp = float(token)
        except ValueError:
            continue
        if math.isfinite(timestamp) and timestamp >= 0:
            timestamps.append(timestamp)
    timestamps = sorted(set(timestamps))
    if not timestamps:
        raise ValueError("ffprobe returned no video frame timestamp")
    return timestamps


def _ffprobe_video_last_pts(video: Path, timeout_s: float) -> float:
    return _ffprobe_video_timestamps(video, timeout_s)[-1]


def _positive_finite(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _nonnegative_finite(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative finite number") from error
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return numeric


def _ratio_timestamps(duration: float, profile: str) -> list[float]:
    duration_value = _positive_finite(duration, "video duration")
    try:
        ratios = FRAME_PROFILES[profile]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown frame profile: {profile}") from error
    return [round(duration_value * ratio, 3) for ratio in ratios]


def _official_repo_indices(total_frames: int, target: int = 16) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if target < 2:
        raise ValueError("target must be at least two")
    if total_frames <= target:
        return list(range(total_frames))
    indices = [math.floor(index * total_frames / target) for index in range(target)]
    indices[0] = 0
    indices[-1] = total_frames - 1
    return sorted(set(indices))


def _ffmpeg_fps_extract(
    video: Path,
    workdir: Path,
    fps: float,
    qscale: int,
    total_timeout_s: float,
) -> list[Path]:
    fps_value = _positive_finite(fps, "extraction fps")
    timeout_value = _positive_finite(total_timeout_s, "extraction timeout")
    if not isinstance(qscale, int) or isinstance(qscale, bool) or not 1 <= qscale <= 31:
        raise ValueError("qscale must be an integer between 1 and 31")

    output_dir = workdir / "official_repo"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in output_dir.glob("frame_*.jpg"):
        stale_frame.unlink()
    output_pattern = output_dir / "frame_%03d.jpg"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-vf", f"fps={fps_value:.8f}",
            "-q:v", str(qscale), str(output_pattern),
        ],
        check=True,
        timeout=timeout_value,
    )
    frames = sorted(
        frame
        for frame in output_dir.glob("frame_*.jpg")
        if frame.is_file() and frame.stat().st_size > 0
    )
    if not frames:
        raise RuntimeError("official-repository extraction produced no frames")
    return frames


def _extract_official_repo_frames(
    video: Path,
    workdir: Path,
    video_fps: float,
    duration: float,
) -> list[Path]:
    fps_value = _positive_finite(video_fps, "video fps")
    duration_value = _positive_finite(duration, "video duration")
    extraction_fps = min(fps_value, 60.0 / duration_value)
    extracted = _ffmpeg_fps_extract(
        video=video,
        workdir=workdir,
        fps=extraction_fps,
        qscale=2,
        total_timeout_s=12.0,
    )
    return [extracted[index] for index in _official_repo_indices(len(extracted))]


def _extract_frames_at_timestamps(
    video: Path,
    workdir: Path,
    timestamps: Sequence[float],
    max_edge: int,
    jpeg_quality: int,
    deadline: float | None = None,
    allow_repeated: bool = False,
) -> list[Path]:
    if not isinstance(max_edge, int) or isinstance(max_edge, bool) or max_edge <= 0:
        raise ValueError("max_edge must be a positive integer")
    if (
        not isinstance(jpeg_quality, int)
        or isinstance(jpeg_quality, bool)
        or not 1 <= jpeg_quality <= 100
    ):
        raise ValueError("jpeg_quality must be an integer between 1 and 100")

    validator = _nonnegative_finite if allow_repeated else _positive_finite
    timestamp_values = [
        validator(timestamp, "frame timestamp") for timestamp in timestamps
    ]
    if not timestamp_values:
        raise ValueError("at least one frame timestamp is required")
    pairs = zip(timestamp_values, timestamp_values[1:])
    out_of_order = (
        any(current < previous for previous, current in pairs)
        if allow_repeated
        else any(current <= previous for previous, current in pairs)
    )
    if out_of_order:
        order = "non-decreasing" if allow_repeated else "strictly increasing"
        raise ValueError(f"frame timestamps must be {order}")

    frames: list[Path] = []
    if deadline is None:
        deadline = time.monotonic() + 12.0
    elif not math.isfinite(deadline):
        raise ValueError("frame extraction deadline must be finite")
    qscale = max(2, min(31, round((100 - jpeg_quality) * 0.29 + 2)))
    scale = (
        f"scale='min({max_edge},iw)':'min({max_edge},ih)':"
        "force_original_aspect_ratio=decrease"
    )
    for index, timestamp in enumerate(timestamp_values, start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("12-second frame extraction budget exhausted")
        target = workdir / f"leader_{index:02d}.jpg"
        target.unlink(missing_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{timestamp:.3f}", "-i", str(video),
                "-frames:v", "1",
                "-vf", scale,
                "-pix_fmt", "yuvj420p",
                "-q:v", str(qscale), str(target),
            ],
            check=True,
            timeout=min(3.0, remaining),
        )
        if target.is_file() and target.stat().st_size > 0:
            frames.append(target)
    if len(frames) != len(timestamp_values):
        raise RuntimeError(
            f"extracted {len(frames)} of {len(timestamp_values)} requested frames"
        )
    return frames


def _extract_ratio_frames(
    video: Path,
    workdir: Path,
    profile: str,
    max_edge: int = 768,
) -> list[Path]:
    deadline = time.monotonic() + 12.0
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("12-second frame extraction budget exhausted")
    frame_timestamps = _ffprobe_video_timestamps(
        video, timeout_s=min(3.0, remaining)
    )
    last_video_pts = frame_timestamps[-1]
    ratio_timestamps = _ratio_timestamps(
        last_video_pts if last_video_pts > 0 else 1.0,
        profile,
    )
    timestamps = [
        min(
            frame_timestamps,
            key=lambda frame_pts: (abs(frame_pts - target), frame_pts),
        )
        for target in ratio_timestamps
    ]
    return _extract_frames_at_timestamps(
        video=video,
        workdir=workdir,
        timestamps=timestamps,
        max_edge=max_edge,
        jpeg_quality=85,
        deadline=deadline,
        allow_repeated=True,
    )


# =============================================================================
# Step 2 — DESCRIBE (VLM): frames + optional transcript → scene facts JSON
# =============================================================================
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential_jitter(initial=0.5, max=3.0),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _chat_content(payload: dict[str, Any]) -> str:
    return await _chat_content_at(FIREWORKS_BASE_URL, FIREWORKS_API_KEY, payload)


# Groq's free tier bursts hard under concurrency: a transient 429 used to
# collapse the whole describe stage to a generic stub (score ~0) because nothing
# absorbed it. Retry 429s inline, honoring Retry-After, before surfacing the
# error to the provider loop. tenacity still covers genuine transport errors.
_MAX_429_RETRIES = int(os.environ.get("HTTP_429_RETRIES", "3"))
_MAX_429_WAIT_S = float(os.environ.get("HTTP_429_MAX_WAIT_S", "8"))
# If the provider says the quota is gone for longer than this (e.g. Groq daily
# limit answers retry-after 600+), retrying inside the request budget is
# pointless: fail over to the next provider immediately instead of stalling.
_RETRY_AFTER_GIVEUP_S = float(os.environ.get("RETRY_AFTER_GIVEUP_S", "60"))


def _retry_after_seconds(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait after a 429: honor Retry-After, else back off, capped.

    Returns -1 when the advertised wait exceeds the give-up threshold —
    the caller should stop retrying and surface the error immediately.
    """
    hdr = (resp.headers.get("retry-after") or "").strip()
    try:
        wait = float(hdr)
    except ValueError:
        wait = 2.0  # ponytail: Retry-After can be an HTTP-date; 2s default is fine
    if wait > _RETRY_AFTER_GIVEUP_S:
        return -1.0
    return min(wait + attempt, _MAX_429_WAIT_S)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential_jitter(initial=0.5, max=3.0),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)
async def _chat_content_at(base_url: str, api_key: str, payload: dict[str, Any]) -> str:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        r: httpx.Response | None = None
        for attempt in range(_MAX_429_RETRIES + 1):
            r = await c.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if r.status_code == 429 and attempt < _MAX_429_RETRIES:
                wait = _retry_after_seconds(r, attempt)
                if wait < 0:  # provider quota gone for minutes — fail over now
                    break
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        # 429 never cleared — surface as HTTPStatusError so the provider loop
        # (which catches httpx.HTTPError) fails over to the next model/provider.
        r.raise_for_status()
        raise httpx.HTTPStatusError("429 not cleared", request=r.request, response=r)


def _shrink_image_part(part: dict[str, Any], max_edge: int) -> dict[str, Any]:
    """Re-encode a base64 image content part to fit max_edge (JPEG q80)."""
    try:
        import io
        from PIL import Image
        url = part["image_url"]["url"]
        b64 = url.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        if max(img.size) <= max_edge:
            return part
        img.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=80)
        small = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{small}"}}
    except Exception as e:  # noqa: BLE001 - shrinking is an optimization, never fatal
        log.warning("frame shrink failed (%s); sending original", e)
        return part


async def _describe(frames: list[Path], transcript_hint: str) -> dict[str, Any]:
    """Ask the VLM for a structured JSON summary of the clip."""
    if not any(_provider_endpoint(provider)[1] for provider in _provider_order("describe")):
        # Degrade gracefully if no key at runtime (e.g. unit tests).
        log.warning("no describe provider API key configured; returning fallback facts")
        return {
            "summary": "A short video clip with visible subjects and actions.",
            "setting": "unknown",
            "subjects": [],
            "actions": [],
            "mood": "neutral",
            "audio_hint": "",
            "tech_visible": False,
        }

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": DESCRIBE_USER.format(
                transcript_hint=transcript_hint or "(no transcript available)"
            ),
        }
    ]
    for fp in frames:
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    last_error: Exception | None = None
    for provider in _provider_order("describe"):
        base_url, api_key, models = _provider_endpoint(provider)
        provider_content = content
        if provider == "groq":
            # Groq free tier meters image tokens hard: 5 frames at 896px blow the
            # TPM limit and every describe 429s into the generic stub. Cap frame
            # count AND resolution for Groq only (other providers keep full size).
            provider_content = [content[0], *(
                _shrink_image_part(p, GROQ_FRAME_MAX_EDGE)
                for p in content[1 : 1 + GROQ_MAX_IMAGES]
            )]
        for model in models:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            DESCRIBE_SYSTEM
                            + "\n\nCritical output rule: do not show reasoning, analysis, "
                            "markdown, bullets, or drafts. The first character must be { "
                            "and the last character must be }."
                        ),
                    },
                    {"role": "user", "content": provider_content},
                ],
                "temperature": 0.2,
                "max_tokens": max(DESCRIBE_MAX_TOKENS, 2200) if provider == "fireworks" else DESCRIBE_MAX_TOKENS,
                "response_format": {"type": "json_object"},
            }
            try:
                text = await _chat_content_at(base_url, api_key, payload)
                obj = _extract_json_object(text)
                if obj is not None and _facts_useful(obj):
                    log.info("describe provider succeeded: %s/%s", provider, model)
                    return obj
                last_error = ValueError("describe returned no useful scene facts")
                log.warning("describe model returned weak facts (%s/%s)", provider, model)
            except (httpx.HTTPError, ValueError) as e:
                last_error = e
                log.warning("describe model failed (%s/%s): %s", provider, model, e)
    if last_error:
        log.warning("all describe providers failed; returning fallback facts")
    return {"summary": "A short video clip with visible subjects and actions."}


# =============================================================================
# Step 3 — STYLE (LLM ×4 in parallel)
# =============================================================================
async def _style_all(facts: dict[str, Any], styles: list[str]) -> dict[str, str]:
    tasks = [asyncio.create_task(_style_one(facts, s)) for s in styles]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, str] = {}
    for style, res in zip(styles, results):
        out[style] = fallback_caption(style, facts) if isinstance(res, Exception) else res
    if EVIDENCE_LOCK_ENABLED:
        out = await _repair_with_sibling_context(out, facts, styles)
    return out


async def _style_one(facts: dict[str, Any], style: str) -> str:
    if not any(_provider_endpoint(provider, style=True)[1] for provider in _provider_order("style")):
        return fallback_caption(style, facts)

    prompt_bundle = STYLE_PROMPTS.get(style)
    if prompt_bundle is None:
        # Unknown style — emit a neutral one-liner rather than nothing.
        return fallback_caption("formal", facts)

    system_prompt, few_shots, user_template = prompt_bundle
    facts_json = json.dumps(facts, ensure_ascii=False)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for ex_facts, ex_caption in few_shots:
        messages.append({"role": "user", "content": user_template.format(facts=ex_facts)})
        messages.append({"role": "assistant", "content": ex_caption})
    messages.append({"role": "user", "content": user_template.format(facts=facts_json)})

    if EVIDENCE_LOCK_ENABLED:
        return await _style_with_evidence_lock(messages, facts, style)

    # FORMAL is scored on completeness+accuracy, not creativity. The describe
    # facts are already rich and verified, so render formal deterministically
    # from them rather than let a style model compress/embellish it. The three
    # creative styles still go through the model (and keep the Gemma bonus).
    if style == "formal" and DETERMINISTIC_FORMAL:
        deterministic = _deterministic_formal(facts)
        if _word_count(deterministic) >= 20:
            return deterministic
        return _ensure_formal_richness(await _call_style_provider(messages, style), facts)

    caption = await _call_style_provider(messages, style)
    if style == "formal":
        caption = _ensure_formal_richness(caption, facts)
    return caption


_RISKY_COLOR = re.compile(
    r"\b(red|blue|green|yellow|white|black|silver|grey|gray|orange|brown|golden)"
    r"(?:-framed|-colored|-coloured)?\s+(?:\w+\s+)?"
    r"(bus|buses|car|cars|truck|trucks|van|vans|sedan|sedans|suv|suvs|vehicle|vehicles|"
    r"monitor|monitors|screen|screens|tv|laptop|laptops|keyboard|keyboards|"
    r"glove|gloves|collar|collars|leash|leash)\b",
    re.IGNORECASE,
)


_RING_HAND = re.compile(
    r"\b(a\s+)?ring\s+(?:is\s+)?(?:worn\s+)?on\s+(?:her|his|the|their|one)?\s*"
    r"(?:left|right)?\s*(?:hand'?s?\s+)?(?:ring|index|middle|little|pinky|pinkie)?\s*finger\b",
    re.IGNORECASE,
)


def _neutralize_risky_colors(facts: dict[str, Any]) -> dict[str, Any]:
    """Strip guessed colors from vehicles and screens in the describe facts.

    The VLM sometimes asserts a color for a vehicle/monitor that the frames do
    not support (e.g. "a red bus" when the bus is blue). Vehicle/screen color is
    low value and a frequent contradiction, so we drop the color adjective and
    keep the noun. Runs on every string field before the facts reach captions.
    """
    def scrub(v: Any) -> Any:
        if isinstance(v, str):
            out = _RISKY_COLOR.sub(lambda m: m.group(2), v)
            # Which hand/finger a ring is on is a frequent contradiction; drop it.
            out = _RING_HAND.sub("a ring", out)
            return out
        if isinstance(v, list):
            return [scrub(x) for x in v]
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items()}
        return v
    return {k: scrub(v) for k, v in facts.items()}


_LEAD_DETERMINERS = {
    "The", "A", "An", "Some", "Several", "Two", "Three", "Four", "This", "There", "Its", "Their",
}


def _lc_lead(phrase: str) -> str:
    """Lowercase a leading determiner so a phrase reads cleanly mid-sentence.

    "The desk is white" -> "the desk is white" when spliced into a list. Only
    known determiners are touched, so acronyms/proper nouns (KOREA, TV) survive.
    """
    first = phrase.split(" ", 1)[0]
    if first in _LEAD_DETERMINERS:
        return phrase[0].lower() + phrase[1:]
    return phrase


def _deterministic_formal(facts: dict[str, Any]) -> str:
    """Assemble a rich, factual formal caption straight from the scene facts.

    base = the describe summary; then append the strongest background phrases
    (buildings, signage, terrain, objects) that the summary did not already
    mention, filling up to the 300-char caption cap. Every token is grounded in
    the describe output — nothing is invented here.
    """
    summary = str(facts.get("summary", "")).strip()
    if not summary:
        return ""
    base = summary.rstrip(".")
    present = base.lower()
    picked: list[str] = []
    seen = {base.lower()}
    for phrase in _background_phrases(facts):
        p = " ".join(phrase.strip().rstrip(".").split())
        if not p or len(p.split()) > 8 or p.lower() in seen:
            continue
        if any(w in present for w in p.lower().split() if len(w) > 4):
            continue
        seen.add(p.lower())
        picked.append(p)
        present += " " + p.lower()  # dedup later phrases against picked ones too
    joined = base + "."
    budget = 292 - len(joined) - len(" Also visible: .")
    kept: list[str] = []
    for p in picked:
        if budget - (len(p) + 2) < 0:
            break
        kept.append(p)
        budget -= len(p) + 2
        if len(kept) >= 4:
            break
    if not kept:
        return joined
    kept = [_lc_lead(p) for p in kept]
    if len(kept) == 1:
        detail = kept[0]
    else:
        detail = ", ".join(kept[:-1]) + f", and {kept[-1]}"
    return f"{joined} Also visible: {detail}."


def _ensure_formal_richness(caption: str, facts: dict[str, Any]) -> str:
    """Guarantee the factual background survives into the formal caption.

    The style model reliably writes the main subject but often drops the rich,
    verified background (buildings, signage, terrain, lighting) even when the
    scene-facts contain it. Rather than fight that stochastically via prompts,
    we deterministically append the strongest unused background phrases when
    the caption is thin. Accuracy-positive: every appended phrase comes from the
    describe facts, never invented.
    """
    if _word_count(caption) >= 34:
        return caption
    present = caption.lower()
    # Prefer concise background phrases (salient_objects are short nouns); keep
    # only those adding NEW information, short enough to read cleanly.
    candidates: list[str] = []
    for phrase in _background_phrases(facts):
        p = phrase.strip().rstrip(".")
        if not p or len(p.split()) > 8:
            continue
        if p.lower() in present:
            continue
        if any(w in present for w in p.lower().split() if len(w) > 4):
            continue
        if p.lower() not in {c.lower() for c in candidates}:
            candidates.append(p)
    joined = caption.rstrip()
    if not joined.endswith("."):
        joined += "."
    # Fill under the 300-char caption cap (leave margin for ", and " + period).
    picked: list[str] = []
    budget = 292 - len(joined) - len(" In the background, .")
    for p in candidates:
        add = len(p) + 2
        if budget - add < 0:
            break
        picked.append(p)
        budget -= add
        if len(picked) >= 3:
            break
    if not picked:
        return caption
    picked = [_lc_lead(p) for p in picked]
    detail = picked[0] if len(picked) == 1 else (
        f"{picked[0]} and {picked[1]}" if len(picked) == 2
        else f"{picked[0]}, {picked[1]}, and {picked[2]}"
    )
    return f"{joined} In the background, {detail}."


STYLE_REASONING_EFFORT = os.environ.get("STYLE_REASONING_EFFORT", "")


async def _call_style_provider(messages: list[dict[str, Any]], style: str) -> str:
    last_error: Exception | None = None
    for provider in _provider_order("style"):
        base_url, api_key, models = _provider_endpoint(provider, style=True)
        for model in models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7 if style != "formal" else 0.3,
                "max_tokens": STYLE_MAX_TOKENS,
            }
            # gpt-oss on Fireworks: low effort keeps reasoning out of the token
            # budget (empty-content risk) and cuts latency ~3x. Fireworks-only —
            # other providers may reject unknown params.
            if provider == "fireworks" and STYLE_REASONING_EFFORT:
                payload["reasoning_effort"] = STYLE_REASONING_EFFORT
            try:
                return _extract_final_caption(await _chat_content_at(base_url, api_key, payload))
            except httpx.HTTPError as e:
                last_error = e
                log.warning("style model failed (%s/%s/%s): %s", style, provider, model, e)

    if last_error:
        raise last_error
    return ""


async def _style_with_evidence_lock(
    messages: list[dict[str, Any]],
    facts: dict[str, Any],
    style: str,
) -> str:
    """Generate, score, and optionally repair captions against visual evidence."""
    candidates: list[str] = []
    try:
        primary = await _call_style_provider(messages, style)
    except Exception as e:  # noqa: BLE001
        log.warning("primary style candidate failed (%s): %s", style, e)
        primary = ""
    if primary and not _evidence_issues(style, primary, facts):
        return primary
    if primary:
        candidates.append(primary)

    for i in range(1, STYLE_CANDIDATES):
        candidate_messages = list(messages)
        candidate_messages.append(
            {
                "role": "user",
                "content": (
                    "Produce an alternate version with the same style, but use "
                    "different wording while preserving concrete visual evidence."
                ),
            }
        )
        try:
            candidates.append(await _call_style_provider(candidate_messages, style))
        except Exception as e:  # noqa: BLE001
            log.warning("style candidate failed (%s/%s): %s", style, i + 1, e)

    fallback = fallback_caption(style, facts)
    if not _evidence_issues(style, fallback, facts):
        candidates.append(fallback)

    if not candidates:
        return fallback_caption(style, facts)

    ranked = sorted(
        candidates,
        key=lambda caption: _evidence_score(style, caption, facts),
        reverse=True,
    )
    best = ranked[0]
    issues = _evidence_issues(style, best, facts)
    if issues and STYLE_REPAIR_ENABLED:
        repaired = await _repair_caption(best, facts, style, issues)
        if _evidence_score(style, repaired, facts) >= _evidence_score(style, best, facts):
            best = repaired
        if _evidence_issues(style, best, facts):
            deterministic = _deterministic_evidence_caption(style, best, facts)
            if _evidence_score(style, deterministic, facts) > _evidence_score(style, best, facts):
                best = deterministic
    return best


async def _repair_caption(
    caption: str,
    facts: dict[str, Any],
    style: str,
    issues: list[str],
) -> str:
    prompt_bundle = STYLE_PROMPTS.get(style)
    if prompt_bundle is None:
        return caption
    system_prompt, _, _ = prompt_bundle
    evidence = ", ".join(_evidence_terms(facts)[:14])
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\nRepair mode: improve the caption without "
                "inventing facts. Keep exactly the requested style."
            ),
        },
        {
            "role": "user",
            "content": (
                "Scene facts:\n"
                f"{json.dumps(facts, ensure_ascii=False)}\n\n"
                f"Caption to repair:\n{caption}\n\n"
                f"Detected issues: {', '.join(issues)}.\n"
                f"Useful visual evidence terms: {evidence}.\n"
                "Rewrite as one richer caption. Preserve at least three visible "
                "details when available. Return only the caption."
            ),
        },
    ]
    try:
        return await _call_style_provider(messages, style)
    except Exception as e:  # noqa: BLE001
        log.warning("style repair failed (%s): %s", style, e)
        return caption


async def _repair_with_sibling_context(
    captions: dict[str, str],
    facts: dict[str, Any],
    styles: list[str],
) -> dict[str, str]:
    repaired = dict(captions)
    if not any(_provider_endpoint(provider, style=True)[1] for provider in _provider_order("style")):
        return repaired
    sibling_context = "\n".join(
        f"- {style}: {caption}" for style, caption in captions.items() if caption.strip()
    )
    for style in styles:
        caption = repaired.get(style, "")
        issues = _evidence_issues(style, caption, facts)
        if not issues:
            continue
        prompt_bundle = STYLE_PROMPTS.get(style)
        if prompt_bundle is None:
            continue
        system_prompt, _, _ = prompt_bundle
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\nCross-style repair mode: use the scene "
                    "facts first. The sibling captions are draft evidence from "
                    "the same video; use them only to recover concrete visual "
                    "details, not to copy their tone."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Scene facts:\n"
                    f"{json.dumps(facts, ensure_ascii=False)}\n\n"
                    f"Caption to repair for style {style}:\n{caption}\n\n"
                    f"Sibling draft captions:\n{sibling_context}\n\n"
                    f"Detected issues: {', '.join(issues)}.\n"
                    "Return one richer caption in the requested style only."
                ),
            },
        ]
        try:
            candidate = await _call_style_provider(messages, style)
        except Exception as e:  # noqa: BLE001
            log.warning("cross-style repair failed (%s): %s", style, e)
            continue
        if _evidence_score(style, candidate, facts) > _evidence_score(style, caption, facts):
            repaired[style] = candidate
    return repaired


_STOPWORDS = {
    "with", "from", "that", "this", "there", "their", "while", "through",
    "under", "above", "into", "onto", "over", "short", "video", "clip",
    "scene", "visible", "shows", "appears", "around", "toward", "towards",
    "present", "factual", "frame", "frames", "camera",
}


def _evidence_terms(facts: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in (
        "summary",
        "setting",
        "camera",
        "temporal_progression",
        "audio_hint",
    ):
        value = facts.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in (
        "subjects",
        "actions",
        "visual_details",
        "fine_grained_observations",
        "salient_objects",
        "spatial_relations",
    ):
        value = facts.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
    raw = " ".join(parts).lower().replace("/", " ")
    terms = [
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", raw)
        if token not in _STOPWORDS and len(token) >= 4
    ]
    return list(dict.fromkeys(terms))


# Framing/motion assertions the VLM gets wrong often and the judge barely
# rewards — excluded from formal background enrichment to avoid contradictions
# like "static shot" / "medium shot" / "no visible change" on a moving camera.
_FRAMING_NOISE = re.compile(
    r"\b(static|medium|wide|close-?up|eye-?level|low[- ]angle|high[- ]angle|"
    r"no (movement|visible change)|does not change|remains (static|unchanged)|shot)\b",
    re.IGNORECASE,
)


def _evidence_phrases(facts: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for key in ("visual_details", "fine_grained_observations", "salient_objects", "spatial_relations"):
        value = facts.get(key)
        if isinstance(value, list):
            phrases.extend(str(item).strip().rstrip(".") for item in value if str(item).strip())
    for key in ("camera", "temporal_progression", "setting"):
        value = facts.get(key)
        if isinstance(value, str) and value.strip():
            phrases.append(value.strip().rstrip("."))
    cleaned: list[str] = []
    for phrase in phrases:
        if 2 <= len(phrase.split()) <= 12 and phrase.lower() not in {p.lower() for p in cleaned}:
            cleaned.append(phrase)
    return cleaned


def _background_phrases(facts: dict[str, Any]) -> list[str]:
    """Object/scenery phrases only (no camera-framing/motion claims), for the
    formal background sentence. Framing/motion assertions are unreliable."""
    phrases: list[str] = []
    for key in ("visual_details", "fine_grained_observations", "salient_objects", "spatial_relations"):
        value = facts.get(key)
        if isinstance(value, list):
            phrases.extend(str(item).strip().rstrip(".") for item in value if str(item).strip())
    setting = facts.get("setting")
    if isinstance(setting, str) and setting.strip():
        phrases.append(setting.strip().rstrip("."))
    cleaned: list[str] = []
    for phrase in phrases:
        if not (2 <= len(phrase.split()) <= 12):
            continue
        if _FRAMING_NOISE.search(phrase):
            continue
        if phrase.lower() not in {p.lower() for p in cleaned}:
            cleaned.append(phrase)
    return cleaned


def _deterministic_evidence_caption(
    style: str,
    caption: str,
    facts: dict[str, Any],
) -> str:
    if style != "formal":
        return caption
    summary = facts.get("summary")
    base = summary.strip().rstrip(".") if isinstance(summary, str) and summary.strip() else caption.strip().rstrip(".")
    phrases = [phrase for phrase in _evidence_phrases(facts) if phrase.lower() not in base.lower()]
    if not phrases:
        return caption
    additions = phrases[:3]
    if len(additions) == 1:
        detail = additions[0]
    elif len(additions) == 2:
        detail = f"{additions[0]} and {additions[1]}"
    else:
        detail = f"{additions[0]}, {additions[1]}, and {additions[2]}"
    repaired = f"{base}, with {detail}."
    return repaired[:300].strip()


def _caption_words(caption: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", caption.lower().replace("/", " "))
        if token not in _STOPWORDS
    }


def _word_count(caption: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", caption))


def _evidence_issues(style: str, caption: str, facts: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    words = _caption_words(caption)
    evidence = set(_evidence_terms(facts))
    anchors = len(words & evidence)
    min_words = 22 if style == "formal" else 18
    min_anchors = 3 if style == "formal" else 2
    if _word_count(caption) < min_words:
        issues.append("too_short")
    if anchors < min_anchors:
        issues.append("missing_visual_anchors")
    if style == "humorous_tech" and not (words & {
        "api", "queue", "latency", "scheduler", "cache", "deploy",
        "production", "commit", "runtime", "server", "rollback", "logs",
        "staging", "pipeline", "code",
    }):
        issues.append("missing_clear_tech_reference")
    return issues


def _evidence_score(style: str, caption: str, facts: dict[str, Any]) -> float:
    if not caption_passes_style_filter(style, caption):
        return -10.0
    words = _caption_words(caption)
    evidence = set(_evidence_terms(facts))
    anchors = len(words & evidence)
    count = _word_count(caption)
    target = 30 if style == "formal" else 24
    length_score = min(count / target, 1.0)
    anchor_score = min(anchors / (4 if style == "formal" else 3), 1.0)
    penalty = 0.2 * len(_evidence_issues(style, caption, facts))
    return length_score + anchor_score - penalty
