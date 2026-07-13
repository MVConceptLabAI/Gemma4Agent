"""
Optional Whisper transcription — big score booster on `humorous_tech` and
`humorous_non_tech` because jokes almost always live in the voice track.

Wire it in by setting env WHISPER_ENABLED=1 in the container.

Trade-offs:
    - faster-whisper `small` (~460 MB): fits any laptop, ~3 s per clip, decent
      quality on clean speech. Recommended default.
    - `large-v3` (~1.5 GB, 3 GB with cudnn): SOTA quality, needs a GPU for
      real-time. Use only if you deploy on AMD Developer Cloud.
    - Alternative: skip local Whisper entirely, POST the audio to Groq's
      Whisper endpoint — 0 GB in the image, ~1 s roundtrip. Requires
      GROQ_API_KEY.

This module returns "" on any failure so the caller never breaks.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("track2.audio")

# Read env once at import time; explicit-off keeps the image small.
_ENABLED = os.environ.get("WHISPER_ENABLED", "0") == "1"
_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")  # tiny/base/small/medium/large-v3
_MODEL = None  # lazy-loaded on first call


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from faster_whisper import WhisperModel
        # int8 on CPU keeps startup < 60s even for `small`.
        _MODEL = WhisperModel(_MODEL_SIZE, device="auto", compute_type="int8")
        log.info("Loaded faster-whisper model=%s", _MODEL_SIZE)
    except Exception as e:  # noqa: BLE001
        log.warning("Whisper not available (%s) — transcription disabled", e)
        _MODEL = False
    return _MODEL


def transcribe(video_path: Path) -> str:
    """Return the audio transcript for the clip, or empty string on failure."""
    if not _ENABLED:
        return ""
    m = _load_model()
    if not m:
        return ""
    # Extract mono 16 kHz WAV — the format Whisper actually consumes.
    wav = video_path.with_suffix(".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000",
                "-f", "wav", str(wav),
                "-y",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning("ffmpeg audio extract failed: %s", e)
        return ""

    try:
        segments, _info = m.transcribe(str(wav), beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text[:600]  # cap so the VLM prompt stays small
    except Exception as e:  # noqa: BLE001
        log.warning("Whisper transcribe failed: %s", e)
        return ""
    finally:
        try:
            wav.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
