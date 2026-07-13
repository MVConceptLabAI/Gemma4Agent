"""
Frame extraction — two strategies:

    1) uniform    — N frames evenly spaced (default, deterministic, fast)
    2) shots+topup — detect scene cuts via FFmpeg's `select='gt(scene,T)'`
                     filter, then top up with uniform samples if fewer than
                     the target N. Wins on multi-shot clips (sports, food,
                     narrative) where the interesting information is
                     concentrated around cuts.

Set FRAME_STRATEGY=shots to enable (2). Recommended once you've validated (1).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("track2.frames")

STRATEGY = os.environ.get("FRAME_STRATEGY", "uniform")
SCENE_THRESHOLD = float(os.environ.get("SCENE_THRESHOLD", "0.35"))


def ffprobe_duration(video: Path) -> float:
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


def _shot_timestamps(video: Path, threshold: float) -> list[float]:
    """Return timestamps (s) of scene cuts detected by FFmpeg."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats",
                "-i", str(video),
                "-vf", f"select='gt(scene,{threshold})',showinfo",
                "-f", "null", "-",
            ],
            capture_output=True, text=True,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("shot-detection failed: %s", e)
        return []
    # showinfo writes lines like: `... pts_time:12.345 ...` to stderr
    stamps: list[float] = []
    for line in proc.stderr.splitlines():
        idx = line.find("pts_time:")
        if idx < 0:
            continue
        try:
            val = float(line[idx + len("pts_time:"):].split()[0])
            stamps.append(val)
        except ValueError:
            pass
    return stamps


def _snap_frame(video: Path, t: float, out_path: Path, max_edge: int) -> Path | None:
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", str(round(t, 3)),
                "-i", str(video),
                "-frames:v", "1",
                "-vf", f"scale='min({max_edge},iw)':-2",
                "-q:v", "4",
                str(out_path),
                "-y",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning("frame snap at %.2fs failed: %s", t, e)
        return None
    return out_path if out_path.exists() else None


def extract(video: Path, workdir: Path, n: int, max_edge: int) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    duration = ffprobe_duration(video) or 60.0
    strategy = STRATEGY

    timestamps: list[float] = []
    if strategy == "shots":
        cuts = _shot_timestamps(video, SCENE_THRESHOLD)
        log.info("scene cuts detected: %d @threshold %.2f", len(cuts), SCENE_THRESHOLD)
        # Sample slightly AFTER each cut so we see the new shot content.
        cut_samples = [min(c + 0.3, duration - 0.5) for c in cuts]
        timestamps.extend(cut_samples)

    # Always include uniform samples so short clips or single-shot clips
    # (nature, weather, animals) also get well covered.
    step = max(duration / (n + 1), 0.1)
    for i in range(1, n + 1):
        timestamps.append(min(round(i * step, 3), duration - 0.2))

    # Sort + de-duplicate near neighbours (within 0.75 s), then cap to N.
    timestamps.sort()
    dedup: list[float] = []
    for t in timestamps:
        if not dedup or (t - dedup[-1] > 0.75):
            dedup.append(t)
    if len(dedup) > n:
        # Prefer keeping evenly spaced ones.
        idx = [int(round(i * (len(dedup) - 1) / (n - 1))) for i in range(n)]
        dedup = [dedup[i] for i in idx]

    frames: list[Path] = []
    for i, t in enumerate(dedup, 1):
        out = workdir / f"f{i:02d}.jpg"
        fp = _snap_frame(video, t, out, max_edge)
        if fp:
            frames.append(fp)
    return frames
