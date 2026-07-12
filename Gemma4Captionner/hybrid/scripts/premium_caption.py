"""End-to-end caption generation with a single strong multimodal model.

Bypasses the two-stage Qwen->Gemma pipeline: one top VLM reads the sampled
frames and writes all four styled captions directly. Used to A/B whether a
stronger model gives more precise + more detailed captions than the pipeline.

    python scripts/premium_caption.py --model google/gemini-2.5-pro \
        --tasks data/sample_tasks.json --out out/gempro_e2e.json

Frames + normalization + style filters are reused from the pipeline so output
stays contract-valid and safe.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

from app import pipeline as P
from app.models import normalize_captions, REQUIRED_STYLES

load_dotenv()

OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

SYSTEM = """You are an expert video-captioning system. You see frames sampled in
order from ONE short clip. Produce four captions of the SAME scene in four
distinct styles.

ACCURACY IS SCORED. Rules:
- Describe only what is clearly visible. Do NOT guess.
- Pack in EVERY concrete detail you can actually verify: subjects and their
  appearance (hairstyle, jewelry, clothing layers+colors, nail color, an
  animal's coat pattern/markings), objects they use, background structures,
  tree color, terrain (hills/mountains), signage, lighting, and whether the
  footage looks like a time-lapse. Richer grounded captions score higher.
- Quote sign/storefront text ONLY if you can read it with certainty; if the
  letters are ambiguous, describe the sign without quoting it (a wrong quote is
  a hallucination). Never invent a company name.
- Do NOT state a person's or animal's eye color (lighting makes it unreliable).
- Attribute colors to the right object (a colored jersey does not make the
  gloves that color); if unsure of a color, omit it.
- Do NOT claim people/occupants, brands, or equipment that are not visible.
- NEVER state a person's race, ethnicity, or skin color, and never mock
  appearance. Describe clothing, hair style, and accessories instead.
- English only. Plain ASCII punctuation.

STYLES:
- formal: professional, objective, factual. Two full sentences, ~40-55 words,
  packing the maximum verified detail (subject+appearance+action, then setting
  +background). No jokes, no exclamations, no first/second person, no tech jargon.
- sarcastic: dry, ironic, deadpan; grounded in real visible detail; no
  exclamations, no tech jargon.
- humorous_tech: genuinely funny with ONE clear tech/programming metaphor
  (API, latency, cache, pipeline, runtime, server, deploy, production...) tied
  to a visible thing; still names the real subject.
- humorous_non_tech: warm everyday humor, NO tech words at all; grounded in a
  visible detail.

Return STRICT JSON only: {"formal": "...", "sarcastic": "...",
"humorous_tech": "...", "humorous_non_tech": "..."}"""


def _downsample_video(src: Path, dst: Path, seconds: int = 12) -> Path:
    """Small mp4 for native-video models: fps=2, 480-wide, no audio."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-t", str(seconds), "-vf", "fps=2,scale=480:-2", "-an", str(dst)],
        check=True,
    )
    return dst


async def _caption(model: str, media, *, video: bool = False) -> dict:
    if video:
        b64 = base64.b64encode(Path(media).read_bytes()).decode("ascii")
        content: list[dict] = [
            {"type": "text", "text": "This is a short video clip."},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        ]
    else:
        content = [{"type": "text", "text": "Frames in order:"}]
        for fp in media:
            b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "temperature": 0.4,
        "max_tokens": 4000,
    }
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(
            f"{OR_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    obj = P._extract_json_object(text) or json.loads(text[text.find("{"): text.rfind("}") + 1])
    return obj


async def _one(model: str, task: dict, video: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        vp = await P._download(task["video_url"], wd / "c.mp4")
        if video:
            media = _downsample_video(vp, wd / "small.mp4")
        else:
            media = P._extract_keyframes(vp, wd, P.NUM_FRAMES, P.FRAME_MAX_EDGE)
        raw = await _caption(model, media, video=video)
        caps = normalize_captions({k: raw.get(k, "") for k in REQUIRED_STYLES}, list(REQUIRED_STYLES), None)
    return {"task_id": task["task_id"], "captions": caps}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true", help="send the real (downsampled) video, not frames")
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    results = []
    for t in tasks:
        try:
            results.append(await _one(args.model, t, args.video))
            print("ok", t["task_id"])
        except Exception as e:  # noqa: BLE001
            print("FAIL", t["task_id"], str(e)[:120])
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    asyncio.run(main())
