"""
Local LLM-Judge proxy - mimics the hackathon harness scoring so you can iterate
FAST without submitting. Two axes, 0-1 each, aggregated per caption:

    caption_accuracy - does the caption match the video content?
    style_match      - does the caption match the requested tone?

Score for a caption = (accuracy + style_match) / 2
Final score        = mean over all (clip, style) pairs

Usage:
    export FIREWORKS_API_KEY=fw_xxx
    python eval/local_judge.py \
        --results out/results.json \
        --clips   eval/clips.json \
        --judge   accounts/fireworks/models/qwen2p5-vl-32b-instruct

`clips.json` maps task_id -> {video_url, ground_truth_facts?}. Ground truth is
optional - the judge can visually inspect frames via a VLM.

This is a PROXY, not the real harness. Use it to compare stacks / prompts, not
as an absolute score.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
JUDGE_PROVIDER_ORDER = os.environ.get("JUDGE_PROVIDER_ORDER", "fireworks,openrouter,groq")
GROQ_JUDGE_MODEL = os.environ.get(
    "GROQ_JUDGE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
OPENROUTER_JUDGE_MODEL = os.environ.get(
    "OPENROUTER_JUDGE_MODEL", "qwen/qwen3-vl-8b-instruct"
)
FIREWORKS_JUDGE_MODEL = os.environ.get(
    "FIREWORKS_JUDGE_MODEL", "accounts/fireworks/models/qwen3p7-plus"
)
GROQ_MAX_IMAGES = int(os.environ.get("GROQ_MAX_IMAGES", "5"))

STYLE_DEFS = {
    "formal": "professional, objective, third-person, factual, no jokes, no jargon",
    "sarcastic": "dry, ironic, understated, lightly mocking, no exclamations, no tech jargon",
    "humorous_tech": "genuinely funny with a clear tech/programming reference and a real punchline",
    "humorous_non_tech": "funny and warm with everyday humour, NO tech jargon at all",
}


JUDGE_SYSTEM = """You are an impartial judge for a video-captioning contest.
You will see sampled frames from a short clip and one caption written in a
specific style. Score two axes independently on a 0.0-1.0 scale.

Accuracy rubric:
- 1.0: all visible claims match the clip and include useful concrete detail
- 0.7: broadly correct but generic or missing important visible detail
- 0.4: partly correct, vague, or overstates uncertain facts
- 0.0: contradicts the clip or invents central facts

Style rubric:
- 1.0: clearly matches the requested style, tasteful, concise, and natural
- 0.7: style is present but generic, safe, or weakly funny
- 0.4: mixed style, cliche, awkward, or only technically follows the style
- 0.0: wrong style, banned jargon, mean joke, or unsafe appearance joke

Return STRICT JSON, nothing else:
{{"accuracy": <0-1>, "style_match": <0-1>, "reason": "<one short sentence>"}}"""


JUDGE_USER = """Requested style: **{style}**
Style definition: {style_def}

Caption to score:
"{caption}"

Score both axes 0.0 - 1.0."""


def _sample_frames(video_url: str, n: int = 4) -> list[Path]:
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "clip.mp4"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        with client.stream(
            "GET",
            video_url,
            headers={"User-Agent": "track2-local-judge/1.0"},
        ) as r:
            r.raise_for_status()
            with video.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    # duration
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        text=True,
    ).strip()
    dur = float(out) if out else 60.0
    step = max(dur / (n + 1), 0.1)
    frames: list[Path] = []
    for i in range(1, n + 1):
        t = round(i * step, 3)
        fp = tmp / f"f{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", str(t), "-i", str(video),
             "-frames:v", "1", "-vf", "scale='min(640,iw)':-2",
             "-q:v", "5", str(fp)],
            check=True,
        )
        frames.append(fp)
    return frames


async def _judge_one(
    client: httpx.AsyncClient,
    model: str,
    frames: list[Path],
    caption: str,
    style: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": JUDGE_USER.format(
                style=style, style_def=STYLE_DEFS[style], caption=caption
            ),
        }
    ]
    judge_frames = frames[:GROQ_MAX_IMAGES] if model.startswith("groq:") else frames
    for fp in judge_frames:
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        content.append(
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    payload = {
        "model": _provider_model_name(model),
        "messages": [
            {
                "role": "system",
                "content": (
                    JUDGE_SYSTEM
                    + "\n\nCritical output rule: do not show reasoning, markdown, "
                    "analysis, or prose outside JSON. The first character must be {."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 800 if not (model.startswith("openrouter:") or model.startswith("groq:")) else 200,
        "response_format": {"type": "json_object"},
    }
    base_url, api_key = _judge_endpoint_for_model(model)
    r = await client.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    obj = _extract_json_object(txt)
    if obj is None:
        # Fallback: try to salvage two floats via regex.
        m = re.search(
            r'"accuracy"\s*:\s*([0-9.]+).*"style_match"\s*:\s*([0-9.]+)', txt, re.S
        )
        obj = (
            {"accuracy": float(m.group(1)), "style_match": float(m.group(2)), "reason": ""}
            if m
            else None
        )
    if obj is None or not _valid_score_object(obj):
        raise ValueError("judge returned invalid JSON score")
    return obj


def _valid_score_object(obj: dict[str, Any]) -> bool:
    try:
        accuracy = float(obj["accuracy"])
        style_match = float(obj["style_match"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= accuracy <= 1 and 0 <= style_match <= 1


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and {"accuracy", "style_match"} <= set(obj):
            return obj
    return None


def _judge_models(default_model: str) -> list[str]:
    models: list[str] = []
    for provider in (p.strip().lower() for p in JUDGE_PROVIDER_ORDER.split(",")):
        if provider == "fireworks" and FIREWORKS_API_KEY:
            models.append(FIREWORKS_JUDGE_MODEL or default_model)
        elif provider == "openrouter" and OPENROUTER_API_KEY:
            models.append(f"openrouter:{OPENROUTER_JUDGE_MODEL}")
        elif provider == "groq" and GROQ_API_KEY:
            models.append(f"groq:{GROQ_JUDGE_MODEL}")
    return models


def _judge_endpoint_for_model(model: str) -> tuple[str, str]:
    if model.startswith("openrouter:"):
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY
    if model.startswith("groq:"):
        return GROQ_BASE_URL, GROQ_API_KEY
    return FIREWORKS_BASE_URL, FIREWORKS_API_KEY


def _provider_model_name(model: str) -> str:
    if model.startswith("openrouter:") or model.startswith("groq:"):
        return model.split(":", 1)[1]
    return model


async def _amain(args: argparse.Namespace) -> None:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    clip_map = {c["task_id"]: c for c in clips}

    scores: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30) as client:
        for r in results:
            tid = r["task_id"]
            if tid not in clip_map:
                print(f"[warn] {tid} not in clips.json - skipped")
                continue
            frames = _sample_frames(clip_map[tid]["video_url"])
            for style, caption in r["captions"].items():
                if not caption:
                    scores.append({"task_id": tid, "style": style, "accuracy": 0, "style_match": 0})
                    continue
                s = None
                for judge_model in _judge_models(args.judge):
                    try:
                        s = await _judge_one(client, judge_model, frames, caption, style)
                        s["judge_model"] = judge_model
                        break
                    except (httpx.HTTPError, ValueError) as e:
                        print(f"[warn] judge failed with {judge_model}: {e}")
                if s is None:
                    s = {
                        "accuracy": 0,
                        "style_match": 0,
                        "reason": "all judge providers failed",
                        "judge_model": "",
                    }
                s.update({"task_id": tid, "style": style, "caption": caption})
                scores.append(s)
                print(f"[{tid} | {style:>18}] acc={s['accuracy']:.2f} style={s['style_match']:.2f}  {s.get('reason','')[:80]}")

    if scores:
        accs = [x["accuracy"] for x in scores]
        sts = [x["style_match"] for x in scores]
        finals = [(a + b) / 2 for a, b in zip(accs, sts)]
        print("\n--- SUMMARY ---")
        print(f"Mean accuracy    : {statistics.mean(accs):.3f}")
        print(f"Mean style match : {statistics.mean(sts):.3f}")
        print(f"Mean final       : {statistics.mean(finals):.3f}")
        print(f"Captions scored  : {len(scores)}")

    Path(args.out).write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetail -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to your results.json")
    p.add_argument("--clips", required=True, help="Path to clips.json (task_id -> video_url)")
    p.add_argument("--judge", default=FIREWORKS_JUDGE_MODEL)
    p.add_argument("--out", default="eval/scores.json")
    args = p.parse_args()
    if not _judge_models(args.judge):
        raise SystemExit(
            "Set one judge key: FIREWORKS_API_KEY, OPENROUTER_API_KEY, or GROQ_API_KEY"
        )
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
