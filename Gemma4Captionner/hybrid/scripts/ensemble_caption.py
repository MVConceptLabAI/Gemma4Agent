"""Ensemble captioning: multiple frontier vision models analyze the frames,
then one writer cross-references all observations and writes maximally rich
captions with NO length limit.

Flow per clip:
  1. Sample N frames.
  2. OBSERVE — K vision models each return an exhaustive JSON list of every
     concrete detail they can verify (independent, blind to each other).
  3. WRITE — one strong writer receives ALL observation lists, cross-references
     them (details seen by >=2 models are high-confidence), and writes the four
     styled captions using every detail that fits, vividly and creatively.

    python scripts/ensemble_caption.py --tasks data/sample_tasks.json \
        --out out/ensemble.json

Grounded: the writer uses only what the observers reported. Precision comes
from cross-model agreement; detection comes from the union of what they saw.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from app import pipeline as P

load_dotenv()
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OBSERVERS = os.environ.get(
    "ENSEMBLE_OBSERVERS",
    "openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.5",
).split(",")
WRITER = os.environ.get("ENSEMBLE_WRITER", "anthropic/claude-opus-4.5")

OBSERVE_SYSTEM = """You are a meticulous visual analyst. You see frames sampled
in order from ONE short video clip. Return a JSON array of SHORT strings, one
per concrete detail you can actually verify across the frames.

Be EXHAUSTIVE — list everything: each subject and its appearance (hairstyle,
clothing layers and colors, jewelry, nails, an animal's coat pattern and
markings, chest/paw color), every object and peripheral, actions and motion
(what changes across frames, direction of movement), the setting, background
structures and their count, tree color, terrain (hills/mountains), lighting and
time of day, and whether it looks like a time-lapse.

Rules: only include what is clearly visible. Do NOT state a person's race,
ethnicity, or skin color. Do NOT state a person's or animal's eye color. Do NOT
quote sign/store text unless the letters are unambiguous (say "an unreadable
sign" otherwise). Attribute a color only to the object it truly belongs to.

Return ONLY the JSON array of strings, nothing else."""

WRITE_SYSTEM = """You are a world-class video-captioning writer. Several expert
vision models each independently listed what they saw in ONE short clip. You
receive all their lists. Cross-reference them: a detail reported by two or more
models is high-confidence - use those freely. A detail from only ONE model is
UNRELIABLE: include it only if generic and safe, and DROP any single-model
SPECIFIC claim (an exact color, brand/logo, count, sign/text, or a left/right or
foreground/background placement) unless another model agrees. When models
conflict, omit the point. A wrong detail costs far more than a missing one -
when in doubt, leave it out. NEVER add anything no model reported.

Write four captions of the SAME scene, one per style. There is NO length limit
- pack in every detail that reads naturally; be vivid, specific, and creative.
Precision still matters: do not state race/skin/eye color, do not quote an
unreadable sign, attribute colors correctly.

Styles:
- formal: professional, objective, richly detailed factual description. As long
  as needed to capture everything. No jokes, no exclamations, no 1st/2nd person.
- sarcastic: dry, ironic, deadpan wit grounded in the real details; no tech jargon.
- humorous_tech: genuinely clever with tech/programming metaphors (API, latency,
  cache, pipeline, runtime, deploy, server...) tied to what is visible.
- humorous_non_tech: warm everyday humor, NO tech words, grounded in visible details.

Return STRICT JSON only: {"formal":"...","sarcastic":"...","humorous_tech":"...","humorous_non_tech":"..."}"""


async def _call(model: str, system: str, content, *, max_tokens: int = 4000) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": 0.5,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=240) as c:
        r = await c.post(
            f"{OR_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _frames_content(frames: list[Path], lead: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": lead}]
    for fp in frames:
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def _parse_list(text: str) -> list[str]:
    s = text[text.find("["): text.rfind("]") + 1]
    try:
        arr = json.loads(s)
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:  # noqa: BLE001
        return [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]


_ASCII = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "}


def _to_ascii(t: str) -> str:
    for a, b in _ASCII.items():
        t = t.replace(a, b)
    return " ".join(t.encode("ascii", "ignore").decode("ascii").split())


def _parse_obj(text: str) -> dict:
    s = text[text.find("{"): text.rfind("}") + 1]
    obj = json.loads(s)
    return {k: _to_ascii(str(v)) for k, v in obj.items()}


async def _observe(model: str, frames: list[Path]) -> tuple[str, list[str]]:
    try:
        txt = await _call(model, OBSERVE_SYSTEM, _frames_content(frames, "Frames in order:"))
        return model, _parse_list(txt)
    except Exception as e:  # noqa: BLE001
        print(f"    observer {model} failed: {str(e)[:80]}")
        return model, []


async def _one(task: dict) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        vp = await P._download(task["video_url"], wd / "c.mp4")
        frames = P._extract_keyframes(vp, wd, P.NUM_FRAMES, P.FRAME_MAX_EDGE)
        obs = await asyncio.gather(*[_observe(m, frames) for m in OBSERVERS])
        blocks = []
        for model, details in obs:
            if details:
                short = model.split("/")[-1]
                blocks.append(f"### Observations from {short} ({len(details)} details):\n"
                              + "\n".join(f"- {d}" for d in details))
        merged = "\n\n".join(blocks)
        write_content = (
            "Here are the independent observation lists from several vision models "
            "for ONE short clip. Cross-reference them and write the four captions.\n\n" + merged
        )
        raw = await _call(WRITER, WRITE_SYSTEM, write_content, max_tokens=2000)
        caps = _parse_obj(raw)
        detail_counts = {m.split("/")[-1]: len(d) for m, d in obs}
    return {"task_id": task["task_id"], "captions": caps, "_observer_details": detail_counts}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    print(f"observers: {OBSERVERS} | writer: {WRITER}")
    results = []
    for t in tasks:
        try:
            r = await _one(t)
            results.append({"task_id": r["task_id"], "captions": r["captions"]})
            print("ok", t["task_id"], "| observer details:", r["_observer_details"])
        except Exception as e:  # noqa: BLE001
            print("FAIL", t["task_id"], str(e)[:140])
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    asyncio.run(main())
