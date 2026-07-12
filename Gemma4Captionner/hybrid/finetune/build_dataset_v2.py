"""
Build dataset_v2.jsonl from scenes.jsonl by generating 4 style captions per scene.

Two modes:

    1) TEACHER mode (recommended before real training) — calls a big LLM
       (Fireworks Gemma 3 27B, GPT-4o, Claude) and asks for all 4 captions.
       Highest quality, small cost (~0.05 $ for 200 scenes on Gemma 3 27B).

    2) OFFLINE mode (default here, no API key needed) — synthesises the four
       captions from templates parameterised by the scene facts. Quality is
       lower than a teacher LLM but the patterns are correct: bans respected,
       lengths capped, styles clearly distinct. Use this to bootstrap the
       LoRA and then re-generate later with TEACHER mode.

Usage:
    # Offline (no API needed)
    python finetune/build_dataset_v2.py --scenes finetune/scenes.jsonl \
        --out finetune/dataset_v2.jsonl

    # Teacher LLM
    export FIREWORKS_API_KEY=fw_xxx
    python finetune/build_dataset_v2.py --scenes finetune/scenes.jsonl \
        --out finetune/dataset_v2.jsonl \
        --mode teacher --model accounts/fireworks/models/gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# OFFLINE style writers — one function per style. Each returns a caption
# grounded in the scene facts. Patterns come straight from the reddit
# consensus on what "style match" means to LLM judges: length ≤ 40 words,
# 1-2 sentences, cross-style bans respected.
# ---------------------------------------------------------------------------

_SUB_NORMALISE = {
    "wildflowers": "wildflowers",
    "wildflowers,sky": "wildflowers",
}


def _joined(items: list[str], max_n: int = 2) -> str:
    """Human-readable list, cap at 2 to keep captions tight."""
    items = items[:max_n]
    if not items:
        return "the scene"
    if len(items) == 1:
        return items[0]
    return f"{items[0]} and {items[1]}"


# --- FORMAL --------------------------------------------------------------
_FORMAL_OPENERS = [
    "The footage captures",
    "The clip presents",
    "The scene shows",
    "The recording depicts",
    "The sequence documents",
]

def formal(scene: dict, rng: random.Random) -> str:
    opener = rng.choice(_FORMAL_OPENERS)
    subjects = _joined(scene["subjects"])
    action = scene["actions"][0] if scene["actions"] else "being observed"
    setting = scene["setting"]
    return f"{opener} {subjects} {action} within a {setting}, filmed in an observational, documentary manner."


# --- SARCASTIC -----------------------------------------------------------
_SARC_OPENERS_A = [
    "Ah yes,",
    "Truly,",
    "Groundbreaking:",
    "Nothing says {mood} like",
    "Bold move by",
]
_SARC_OPENERS_B = [
    "Definitely worth the storage.",
    "Someone will call this content.",
    "We were all waiting for this.",
    "History will remember this moment.",
    "The world may never recover.",
]

def sarcastic(scene: dict, rng: random.Random) -> str:
    subjects = _joined(scene["subjects"])
    action = scene["actions"][0] if scene["actions"] else "existing"
    setting = scene["setting"]
    opener = rng.choice(_SARC_OPENERS_A).replace("{mood}", scene.get("mood", "life"))
    closer = rng.choice(_SARC_OPENERS_B)
    return f"{opener} {subjects} {action} in a {setting}. {closer}"


# --- HUMOROUS_TECH -------------------------------------------------------
_HTECH_METAPHORS = [
    "shipping to prod with zero rollback plan",
    "running on eventual consistency and vibes",
    "successfully passing the staging environment",
    "still waiting on the PR review from nature",
    "running at 24 FPS with no dropped frames",
    "the demo works on the first try, incredibly suspicious",
    "obviously written in a language with no null checks",
    "cache miss followed by a very expensive lookup",
    "the merge conflict was resolved off-camera",
    "hot-reloading itself between takes",
]

def humorous_tech(scene: dict, rng: random.Random) -> str:
    subjects = _joined(scene["subjects"])
    action = scene["actions"][0] if scene["actions"] else "existing"
    metaphor = rng.choice(_HTECH_METAPHORS)
    setting = scene["setting"]
    return f"{subjects} {action} in the {setting}, {metaphor}."


# --- HUMOROUS_NON_TECH ---------------------------------------------------
_HNT_JOKES = [
    "which, statistically, is the highlight of somebody's Tuesday",
    "a moment their group chat will absolutely hear about later",
    "and everyone present is pretending to be more chill than they feel",
    "and the vibe is either very charming or a small emergency, hard to tell",
    "which is somehow both the most and least dramatic thing happening today",
    "and yes, someone is definitely going to text about this",
    "with the confidence of a person who did not read the instructions",
    "and honestly, mood",
    "in what appears to be the calmest chaos ever recorded",
    "a scene that could easily double as a fridge magnet in twenty years",
]

def humorous_non_tech(scene: dict, rng: random.Random) -> str:
    subjects = _joined(scene["subjects"])
    action = scene["actions"][0] if scene["actions"] else "existing"
    joke = rng.choice(_HNT_JOKES)
    setting = scene["setting"]
    return f"{subjects} {action} in a {setting}, {joke}."


STYLE_WRITERS = {
    "formal": formal,
    "sarcastic": sarcastic,
    "humorous_tech": humorous_tech,
    "humorous_non_tech": humorous_non_tech,
}


# ---------------------------------------------------------------------------
# TEACHER mode (optional)
# ---------------------------------------------------------------------------
TEACHER_SYSTEM = """You are helping build a fine-tune dataset for a video-caption \
style-transfer model. Given a scene-facts JSON, return 4 captions — one per style — \
that respect the definitions below. Each caption must be 1-2 sentences and grounded \
in the facts (no invented events).

STYLES:
- formal: professional, objective, third-person, factual. NO jokes, NO exclamations.
- sarcastic: dry, ironic, understated, lightly mocking. NO exclamations, NO tech jargon.
- humorous_tech: genuinely funny with a clear tech/programming reference and a punchline.
- humorous_non_tech: warm everyday humour, NO computer/code/API/model/deploy references.

Return STRICT JSON, nothing else:
{"formal":"...","sarcastic":"...","humorous_tech":"...","humorous_non_tech":"..."}"""


def teacher_call(model: str, scene: dict) -> dict[str, str]:
    import httpx  # local import so offline mode doesn't need httpx
    key = os.environ["FIREWORKS_API_KEY"]
    base = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    with httpx.Client(timeout=60) as c:
        r = c.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": TEACHER_SYSTEM},
                    {"role": "user", "content": f"Scene facts:\n{json.dumps(scene)}\n\nWrite the 4 captions."},
                ],
                "temperature": 0.7,
                "max_tokens": 400,
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Post-generation validation — kills rows that violate the style bans.
# This is what makes an offline-synthesised dataset usable for LoRA.
# ---------------------------------------------------------------------------
_TECH_TERMS = (
    "prod", "api", "algorithm", "algorithms", "commit", "merge conflict",
    "rollback", "cache", "cache miss", "PR ", "pull request", "server",
    "servers", "FPS", "24 fps", "staging", "hot-reload", "hot reload",
    "null check", "eventual consistency",
)

def _validate(caption: str, style: str) -> bool:
    if not caption or len(caption) < 15 or len(caption) > 260:
        return False
    lower = caption.lower()
    if style == "humorous_non_tech":
        if any(t.lower() in lower for t in _TECH_TERMS):
            return False
    if style == "formal":
        if any(w in lower for w in ("lol", "haha", "!", "😀")):
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _iter_scenes(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["offline", "teacher"], default="offline")
    p.add_argument("--model", default="accounts/fireworks/models/gemma-3-27b-it")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    scenes = list(_iter_scenes(args.scenes))
    n_dropped = 0

    for i, sc in enumerate(scenes, 1):
        facts = {k: v for k, v in sc.items() if k != "category"}
        try:
            if args.mode == "teacher":
                caps = teacher_call(args.model, facts)
            else:
                caps = {name: fn(sc, rng) for name, fn in STYLE_WRITERS.items()}
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(scenes)}] FAIL: {e}", file=sys.stderr)
            continue

        for style, cap in caps.items():
            if not _validate(cap, style):
                n_dropped += 1
                continue
            out_rows.append({"facts": facts, "style": style, "caption": cap})

        if i % 25 == 0:
            print(f"[{i}/{len(scenes)}] cumulative rows={len(out_rows)} dropped={n_dropped}")

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows),
        encoding="utf-8",
    )
    print(f"\nWrote {len(out_rows)} rows -> {args.out}  (dropped {n_dropped} on validation)")


if __name__ == "__main__":
    main()
