"""
Grow the seed dataset from ~40 rows to 200+ using a teacher LLM (GPT-4o, Claude,
or Fireworks' best model). Reads a list of NEW scene-fact JSONs you invent and
asks the teacher to produce all 4 style captions for each.

    python finetune/augment_dataset.py \
        --scenes finetune/new_scenes.jsonl \
        --model  accounts/fireworks/models/gemma-3-27b-it \
        --out    finetune/dataset_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")

TEACHER_SYSTEM = """You are helping build a fine-tune dataset for a video-caption \
style-transfer model. Given a scene-facts JSON, return 4 captions — one per style — \
that respect the definitions below. Each caption must be 1-2 sentences and \
grounded in the facts (no invented events).

STYLES:
- formal: professional, objective, third-person, factual. NO jokes.
- sarcastic: dry, ironic, understated, lightly mocking. NO exclamations, NO tech jargon.
- humorous_tech: genuinely funny with a clear tech/programming reference and a punchline.
- humorous_non_tech: warm everyday humour, NO computer/code/API/model/deploy references.

Return STRICT JSON:
{"formal":"...", "sarcastic":"...", "humorous_tech":"...", "humorous_non_tech":"..."}"""


def _call(client: httpx.Client, model: str, facts: dict) -> dict:
    r = client.post(
        f"{FIREWORKS_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": f"Scene facts:\n{json.dumps(facts)}\n\nWrite the 4 captions."},
            ],
            "temperature": 0.7,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=Path, required=True,
                   help="JSONL of NEW scene-fact objects (one per line)")
    p.add_argument("--model", default="accounts/fireworks/models/gemma-3-27b-it")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not FIREWORKS_API_KEY:
        print("FIREWORKS_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    scenes = [json.loads(l) for l in args.scenes.read_text().splitlines() if l.strip()]
    out_rows: list[dict] = []
    with httpx.Client() as client:
        for i, facts in enumerate(scenes, 1):
            try:
                caps = _call(client, args.model, facts)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(scenes)}] FAILED: {e}", file=sys.stderr)
                continue
            for style, cap in caps.items():
                out_rows.append({"facts": facts, "style": style, "caption": cap})
            print(f"[{i}/{len(scenes)}] +4 rows")

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
