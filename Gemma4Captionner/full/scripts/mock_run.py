"""
Full pipeline mock: shows the entire flow end-to-end WITHOUT any API call.

Useful to:
    - Sanity-check the JSON contract offline
    - Preview what the harness will see if your credentials are missing
    - Compare offline caption quality (from build_dataset_v2) against
      what your real Fireworks-backed pipeline would produce

Usage:
    python scripts/mock_run.py --tasks data/sample_tasks.json --out out/mock_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Make `app` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.build_dataset_v2 import STYLE_WRITERS  # noqa: E402


def _mock_facts_for(task: dict) -> dict:
    """Fake scene facts derived from task_id — enough to exercise STYLE_WRITERS."""
    return {
        "summary": f"A short mock scene for task {task['task_id']}.",
        "setting": "unknown environment",
        "subjects": ["a subject", "another element"],
        "actions": ["moving"],
        "mood": "neutral",
        "audio_hint": "silence",
        "tech_visible": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    tasks = json.loads(args.tasks.read_text())
    results = []
    for t in tasks:
        facts = _mock_facts_for(t)
        styles = t.get("styles") or list(STYLE_WRITERS.keys())
        # STYLE_WRITERS expects a scene dict with 'category' — synthesise it.
        scene = {**facts, "category": "mock"}
        captions = {s: STYLE_WRITERS[s](scene, rng) for s in styles if s in STYLE_WRITERS}
        # Fill any unknown-style keys with a neutral placeholder.
        for s in styles:
            captions.setdefault(s, "")
        results.append({"task_id": t["task_id"], "captions": captions})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} mock result(s) -> {args.out}")


if __name__ == "__main__":
    main()
