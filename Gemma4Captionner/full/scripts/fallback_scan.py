"""Count hardcoded-fallback captions in a results.json.

Every fallback that fires in a live run is a caption NOT written from the
actual video — the single biggest accuracy risk on unseen clips. Prompt or
model changes should drive this count to zero.

Usage: python scripts/fallback_scan.py --results out/foo.json [--strict]
"""
from __future__ import annotations

import argparse
import json
import sys

# Signature substrings of every static/templated fallback in app/models.py.
SIGNATURES = [
    # FALLBACK_CAPTIONS
    "visible people or objects, movement, foreground elements",
    "waiting for this exact tiny ceremony",
    "visual runtime keeps people, motion, objects, and background context",
    "rehearsing quietly for the spotlight",
    # keyed hardcoded branches in fallback_caption()
    "grave importance of a royal inspection",
    "daily masterpiece of going somewhere slowly",
    "suspiciously calm performance review",
    "excellent obstacle detection and zero concern for documentation",
    "traffic scheduler is live in production and every lane",
    "keyboard events stream into production while the potted plant",
    "tiny manager checking whether the garden is up to standard",
    "the confidence of a tiny commute with a full audience",
    "quietly carries the room",
    "with all the ceremony of a routine pretending to be an event",
    "a small moment doing its best to become the main event",
    "looks for a grounded punchline",
    # templated evidence-word fallbacks
    "Production just received",
    "because apparently this moment required the full documentary treatment",
    "a tiny spotlight, like it arrived five minutes early",
    "The clip shows visible",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--strict", action="store_true", help="exit 1 if any fallback found")
    args = p.parse_args()

    rows = json.loads(open(args.results, encoding="utf-8").read())
    hits = [
        (row["task_id"], style, cap[:70])
        for row in rows
        for style, cap in row["captions"].items()
        if any(sig.lower() in cap.lower() for sig in SIGNATURES)
    ]
    total = sum(len(r["captions"]) for r in rows)
    for task, style, cap in hits:
        print(f"FALLBACK [{task}/{style}] {cap}...")
    print(f"FALLBACK SCAN - {len(hits)}/{total} captions are fallbacks.")
    return 1 if (hits and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
