"""Heuristic detail-density audit for local prompt iteration.

This catches regressions where captions satisfy the hard contract and style
rubric but become too generic or visually thin for a human demo review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


MIN_WORDS = {
    "formal": 18,
    "sarcastic": 17,
    "humorous_tech": 16,
    "humorous_non_tech": 16,
}

ANCHORS = {
    "v1": {
        "traffic": {"traffic", "cars", "vehicles", "tires", "lanes", "intersection", "road", "street", "boulevard"},
        "autumn": {"golden", "yellow", "autumn", "trees", "leaves", "leaf"},
        "city": {"city", "urban", "buildings", "high-rise", "banners"},
    },
    "v2": {
        "kitten": {"kitten", "cat", "orange", "furball"},
        "foliage": {"leaves", "leafy", "foliage", "branches", "green", "garden", "forest"},
        "motion": {"walks", "walking", "strides", "toward", "camera", "lens", "paw"},
    },
    "v3": {
        "person": {"woman", "worker", "she"},
        "workspace": {"office", "desk", "laptop", "keyboard", "monitor", "screen", "mouse"},
        "visual_detail": {"beige", "orange", "plant", "green", "pink", "nails", "white", "silver"},
    },
}


def _tokens(text: str) -> set[str]:
    return {
        token.strip(".,!?;:()[]{}\"'`").lower()
        for token in text.replace("/", " ").split()
        if token.strip(".,!?;:()[]{}\"'`")
    }


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def audit_caption(task_id: str, style: str, caption: str) -> list[str]:
    issues: list[str] = []
    if _word_count(caption) < MIN_WORDS.get(style, 16):
        issues.append("too_few_words_for_detail")

    expected = ANCHORS.get(task_id)
    if expected:
        words = _tokens(caption)
        matched_groups = [
            group for group, choices in expected.items() if words & choices
        ]
        min_groups = 3 if style == "formal" else 2
        if len(matched_groups) < min_groups:
            issues.append(
                f"too_few_visual_anchor_groups:{len(matched_groups)}/{min_groups}"
            )
    return [f"[{task_id}/{style}] {issue}" for issue in issues]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = json.loads(args.results.read_text(encoding="utf-8"))
    issues: list[str] = []
    for row in rows:
        for style, caption in row.get("captions", {}).items():
            issues.extend(audit_caption(row.get("task_id", "?"), style, caption))

    if issues:
        print(f"DETAIL AUDIT WARN - {len(issues)} issue(s):")
        for issue in issues:
            print("  -", issue)
        if args.strict:
            return 2
    else:
        print(f"DETAIL AUDIT OK - {len(rows)} row(s), captions keep visual density.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
