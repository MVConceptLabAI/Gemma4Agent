"""Public-sample grounding audit.

This is intentionally conservative. It catches terms that looked plausible to a
text judge but are unsupported, ambiguous, or too distracting after inspecting
the actual public sample frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


UNSUPPORTED_TERMS = {
    "v1": {
        "mountain",
        "mountains",
        "gps",
        "coffee shop",
        "exactly",
    },
    "v2": {
        "breed",
        "persian",
        "maine coon",
        "green eyes",
        "blue eyes",
        "emerald",
        "emeralds",
        "snack",
        "photobooth",
        "selfie",
    },
    "v3": {
        "code",
        "commit log",
        "ci/cd",
        "caffeine",
        "lunch",
        "snack",
        "red nails",
        "blue nails",
    },
}


def audit_caption(task_id: str, style: str, caption: str) -> list[str]:
    issues: list[str] = []
    low = caption.lower()
    for term in UNSUPPORTED_TERMS.get(task_id, set()):
        if term in low:
            issues.append(f"unsupported_or_distracting_term:{term}")
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
        print(f"GROUNDING AUDIT WARN - {len(issues)} issue(s):")
        for issue in issues:
            print("  -", issue)
        if args.strict:
            return 2
    else:
        print(f"GROUNDING AUDIT OK - {len(rows)} row(s), no known unsupported sample terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
