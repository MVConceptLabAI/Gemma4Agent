"""Deterministic caption quality audit.

This complements self_check.py. self_check enforces hard submission invariants;
quality_audit catches weaker "AI slop" signals that can lower an LLM judge score:
generic filler, stale metaphors, unsafe taste, and style bleed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


AI_TELL_PHRASES = {
    "a modest triumph for moving pictures",
    "just enough visual data",
    "vibe check",
    "symphony of",
    "existential dread",
    "navigate the complexities",
    "in the realm of",
    "rich tapestry",
    "stands as a testament",
    "plays a crucial role",
    "elevate",
    "seamless",
    "cutting-edge",
    "game-changer",
    "probably",
    "maybe",
    "perhaps",
    "apparently",
}

LOW_TASTE_PHRASES = {
    "cog",
    "specimen",
    "squirrel",
    "rat",
    "body",
    "ugly",
    "fat",
    "afro",
    "skin tone",
    "race",
    "ethnicity",
}

GENERIC_CAPTION_PATTERNS = [
    re.compile(r"\ba short video clip\b", re.I),
    re.compile(r"\bvisible subjects and (actions|activity)\b", re.I),
    re.compile(r"\bvarious (features|objects|elements)\b", re.I),
    re.compile(r"\bthe scene (shows|depicts)\b", re.I),
]

TECH_TERMS = {
    "api",
    "algorithm",
    "cache",
    "code",
    "commit",
    "deploy",
    "docker",
    "fps",
    "frames",
    "git",
    "ide",
    "kubernetes",
    "latency",
    "logs",
    "pipeline",
    "prod",
    "production",
    "queue",
    "rollback",
    "runtime",
    "scheduler",
    "server",
    "staging",
}


def _words(text: str) -> set[str]:
    return {
        token.strip(".,!?;:()[]{}\"'`").lower()
        for token in text.replace("/", " ").replace("-", " ").split()
    }


def audit_caption(task_id: str, style: str, caption: str) -> list[str]:
    issues: list[str] = []
    low = caption.lower()
    if any(phrase in low for phrase in AI_TELL_PHRASES):
        issues.append("ai_tell_phrase")
    if any(phrase in low for phrase in LOW_TASTE_PHRASES):
        issues.append("low_taste_or_sensitive_phrase")
    if any(pattern.search(caption) for pattern in GENERIC_CAPTION_PATTERNS):
        issues.append("generic_caption")
    if style == "humorous_non_tech" and (_words(caption) & TECH_TERMS):
        issues.append("non_tech_style_bleed")
    if style == "sarcastic" and (_words(caption) & TECH_TERMS):
        issues.append("sarcastic_tech_bleed")
    if style == "humorous_tech" and not (_words(caption) & TECH_TERMS):
        issues.append("tech_style_missing_tech_reference")
    if style == "formal" and len(caption.split()) < 10:
        issues.append("formal_too_thin")
    if style != "formal" and len(caption.split()) < 7:
        issues.append("joke_too_thin")
    return [f"[{task_id}/{style}] {issue}" for issue in issues]


def main() -> None:
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
        print(f"QUALITY AUDIT WARN - {len(issues)} issue(s):")
        for issue in issues:
            print("  -", issue)
        if args.strict:
            sys.exit(2)
    else:
        print(f"QUALITY AUDIT OK - {len(rows)} row(s), no quality warnings.")


if __name__ == "__main__":
    main()
