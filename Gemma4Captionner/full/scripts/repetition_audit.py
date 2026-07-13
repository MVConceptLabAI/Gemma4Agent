#!/usr/bin/env python3
"""Report exact and suspicious phrase repetition in Track 2 results JSON."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
STOP = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "with",
}


def words(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--ngram", type=int, default=4)
    parser.add_argument("--fail-on-exact", action="store_true")
    args = parser.parse_args()

    rows = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("results must be a JSON list")

    exact: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    phrases: dict[tuple[str, tuple[str, ...]], set[str]] = collections.defaultdict(set)
    for row in rows:
        task_id = str(row.get("task_id", "?"))
        captions = row.get("captions", {})
        for style in STYLES:
            caption = str(captions.get(style, "")).strip()
            exact[(style, caption.casefold())].append(task_id)
            tokens = words(caption)
            for index in range(len(tokens) - args.ngram + 1):
                gram = tuple(tokens[index : index + args.ngram])
                if sum(token not in STOP for token in gram) >= 2:
                    phrases[(style, gram)].add(task_id)

    exact_hits = [
        (style, ids, text) for (style, text), ids in exact.items() if text and len(ids) > 1
    ]
    phrase_hits = sorted(
        (
            (style, sorted(ids), " ".join(gram))
            for (style, gram), ids in phrases.items()
            if len(ids) > 1
        ),
        key=lambda item: (-len(item[1]), item[0], item[2]),
    )

    print(f"captions={len(rows) * len(STYLES)} exact_duplicates={len(exact_hits)}")
    for style, ids, text in exact_hits:
        print(f"EXACT {style} tasks={','.join(ids)} :: {text}")
    print(f"repeated_{args.ngram}grams={len(phrase_hits)}")
    for style, ids, phrase in phrase_hits[:30]:
        print(f"PHRASE {style} tasks={','.join(ids)} :: {phrase}")
    return 1 if args.fail_on_exact and exact_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
