"""Offline guard for the Gemma 4 creative-writer routing experiment."""

from __future__ import annotations

import importlib
import os
import sys


def main() -> None:
    original = dict(os.environ)
    try:
        os.environ.update(
            {
                "ENSEMBLE_WRITER": "anthropic/claude-opus-4.5",
                "W4_FORMAL_WRITER": "anthropic/claude-opus-4.5",
                "W4_SARCASTIC_WRITER": "google/gemma-4-31b-it",
                "W4_HUMOROUS_TECH_WRITER": "google/gemma-4-31b-it",
                "W4_HUMOROUS_NON_TECH_WRITER": "google/gemma-4-31b-it",
            }
        )
        sys.modules.pop("app.ensemble", None)
        ensemble = importlib.import_module("app.ensemble")
        assert ensemble.W4_STYLE_WRITERS == {
            "formal": "anthropic/claude-opus-4.5",
            "sarcastic": "google/gemma-4-31b-it",
            "humorous_tech": "google/gemma-4-31b-it",
            "humorous_non_tech": "google/gemma-4-31b-it",
        }
        print("w4_hybrid_writers_ok")
    finally:
        os.environ.clear()
        os.environ.update(original)
        sys.modules.pop("app.ensemble", None)


if __name__ == "__main__":
    main()
