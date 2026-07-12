from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

EXPECTED_INDEX = "sha256:161efc8b098a6a46f395f01fb83ce7c41a9e71c61a51205b59934393bac5f19d"
EXPECTED_AMD64 = "sha256:1b39f65c4b99642a318b353a2e0281c4d5ddb4d346510766e40c27ae5ed0ac07"


def validate_v36_manifest(payload: dict[str, object]) -> str:
    if payload.get("index_digest") != EXPECTED_INDEX:
        raise ValueError("v36 index digest mismatch")

    children = payload.get("manifests")
    if not isinstance(children, list):
        raise ValueError("v36 linux/amd64 child mismatch")

    platform = {"os": "linux", "architecture": "amd64"}
    if len(children) != 1:
        raise ValueError("v36 linux/amd64 child mismatch")

    child = children[0]
    if (
        not isinstance(child, dict)
        or child.get("platform") != platform
        or child.get("digest") != EXPECTED_AMD64
    ):
        raise ValueError("v36 linux/amd64 child mismatch")
    return EXPECTED_AMD64


def sanitized_manifest(payload: dict[str, object]) -> dict[str, object]:
    child_digest = validate_v36_manifest(payload)
    return {
        "index_digest": EXPECTED_INDEX,
        "manifests": [
            {
                "digest": child_digest,
                "platform": {"os": "linux", "architecture": "amd64"},
            }
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a sanitized v36 OCI manifest without registry access."
    )
    parser.add_argument("manifest", type=Path, help="Path to sanitized manifest JSON")
    parser.add_argument("--output", type=Path, help="Write a deterministic sanitized JSON copy")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest payload must be a JSON object")

        sanitized = sanitized_manifest(payload)
        if args.output is not None:
            rendered = json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
            args.output.write_text(rendered, encoding="utf-8", newline="\n")
    except (OSError, RecursionError, ValueError):
        print("baseline manifest validation failed", file=sys.stderr)
        return 2

    print(EXPECTED_AMD64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
