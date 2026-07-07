from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / f"aireceipes-gemma-bakeoff-portable-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.zip"

ROOT_FILES = [
    "README.md",
    "PORTABLE_BENCHMARK.md",
    "pyproject.toml",
    "Dockerfile",
    "docker-compose.yml",
]
ROOT_DIRS = ["src", "recipes", "tests", "scripts", "mcp", "skills"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", "venv", ".git", "dist", "models", "runs"}


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)


def _write_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    info = zipfile.ZipInfo.from_file(path, arcname)
    if path.suffix == ".sh":
        info.create_system = 3
        info.external_attr = (0o100755 << 16)
    with path.open("rb") as fh:
        zf.writestr(info, fh.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _write_tree(zf: zipfile.ZipFile, path: Path, prefix: str = "AiReceipes") -> int:
    count = 0
    for item in sorted(path.rglob("*")):
        if item.is_dir() or _should_skip(item):
            continue
        rel = item.relative_to(ROOT).as_posix()
        _write_file(zf, item, f"{prefix}/{rel}")
        count += 1
    return count


def _write_example_run(zf: zipfile.ZipFile, suite_dir: Path, prefix: str = "AiReceipes") -> int:
    count = 0
    suite_dir = suite_dir.resolve()
    if not suite_dir.exists():
        return 0
    allowed_suffixes = {".json", ".html", ".log", ".md", ".csv"}
    for item in sorted(suite_dir.rglob("*")):
        if item.is_dir() or item.suffix.lower() not in allowed_suffixes:
            continue
        rel = item.relative_to(suite_dir).as_posix()
        _write_file(zf, item, f"{prefix}/examples/{suite_dir.name}/{rel}")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a portable AiReceipes Gemma bakeoff ZIP without model binaries or local venvs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-run",
        type=Path,
        action="append",
        default=[],
        help="Optional completed suite dir to include as a small example report. Can be passed multiple times.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in ROOT_FILES:
            path = ROOT / name
            if path.exists():
                _write_file(zf, path, f"AiReceipes/{name}")
                count += 1
        for name in ROOT_DIRS:
            path = ROOT / name
            if path.exists():
                count += _write_tree(zf, path)
        for include_run in args.include_run:
            count += _write_example_run(zf, include_run)
    print(f"ZIP: {args.output}")
    print(f"files: {count}")
    print(f"size_bytes: {args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
