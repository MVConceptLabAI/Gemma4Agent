from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    return proc.returncode == 0, output


def _score_summary(scores_path: Path) -> dict[str, object]:
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if not scores:
        return {"count": 0, "accuracy": 0, "style_match": 0, "final": 0, "weakest": []}
    rows = [
        {
            **row,
            "final": (float(row.get("accuracy", 0)) + float(row.get("style_match", 0))) / 2,
        }
        for row in scores
    ]
    return {
        "count": len(rows),
        "accuracy": mean(float(row.get("accuracy", 0)) for row in rows),
        "style_match": mean(float(row.get("style_match", 0)) for row in rows),
        "final": mean(float(row["final"]) for row in rows),
        "weakest": sorted(rows, key=lambda row: row["final"])[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="out/demo_quality_results.json")
    parser.add_argument("--scores", default="eval/scores_quality_openrouter.json")
    parser.add_argument("--min-final", type=float, default=0.93)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results = Path(args.results)
    scores = Path(args.scores)
    ok = True

    checks: list[tuple[str, bool, str]] = []
    for name, cmd in (
        ("self_check", [PY, "eval/self_check.py", "--results", str(results)]),
        ("quality_audit", [PY, "eval/quality_audit.py", "--results", str(results), "--strict"]),
        ("detail_audit", [PY, "eval/detail_audit.py", "--results", str(results), "--strict"]),
        ("grounding_audit", [PY, "eval/grounding_audit.py", "--results", str(results), "--strict"]),
    ):
        passed, output = _run(cmd)
        checks.append((name, passed, output))
        ok &= passed

    summary: dict[str, object] | None = None
    if scores.exists():
        summary = _score_summary(scores)
        score_ok = float(summary["final"]) >= args.min_final
        checks.append(
            (
                "judge_score",
                score_ok,
                (
                    f"final={float(summary['final']):.3f} "
                    f"accuracy={float(summary['accuracy']):.3f} "
                    f"style={float(summary['style_match']):.3f} "
                    f"n={summary['count']}"
                ),
            )
        )
        ok &= score_ok
    else:
        checks.append(("judge_score", not args.strict, f"missing {scores}"))
        ok &= not args.strict

    print("== Quality Gate")
    for name, passed, output in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}")
        if output:
            for line in output.splitlines()[:12]:
                print(f"  {line}")

    if summary:
        print("\n== Weakest Captions")
        for row in summary["weakest"]:  # type: ignore[index]
            print(
                f"- {row.get('task_id')}/{row.get('style')}: "
                f"{float(row.get('final', 0)):.2f} - {row.get('reason', '')}"
            )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
