from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aireceipes.catalog import Catalog, DEFAULT_RECIPES_DIR  # noqa: E402
from aireceipes.suite import RecipeRunReport, build_comparison_html, build_comparison_payload  # noqa: E402


def _run_id_from_suite_dir(suite_dir: Path) -> str:
    name = suite_dir.name
    suffix = "-gemma-variant-bakeoff"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate comparison.json/html from existing per-recipe metrics.json files.")
    parser.add_argument("suite_dir", type=Path, help="Suite folder containing 01-*/metrics.json, 02-*/metrics.json, ...")
    parser.add_argument("--recipes-dir", type=Path, default=DEFAULT_RECIPES_DIR)
    parser.add_argument("--run-id", help="Override run id written into comparison artifacts.")
    args = parser.parse_args()

    suite_dir = args.suite_dir.resolve()
    metric_paths = sorted(suite_dir.glob("[0-9][0-9]-*/metrics.json"))
    if not metric_paths:
        raise SystemExit(f"No per-recipe metrics found under {suite_dir}")

    catalog = Catalog(args.recipes_dir)
    recipe_runs: list[RecipeRunReport] = []
    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recipe_id = str(metrics["recipe_id"])
        try:
            recipe_name = catalog.get(recipe_id).name
        except Exception:
            recipe_name = recipe_id
        lifecycle = metrics.get("lifecycle", {}) if isinstance(metrics.get("lifecycle"), dict) else {}
        recipe_runs.append(
            RecipeRunReport(
                recipe_id=recipe_id,
                recipe_name=recipe_name,
                run_dir=metrics_path.parent,
                metrics_path=metrics_path,
                metrics=metrics,
                lifecycle=lifecycle,
            )
        )

    comparison = build_comparison_payload(recipe_runs, run_id=args.run_id or _run_id_from_suite_dir(suite_dir))
    comparison_json_path = suite_dir / "comparison.json"
    comparison_html_path = suite_dir / "comparison.html"
    comparison_json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    comparison_html_path.write_text(build_comparison_html(comparison), encoding="utf-8")
    print(f"Comparison JSON: {comparison_json_path}")
    print(f"Comparison HTML: {comparison_html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
