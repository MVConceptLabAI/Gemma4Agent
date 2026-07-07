from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .catalog import Catalog, DEFAULT_RECIPES_DIR
from .router import write_route_plan
from .runner import run_recipe
from .suite import DEFAULT_GEMMA_BAKEOFF_RECIPES, load_test_wrapper_command, run_bakeoff_suite


def _add_recipes_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recipes-dir",
        default=str(DEFAULT_RECIPES_DIR),
        help="Directory containing classified recipe TOML files.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aireceipes", description="Run classified AI benchmark recipes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available recipes.")
    list_parser.add_argument("--category", choices=["inference", "finetuning", "agentic"], help="Filter by recipe category.")
    _add_recipes_dir(list_parser)

    show_parser = subparsers.add_parser("show", help="Show one recipe as JSON metadata.")
    show_parser.add_argument("recipe_id")
    _add_recipes_dir(show_parser)

    run_parser = subparsers.add_parser("run", help="Run one recipe and collect KPI artifacts.")
    run_parser.add_argument("recipe_id")
    run_parser.add_argument("--output-dir", default="runs", help="Directory where run artifacts are written.")
    run_parser.add_argument("--run-id", help="Optional stable run id for repeatable output paths.")
    _add_recipes_dir(run_parser)

    bakeoff_parser = subparsers.add_parser(
        "bakeoff",
        help="Sequentially load, test, unload, and compare the Gemma 4 / MTP / DiffusionGemma recipes.",
    )
    bakeoff_parser.add_argument(
        "--recipe",
        action="append",
        help="Recipe id to include. Repeat to override the default three-model bakeoff order.",
    )
    bakeoff_parser.add_argument("--output-dir", default="runs", help="Directory where suite artifacts are written.")
    bakeoff_parser.add_argument("--run-id", help="Optional stable suite id for repeatable output paths.")
    bakeoff_parser.add_argument(
        "--wrapper-command-file",
        "--test-wrapper-cmd",
        type=Path,
        help="Optional .cmd file; non-comment lines are executed before each recipe with $result set to the suite directory.",
    )
    bakeoff_parser.add_argument(
        "--no-lifecycle-reuse",
        action="store_true",
        help="Disable reuse when adjacent recipes have the same lifecycle_key/model_memory_key.",
    )
    bakeoff_parser.add_argument(
        "--no-lifecycle",
        action="store_true",
        help="Do not run recipe start/stop commands; use already-running endpoints.",
    )
    _add_recipes_dir(bakeoff_parser)

    route_parser = subparsers.add_parser(
        "route",
        help="Build an agent route policy from one or more benchmark comparison.json files.",
    )
    route_parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="Path to comparison.json. Repeat to merge several hardware/backend benchmark runs.",
    )
    route_parser.add_argument("--workflow", default="agentic", help="Workflow/vertical to optimize, e.g. agentic, chat, code.")
    route_parser.add_argument(
        "--strategy",
        choices=["balanced", "fastest", "cheapest", "quality"],
        default="balanced",
        help="Route selection strategy.",
    )
    route_parser.add_argument("--min-accuracy", type=float, default=0.0, help="Reject candidates below this quality floor.")
    route_parser.add_argument("--output", required=True, help="Where to write the route plan JSON.")
    return parser


def _print_recipe_table(catalog: Catalog, category: str | None = None) -> None:
    recipes = catalog.list(category=category)
    if not recipes:
        print("No recipes found.")
        return
    print(f"{'ID':32} {'CATEGORY':12} {'TAGS':24} NAME")
    print(f"{'-' * 32} {'-' * 12} {'-' * 24} {'-' * 20}")
    for recipe in recipes:
        tags = ",".join(recipe.tags)
        print(f"{recipe.id:32} {recipe.category:12} {tags:24} {recipe.name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            _print_recipe_table(Catalog(Path(args.recipes_dir)), category=args.category)
            return 0
        if args.command == "show":
            recipe = Catalog(Path(args.recipes_dir)).get(args.recipe_id)
            payload = {
                "id": recipe.id,
                "category": recipe.category,
                "name": recipe.name,
                "description": recipe.description,
                "tags": recipe.tags,
                "classification": recipe.classification,
                "runtime": recipe.runtime,
                "parameters": recipe.parameters,
                "dataset": recipe.dataset,
                "kpis": recipe.kpis,
                "path": str(recipe.path),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            result = run_recipe(
                args.recipe_id,
                recipes_dir=Path(args.recipes_dir),
                output_dir=Path(args.output_dir),
                run_id=args.run_id,
            )
            print(f"Run directory: {result.run_dir}")
            print(f"Metrics: {result.metrics_path}")
            print(json.dumps(result.metrics["kpis"], indent=2, sort_keys=True))
            return 0
        if args.command == "bakeoff":
            wrapper_command = load_test_wrapper_command(args.wrapper_command_file)
            result = run_bakeoff_suite(
                args.recipe or DEFAULT_GEMMA_BAKEOFF_RECIPES,
                recipes_dir=Path(args.recipes_dir),
                output_dir=Path(args.output_dir),
                run_id=args.run_id,
                lifecycle=not args.no_lifecycle,
                test_wrapper_command=wrapper_command,
                reuse_lifecycle=not args.no_lifecycle_reuse,
            )
            print(f"Suite directory: {result.suite_dir}")
            for recipe_run in result.recipe_runs:
                print(f"Metrics ({recipe_run.recipe_id}): {recipe_run.metrics_path}")
            print(f"Comparison JSON: {result.comparison_json_path}")
            print(f"Comparison HTML: {result.comparison_html_path}")
            return 0
        if args.command == "route":
            output_path = write_route_plan(
                [Path(item) for item in args.comparison],
                Path(args.output),
                workflow=args.workflow,
                strategy=args.strategy,
                min_accuracy=args.min_accuracy,
            )
            print(f"Route plan: {output_path}")
            print(output_path.read_text(encoding="utf-8"))
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
