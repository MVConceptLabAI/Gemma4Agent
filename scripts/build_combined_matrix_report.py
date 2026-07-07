from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aireceipes.suite import REPORT_METRICS, _best_by_metric, build_comparison_html  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_key(model: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    return (
        model.get("mode"),
        model.get("model_family"),
        model.get("context_size"),
        model.get("reasoning"),
        model.get("mtp_draft_n"),
    )


def _is_failed(model: dict[str, Any]) -> bool:
    return (model.get("success_count") or 0) == 0 or (model.get("error_count") or 0) > 0


def _source_label(source_suite: Path) -> str:
    text = str(source_suite).lower()
    if "a6000" in text:
        return "a6000"
    if "a100" in text:
        return "a100"
    return source_suite.name.replace("-gemma-matrix-bakeoff", "")


def _copy_successful_union_model(model: dict[str, Any], *, order: int, source_suite: Path) -> dict[str, Any]:
    copied = dict(model)
    source_label = _source_label(source_suite)
    copied["order"] = order
    copied["source_suite"] = str(source_suite)
    copied["source_label"] = source_label
    recipe_id = str(copied.get("recipe_id") or f"row-{order}")
    # Keep recipe ids unique in union reports because A6000 and A100 runs have the same matrix ids.
    if not recipe_id.endswith(f"-{source_label}"):
        copied["recipe_id"] = f"{recipe_id}-{source_label}"
    target_gpu = copied.get("target_gpu") or (copied.get("lifecycle") or {}).get("target_gpu")
    if target_gpu and target_gpu not in str(copied.get("name") or ""):
        copied["name"] = f"{copied.get('name') or copied['recipe_id']} [{target_gpu}]"
    return copied


def _write_csv(comparison: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model in comparison["models"]:
        kpis = model.get("kpis", {}) if isinstance(model.get("kpis"), dict) else {}
        rows.append(
            {
                "recipe_id": model.get("recipe_id"),
                "name": model.get("name"),
                "model_family": model.get("model_family"),
                "mode": model.get("mode"),
                "backend": model.get("backend"),
                "context_size": model.get("context_size"),
                "reasoning": model.get("reasoning"),
                "mtp_draft_n": model.get("mtp_draft_n"),
                "target_gpu": model.get("target_gpu"),
                "source_label": model.get("source_label"),
                "source_suite": model.get("source_suite"),
                "success_count": model.get("success_count"),
                "error_count": model.get("error_count"),
                "prompt_count": model.get("prompt_count"),
                "accuracy": kpis.get("accuracy"),
                "effective_tokens_per_sec": kpis.get("effective_tokens_per_sec"),
                "end_to_end_tokens_per_sec": kpis.get("end_to_end_tokens_per_sec"),
                "tokens_per_sec": kpis.get("tokens_per_sec"),
                "latency_ms_avg": kpis.get("latency_ms_avg"),
                "ttft_ms_avg": kpis.get("ttft_ms_avg"),
                "primary_model_filename": model.get("primary_model_filename"),
                "draft_model_filename": model.get("draft_model_filename"),
                "replacement_source": model.get("replacement_source"),
            }
        )
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(comparison: dict[str, Any], path: Path) -> None:
    models = comparison["models"]
    succeeded = [model for model in models if not _is_failed(model)]
    failed = [model for model in models if _is_failed(model)]
    replacements = comparison.get("replacements", [])

    def top(predicate, n: int = 10) -> list[dict[str, Any]]:
        rows = [
            model
            for model in succeeded
            if predicate(model) and model.get("kpis", {}).get("effective_tokens_per_sec") is not None
        ]
        return sorted(rows, key=lambda model: model["kpis"]["effective_tokens_per_sec"], reverse=True)[:n]

    lines: list[str] = []
    lines.append(f"# {comparison['title']}")
    lines.append("")
    lines.append(f"- Specs in report: {len(models)}")
    lines.append(f"- Successful specs: {len(succeeded)}")
    lines.append(f"- Failed specs: {len(failed)}")
    lines.append(f"- Replacements from fallback runs: {len(replacements)}")
    if comparison.get("combined_strategy"):
        lines.append(f"- Combined strategy: `{comparison['combined_strategy']}`")
    lines.append("")
    if replacements:
        lines.append("## Fallback replacements")
        for item in replacements:
            lines.append(
                f"- `{item['recipe_id']}`: original `{item['original_target_gpu']}` failed with `{item['original_error']}`; replaced by `{item['replacement_target_gpu']}` from `{item['replacement_suite']}`."
            )
        lines.append("")
    for title, predicate in [
        ("Top overall by effective tok/s", lambda model: True),
        ("Top regular", lambda model: model.get("mode") == "regular"),
        ("Top MTP", lambda model: model.get("mode") == "mtp"),
        ("Top DiffusionGemma", lambda model: str(model.get("mode") or "").startswith("diffusion")),
    ]:
        lines.append(f"## {title}")
        lines.append("| eff tok/s | e2e tok/s | acc | target GPU | source | mode | model | ctx | reasoning | mtp n | recipe |")
        lines.append("|---:|---:|---:|---|---|---|---|---:|---|---:|---|")
        for model in top(predicate):
            kpis = model["kpis"]
            lines.append(
                "| "
                f"{(kpis.get('effective_tokens_per_sec') or 0):.1f} | "
                f"{(kpis.get('end_to_end_tokens_per_sec') or 0):.1f} | "
                f"{(kpis.get('accuracy') or 0):.3f} | "
                f"{model.get('target_gpu') or ''} | "
                f"{model.get('source_label') or ''} | "
                f"{model.get('mode') or ''} | "
                f"{model.get('model_family') or ''} | "
                f"{model.get('context_size') or ''} | "
                f"{model.get('reasoning') or ''} | "
                f"{model.get('mtp_draft_n') or ''} | "
                f"`{model.get('recipe_id')}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_error(model: dict[str, Any]) -> str | None:
    lifecycle = model.get("lifecycle") if isinstance(model.get("lifecycle"), dict) else {}
    if lifecycle.get("error"):
        return str(lifecycle["error"])
    metrics_path = model.get("metrics_path")
    if metrics_path:
        try:
            metrics = _load(Path(metrics_path))
            return metrics.get("error")
        except Exception:
            return None
    return None


def _write_outputs(combined: dict[str, Any], output_suite: Path, base_suite: Path, fallback_suite: Path) -> None:
    combined.setdefault("run_id", output_suite.name)
    combined.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    if output_suite.exists():
        shutil.rmtree(output_suite)
    output_suite.mkdir(parents=True)
    (output_suite / "comparison.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    (output_suite / "comparison.html").write_text(build_comparison_html(combined), encoding="utf-8")
    _write_csv(combined, output_suite / "matrix_summary.csv")
    _write_md(combined, output_suite / "matrix_summary.md")
    (output_suite / "README.md").write_text(
        "# Combined matrix report\n\n"
        f"Base suite: `{base_suite}`\n\n"
        f"Fallback suite: `{fallback_suite}`\n\n"
        f"Strategy: `{combined.get('combined_strategy') or 'replace-failed'}`\n\n"
        "See `comparison.json`, `comparison.html`, `matrix_summary.csv`, and `matrix_summary.md` for the generated artifacts.\n",
        encoding="utf-8",
    )


def _build_union(base_suite: Path, fallback_suite: Path, base: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    combined_models: list[dict[str, Any]] = []
    for source_suite, comparison in [(base_suite, base), (fallback_suite, fallback)]:
        for model in comparison["models"]:
            if _is_failed(model):
                continue
            combined_models.append(_copy_successful_union_model(model, order=len(combined_models) + 1, source_suite=source_suite))

    combined = dict(base)
    combined["title"] = "Gemma Matrix Bakeoff — combined A6000 + A100 successful report"
    combined["source_suites"] = [str(base_suite), str(fallback_suite)]
    combined["source_base_suite"] = str(base_suite)
    combined["source_fallback_suite"] = str(fallback_suite)
    combined["combined_strategy"] = "union"
    combined["replacements"] = []
    combined["models"] = combined_models
    combined["model_count"] = len(combined_models)
    combined["best_by_metric"] = _best_by_metric(combined_models, REPORT_METRICS)
    combined["notes"] = [
        "Union report: includes every successful row from both source suites.",
        "Failed/empty rows are omitted so JSON, HTML, CSV, and Markdown counters reflect only valid benchmark results.",
    ]
    return combined


def _build_replace_failed(base_suite: Path, fallback_suite: Path, base: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    fallback_by_key = {_model_key(model): model for model in fallback["models"] if not _is_failed(model)}

    replacements: list[dict[str, Any]] = []
    combined_models: list[dict[str, Any]] = []
    for model in base["models"]:
        key = _model_key(model)
        replacement = fallback_by_key.get(key)
        if _is_failed(model) and replacement is not None:
            new_model = dict(replacement)
            new_model["order"] = model.get("order")
            new_model["replacement_source"] = str(fallback_suite)
            replacements.append(
                {
                    "recipe_id": model.get("recipe_id"),
                    "key": key,
                    "original_target_gpu": model.get("target_gpu"),
                    "replacement_target_gpu": replacement.get("target_gpu"),
                    "original_error": _extract_error(model),
                    "replacement_suite": str(fallback_suite),
                }
            )
            combined_models.append(new_model)
        else:
            combined_models.append(model)

    combined_models = sorted(combined_models, key=lambda model: int(model.get("order") or 0))
    combined = dict(base)
    combined["title"] = "Gemma Matrix Bakeoff — combined successful report"
    combined["source_base_suite"] = str(base_suite)
    combined["source_fallback_suite"] = str(fallback_suite)
    combined["combined_strategy"] = "replace-failed"
    combined["replacements"] = replacements
    combined["models"] = combined_models
    combined["model_count"] = len(combined_models)
    combined["best_by_metric"] = _best_by_metric(combined_models, REPORT_METRICS)
    combined["notes"] = [
        "Base run remains the source of truth. Failed base rows are replaced only when a matching successful fallback row exists.",
        "Use --strategy union when both source suites are complete and every successful row from both GPUs should appear in the same report.",
    ]
    return combined


def build_combined(base_suite: Path, fallback_suite: Path, output_suite: Path, *, strategy: str = "replace-failed") -> None:
    base = _load(base_suite / "comparison.json")
    fallback = _load(fallback_suite / "comparison.json")

    if strategy == "union":
        combined = _build_union(base_suite, fallback_suite, base, fallback)
    elif strategy == "replace-failed":
        combined = _build_replace_failed(base_suite, fallback_suite, base, fallback)
    else:
        raise ValueError(f"Unknown combined report strategy: {strategy}")

    _write_outputs(combined, output_suite, base_suite, fallback_suite)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a combined Gemma matrix comparison from two source suites.")
    parser.add_argument("--base-suite", type=Path, required=True)
    parser.add_argument("--fallback-suite", type=Path, required=True)
    parser.add_argument("--output-suite", type=Path, required=True)
    parser.add_argument("--strategy", choices=["replace-failed", "union"], default="replace-failed")
    args = parser.parse_args()
    build_combined(args.base_suite, args.fallback_suite, args.output_suite, strategy=args.strategy)
    print(f"Combined suite: {args.output_suite}")
    print(f"Comparison JSON: {args.output_suite / 'comparison.json'}")
    print(f"Comparison HTML: {args.output_suite / 'comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
