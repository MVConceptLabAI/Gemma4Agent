from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_combined_matrix_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_combined_matrix_report", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison(gpu: str, *, start_order: int = 1) -> dict:
    models = []
    for offset, (ctx, reasoning) in enumerate([(32768, "off"), (32768, "on")], start_order):
        models.append(
            {
                "order": offset,
                "recipe_id": f"inference.matrix-diffusiongemma-unsloth-{reasoning}-ctx{ctx // 1024}k",
                "name": f"DiffusionGemma {gpu} {reasoning} {ctx}",
                "mode": "diffusion-unsloth",
                "model_family": "diffusiongemma-26b-a4b",
                "backend": "unsloth-studio-or-compatible",
                "context_size": ctx,
                "reasoning": reasoning,
                "mtp_draft_n": None,
                "target_gpu": gpu,
                "success_count": 12,
                "error_count": 0,
                "prompt_count": 12,
                "primary_model_filename": "diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
                "draft_model_filename": None,
                "kpis": {
                    "success_rate": 1.0,
                    "accuracy": 0.75,
                    "effective_tokens_per_sec": 100.0 + offset,
                    "end_to_end_tokens_per_sec": 120.0 + offset,
                    "tokens_per_sec": 1000.0 + offset,
                    "latency_ms_avg": 1500.0,
                    "ttft_ms_avg": 1490.0,
                },
                "lifecycle": {"target_gpu": gpu},
            }
        )
    return {
        "title": f"{gpu} comparison",
        "metrics": ["success_rate", "accuracy", "effective_tokens_per_sec"],
        "metric_explanations": {},
        "model_count": len(models),
        "models": models,
        "best_by_metric": {},
    }


def test_build_combined_union_keeps_successful_rows_from_both_suites(tmp_path):
    module = _load_module()
    base_suite = tmp_path / "a6000"
    fallback_suite = tmp_path / "a100"
    output_suite = tmp_path / "combined"
    base_suite.mkdir()
    fallback_suite.mkdir()
    (base_suite / "comparison.json").write_text(json.dumps(_comparison("RTX A6000")), encoding="utf-8")
    (fallback_suite / "comparison.json").write_text(json.dumps(_comparison("A100-SXM4-80GB")), encoding="utf-8")

    module.build_combined(base_suite, fallback_suite, output_suite, strategy="union")

    combined = json.loads((output_suite / "comparison.json").read_text(encoding="utf-8"))
    assert combined["model_count"] == 4
    assert len(combined["models"]) == 4
    assert {model["target_gpu"] for model in combined["models"]} == {"RTX A6000", "A100-SXM4-80GB"}
    assert all(model["success_count"] == 12 for model in combined["models"])
    assert all(model["error_count"] == 0 for model in combined["models"])
    assert combined["combined_strategy"] == "union"
    assert combined["source_suites"] == [str(base_suite), str(fallback_suite)]
    assert (output_suite / "comparison.html").exists()
    assert (output_suite / "matrix_summary.csv").exists()
    summary = (output_suite / "matrix_summary.md").read_text(encoding="utf-8")
    assert "Top DiffusionGemma" in summary
    assert "diffusion-unsloth" in summary
