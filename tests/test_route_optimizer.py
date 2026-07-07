from __future__ import annotations

import json
from pathlib import Path

from aireceipes.cli import main
from aireceipes.router import build_route_plan, write_route_plan


def _comparison(path: Path) -> Path:
    payload = {
        "title": "Synthetic agentic route comparison",
        "run_id": "route-fixture",
        "models": [
            {
                "recipe_id": "inference.fast-expensive",
                "name": "Fast expensive model",
                "backend": "local-llama",
                "model_family": "gemma-4",
                "mode": "regular",
                "target_gpu": "RTX A6000",
                "success_count": 12,
                "prompt_count": 12,
                "kpis": {
                    "accuracy": 0.90,
                    "case_success_rate": 0.90,
                    "effective_tokens_per_sec": 240.0,
                    "latency_ms_avg": 620.0,
                },
                "verticals": {
                    "agentic": {
                        "accuracy": 0.90,
                        "case_success_rate": 0.90,
                        "effective_tokens_per_sec": 240.0,
                        "latency_ms_avg": 620.0,
                    }
                },
                "routing": {"cost_per_1m_output_tokens_usd": 3.0},
            },
            {
                "recipe_id": "inference.cheap-good",
                "name": "Cheap good model",
                "backend": "diffusiongemma",
                "model_family": "diffusiongemma",
                "mode": "diffusion",
                "target_gpu": "A100",
                "success_count": 12,
                "prompt_count": 12,
                "kpis": {
                    "accuracy": 0.92,
                    "case_success_rate": 0.92,
                    "effective_tokens_per_sec": 130.0,
                    "latency_ms_avg": 900.0,
                },
                "verticals": {
                    "agentic": {
                        "accuracy": 0.92,
                        "case_success_rate": 0.92,
                        "effective_tokens_per_sec": 130.0,
                        "latency_ms_avg": 900.0,
                    }
                },
                "routing": {"cost_per_1m_output_tokens_usd": 0.20},
            },
            {
                "recipe_id": "inference.fast-bad",
                "name": "Fast but below quality floor",
                "backend": "local-llama",
                "model_family": "tiny",
                "mode": "regular",
                "success_count": 12,
                "prompt_count": 12,
                "kpis": {
                    "accuracy": 0.50,
                    "case_success_rate": 0.50,
                    "effective_tokens_per_sec": 600.0,
                    "latency_ms_avg": 200.0,
                },
                "verticals": {
                    "agentic": {
                        "accuracy": 0.50,
                        "case_success_rate": 0.50,
                        "effective_tokens_per_sec": 600.0,
                        "latency_ms_avg": 200.0,
                    }
                },
                "routing": {"cost_per_1m_output_tokens_usd": 0.01},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_route_optimizer_selects_balanced_speed_cost_candidate_for_workflow(tmp_path):
    comparison_path = _comparison(tmp_path / "comparison.json")

    plan = build_route_plan(
        [comparison_path],
        workflow="agentic",
        strategy="balanced",
        min_accuracy=0.80,
    )

    assert plan["workflow"] == "agentic"
    assert plan["strategy"] == "balanced"
    assert plan["selected"]["recipe_id"] == "inference.cheap-good"
    assert plan["selected"]["quality"] == 0.92
    assert plan["selected"]["cost_per_1m_output_tokens_usd"] == 0.20
    assert plan["candidates"][0]["recipe_id"] == "inference.cheap-good"
    assert {item["recipe_id"] for item in plan["rejected"]} == {"inference.fast-bad"}
    assert plan["route_policy"][0]["match"] == {"vertical": "agentic"}
    assert plan["route_policy"][0]["route_to"]["recipe_id"] == "inference.cheap-good"


def test_route_optimizer_can_choose_fastest_candidate_for_workflow(tmp_path):
    comparison_path = _comparison(tmp_path / "comparison.json")

    plan = build_route_plan(
        [comparison_path],
        workflow="agentic",
        strategy="fastest",
        min_accuracy=0.80,
    )

    assert plan["selected"]["recipe_id"] == "inference.fast-expensive"
    assert plan["selected"]["effective_tokens_per_sec"] == 240.0


def test_write_route_plan_and_cli_route_write_agent_policy(tmp_path, capsys):
    comparison_path = _comparison(tmp_path / "comparison.json")
    output_path = tmp_path / "route-plan.json"

    written = write_route_plan(
        [comparison_path],
        output_path,
        workflow="agentic",
        strategy="balanced",
        min_accuracy=0.80,
    )

    assert written == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["selected"]["recipe_id"] == "inference.cheap-good"

    cli_output = tmp_path / "cli-route-plan.json"
    exit_code = main(
        [
            "route",
            "--comparison",
            str(comparison_path),
            "--workflow",
            "agentic",
            "--strategy",
            "balanced",
            "--min-accuracy",
            "0.80",
            "--output",
            str(cli_output),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Route plan:" in out
    assert "inference.cheap-good" in out
    assert json.loads(cli_output.read_text(encoding="utf-8"))["selected"]["recipe_id"] == "inference.cheap-good"
