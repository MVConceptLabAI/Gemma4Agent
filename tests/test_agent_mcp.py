from __future__ import annotations

import json
from pathlib import Path

from aireceipes.hardware import detect_hardware, launch_hardware_benchmark
from aireceipes.mcp_server import get_tool_manifest, handle_tool_call


def test_detect_hardware_returns_agent_safe_snapshot(monkeypatch):
    monkeypatch.setattr("aireceipes.hardware._run_probe", lambda command, timeout_seconds=5.0: {"returncode": 1, "stdout": "", "stderr": "missing"})
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    monkeypatch.setattr("platform.processor", lambda: "x86_64")

    snapshot = detect_hardware()

    assert snapshot["platform"] == "Windows"
    assert snapshot["machine"] == "AMD64"
    assert snapshot["cpu"] == "x86_64"
    assert snapshot["gpus"] == []
    assert snapshot["probes"]["nvidia-smi"]["ok"] is False


def test_launch_hardware_benchmark_dry_run_builds_matrix_command_and_route_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "aireceipes.hardware.detect_hardware",
        lambda: {"platform": "Windows", "machine": "AMD64", "cpu": "x86_64", "gpus": [{"name": "NVIDIA A100", "memory_total_mb": 81920}]},
    )

    result = launch_hardware_benchmark(
        hardware="auto",
        model="gemma-4-12b",
        workflow="agentic",
        output_dir=tmp_path / "runs",
        run_id="agent-auto",
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert result["hardware"]["gpus"][0]["name"] == "NVIDIA A100"
    assert result["benchmark"]["command"][0].endswith("python") or "python" in result["benchmark"]["command"][0]
    assert "scripts/run_gemma_matrix.py" in result["benchmark"]["command"]
    assert "--dry-run" in result["benchmark"]["command"]
    assert result["route_hint"]["workflow"] == "agentic"
    assert result["route_hint"]["optimize_for"] == "balanced"


def test_mcp_manifest_exposes_benchmark_and_route_tools():
    manifest = get_tool_manifest()

    tool_names = {tool["name"] for tool in manifest["tools"]}
    assert {"aireceipes.detect_hardware", "aireceipes.launch_benchmark", "aireceipes.plan_route"}.issubset(tool_names)
    launch_tool = next(tool for tool in manifest["tools"] if tool["name"] == "aireceipes.launch_benchmark")
    assert launch_tool["input_schema"]["properties"]["hardware"]["default"] == "auto"
    assert "dry_run" in launch_tool["input_schema"]["properties"]


def test_mcp_handle_tool_call_launches_dry_run_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "aireceipes.hardware.detect_hardware",
        lambda: {"platform": "Linux", "machine": "x86_64", "cpu": "EPYC", "gpus": [{"name": "RTX A6000", "memory_total_mb": 49140}]},
    )

    response = handle_tool_call(
        "aireceipes.launch_benchmark",
        {
            "hardware": "auto",
            "model": "gemma-4-12b",
            "workflow": "agentic",
            "output_dir": str(tmp_path / "runs"),
            "run_id": "mcp-dry-run",
            "dry_run": True,
        },
    )

    assert response["ok"] is True
    assert response["content"]["status"] == "planned"
    assert response["content"]["hardware"]["gpus"][0]["name"] == "RTX A6000"


def test_mcp_plan_route_returns_structured_policy(tmp_path):
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "recipe_id": "inference.local-fast",
                        "name": "Local fast",
                        "backend": "local",
                        "success_count": 4,
                        "prompt_count": 4,
                        "kpis": {"accuracy": 1.0, "effective_tokens_per_sec": 100.0, "latency_ms_avg": 200.0},
                        "verticals": {"agentic": {"accuracy": 1.0, "effective_tokens_per_sec": 100.0, "latency_ms_avg": 200.0}},
                        "routing": {"cost_per_1m_output_tokens_usd": 0.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    response = handle_tool_call(
        "aireceipes.plan_route",
        {
            "comparison": [str(comparison_path)],
            "workflow": "agentic",
            "strategy": "fastest",
            "min_accuracy": 0.8,
        },
    )

    assert response["ok"] is True
    assert response["content"]["selected"]["recipe_id"] == "inference.local-fast"
    assert response["content"]["route_policy"][0]["route_to"]["recipe_id"] == "inference.local-fast"
