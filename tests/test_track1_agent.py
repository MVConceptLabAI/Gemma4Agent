from __future__ import annotations

import json
from pathlib import Path

from aireceipes.track1_agent import main, run_competition_agent, select_model


def _write_tasks(path: Path, tasks: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks), encoding="utf-8")


def test_track1_agent_writes_valid_results_with_zero_token_local_solvers(tmp_path):
    input_path = tmp_path / "input" / "tasks.json"
    output_path = tmp_path / "output" / "results.json"
    _write_tasks(
        input_path,
        [
            {"task_id": "math-1", "prompt": "What is 17 + 25? Answer with only the number."},
            {"task_id": "sent-1", "prompt": "Classify the sentiment as positive, negative, or neutral: I love this fast benchmark."},
        ],
    )

    exit_code = run_competition_agent(input_path=input_path, output_path=output_path)

    assert exit_code == 0
    results = json.loads(output_path.read_text(encoding="utf-8"))
    assert results == [
        {"task_id": "math-1", "answer": "42"},
        {"task_id": "sent-1", "answer": "positive"},
    ]
    assert [set(row) for row in results] == [{"task_id", "answer"}, {"task_id", "answer"}]

    audit = json.loads((output_path.parent / "route_audit.json").read_text(encoding="utf-8"))
    assert audit[0]["route"] == "local"
    assert audit[0]["remote_used"] is False
    assert audit[1]["category"] == "sentiment_classification"


def test_track1_cli_accepts_explicit_input_and_output_paths(tmp_path):
    input_path = tmp_path / "input" / "tasks.json"
    output_path = tmp_path / "output" / "results.json"
    _write_tasks(input_path, [{"task_id": "math-cli", "prompt": "Calculate 9 * 9."}])

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"task_id": "math-cli", "answer": "81"}]


def test_track1_agent_uses_fireworks_fallback_from_environment_for_uncertain_tasks(tmp_path, monkeypatch):
    input_path = tmp_path / "input" / "tasks.json"
    output_path = tmp_path / "output" / "results.json"
    _write_tasks(input_path, [{"task_id": "fact-1", "prompt": "Who wrote Pride and Prejudice?"}])
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://fireworks-proxy.example/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/fireworks/models/tiny,accounts/fireworks/models/gemma-test")
    calls: list[dict[str, object]] = []

    def fake_remote_answerer(prompt: str, category: str, config, max_tokens: int) -> str:
        calls.append(
            {
                "prompt": prompt,
                "category": category,
                "base_url": config.base_url,
                "model": config.model,
                "max_tokens": max_tokens,
            }
        )
        return "Jane Austen"

    exit_code = run_competition_agent(input_path=input_path, output_path=output_path, remote_answerer=fake_remote_answerer)

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"task_id": "fact-1", "answer": "Jane Austen"}]
    assert calls == [
        {
            "prompt": "Who wrote Pride and Prejudice?",
            "category": "factual_knowledge",
            "base_url": "https://fireworks-proxy.example/v1",
            "model": "accounts/fireworks/models/gemma-test",
            "max_tokens": 256,
        }
    ]

    audit = json.loads((output_path.parent / "route_audit.json").read_text(encoding="utf-8"))
    assert audit[0]["route"] == "fireworks"
    assert audit[0]["remote_used"] is True


def test_select_model_uses_only_allowed_models_and_prefers_gemma_when_available():
    assert (
        select_model(
            "factual_knowledge",
            ["accounts/fireworks/models/small", "accounts/fireworks/models/gemma-optimized"],
        )
        == "accounts/fireworks/models/gemma-optimized"
    )
    assert select_model("code_generation", ["allowed-a", "allowed-b"]) == "allowed-a"
