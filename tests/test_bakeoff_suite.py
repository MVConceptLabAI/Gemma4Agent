import json
import sys
from pathlib import Path

from aireceipes.catalog import Recipe
from aireceipes.cli import main
from aireceipes.suite import (
    build_comparison_html,
    load_test_wrapper_command,
    run_bakeoff_suite,
    _run_shell_command,
    _start_lifecycle,
)


def _marker_command(log_path: Path, marker: str) -> str:
    log = log_path.as_posix()
    return f'"{sys.executable}" -c "from pathlib import Path; Path(r\'{log}\').open(\'a\', encoding=\'utf-8\').write({marker!r})"'


def _wrapper_command(log_path: Path) -> str:
    code = (
        "import os; "
        "from pathlib import Path; "
        f"Path(r'{log_path.as_posix()}').open('a', encoding='utf-8').write('wrapper:' + os.environ['recipe_id'] + chr(10)); "
        "Path(os.environ['result'], 'wrapper-env.txt').open('a', encoding='utf-8').write(os.environ['recipe_id'] + '|' + os.environ['test_result'] + chr(10))"
    )
    return f'"{sys.executable}" -c {json.dumps(code)}'


def _write_echo_recipe(recipes_dir: Path, recipe_id: str, name: str, log_path: Path) -> None:
    recipe_dir = recipes_dir / "inference"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    start_command = _marker_command(log_path, f"start:{recipe_id}\n")
    stop_command = _marker_command(log_path, f"stop:{recipe_id}\n")
    (recipe_dir / f"{recipe_id.split('.')[-1]}.toml").write_text(
        f'''
id = "{recipe_id}"
category = "inference"
name = "{name}"
description = "Fake model recipe for sequential suite testing."
tags = ["fake", "suite"]

[classification]
task = "suite-test"
variant = "{recipe_id}"

[runtime]
adapter = "echo"
target_gpu = "RTX A6000"
start_command = {json.dumps(start_command)}
stop_command = {json.dumps(stop_command)}
startup_grace_seconds = 15
unload_wait_seconds = 15

[parameters]
repetitions = 1

[[dataset.cases]]
id = "case-one"
vertical = "chat"
prompt = "hello"
expected_contains = "Echo: hello"

[kpis]
metrics = ["success_rate", "accuracy", "effective_tokens_per_sec"]
'''.strip(),
        encoding="utf-8",
    )


def test_bakeoff_suite_loads_and_unloads_each_recipe_then_writes_json_and_html(tmp_path):
    recipes_dir = tmp_path / "recipes"
    log_path = tmp_path / "lifecycle.log"
    recipe_ids = [
        "inference.fake-regular",
        "inference.fake-mtp",
        "inference.fake-diffusion",
    ]
    for recipe_id in recipe_ids:
        _write_echo_recipe(recipes_dir, recipe_id, recipe_id.rsplit(".", 1)[-1], log_path)

    result = run_bakeoff_suite(
        recipe_ids,
        recipes_dir=recipes_dir,
        output_dir=tmp_path / "runs",
        run_id="suite-test",
        lifecycle=True,
        reuse_lifecycle=False,
    )

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "start:inference.fake-regular",
        "stop:inference.fake-regular",
        "start:inference.fake-mtp",
        "stop:inference.fake-mtp",
        "start:inference.fake-diffusion",
        "stop:inference.fake-diffusion",
    ]
    assert len(result.recipe_runs) == 3
    for recipe_run in result.recipe_runs:
        assert recipe_run.metrics_path.exists()
        metrics = json.loads(recipe_run.metrics_path.read_text(encoding="utf-8"))
        assert metrics["lifecycle"]["target_gpu"] == "RTX A6000"
        assert metrics["kpis"]["success_rate"] == 1.0

    comparison = json.loads(result.comparison_json_path.read_text(encoding="utf-8"))
    assert comparison["run_id"] == "suite-test"
    assert [item["recipe_id"] for item in comparison["models"]] == recipe_ids
    assert comparison["best_by_metric"]["accuracy"]["recipe_id"] == "inference.fake-regular"
    assert result.comparison_html_path.exists()
    html = result.comparison_html_path.read_text(encoding="utf-8")
    assert "Gemma Variant Bakeoff" in html
    assert "inference.fake-regular" in html
    assert "effective_tokens_per_sec" in html
    assert "Model family" in html
    assert "Primary GGUF" in html
    assert "Response model" in html
    assert "Recipe / exact model" not in html


def test_comparison_html_includes_unsloth_preflight_check():
    comparison = {
        "title": "Gemma Variant Bakeoff",
        "run_id": "preflight-test",
        "generated_at": "2026-07-05T00:00:00+00:00",
        "model_count": 0,
        "metrics": [],
        "metric_explanations": {},
        "prompt_evaluation_note": "",
        "models": [],
        "best_by_metric": {},
        "preflight_checks": [
            {
                "name": "Unsloth backend dependency test",
                "backend": "unsloth-studio-or-compatible",
                "status": "missing_not_installed",
                "checked_modules": ["unsloth", "torch", "fastapi", "uvicorn"],
                "missing_modules": ["unsloth", "torch"],
                "install_requested": False,
                "message": "Pass --install-unsloth-if-required to install missing dependencies.",
            }
        ],
    }

    html = build_comparison_html(comparison)

    assert "Backend preflight checks" in html
    assert "Unsloth backend dependency test" in html
    assert "missing_not_installed" in html
    assert "unsloth, torch" in html


def test_lifecycle_readiness_forwards_json_key_and_bearer_header(tmp_path, monkeypatch):
    captured = {}

    def fake_wait_for_ready(url, timeout_seconds, poll_seconds, process=None, headers=None, json_ready_key=None):
        captured.update(
            {
                "url": url,
                "timeout_seconds": timeout_seconds,
                "poll_seconds": poll_seconds,
                "process_running": process is not None and process.poll() is None,
                "headers": headers,
                "json_ready_key": json_ready_key,
            }
        )
        return {"ok": True, "url": url, "json_ready_key": json_ready_key, "json_ready_value": "loaded-model"}

    monkeypatch.setattr("aireceipes.suite._wait_for_ready", fake_wait_for_ready)
    recipe = Recipe(
        id="inference.ready-json",
        category="inference",
        name="ready-json",
        description="ready json test",
        tags=[],
        classification={},
        runtime={
            "start_command": f'"{sys.executable}" -c "import time; time.sleep(30)"',
            "ready_url": "http://127.0.0.1:18084/api/inference/status",
            "ready_api_key": "sk-unsloth-test",
            "ready_json_key": "active_model",
            "ready_timeout_seconds": 12,
            "ready_poll_seconds": 0.5,
            "startup_grace_seconds": 0.01,
        },
        parameters={},
        dataset={},
        kpis={},
        path=tmp_path / "recipe.toml",
    )

    process, lifecycle = _start_lifecycle(recipe, tmp_path)
    try:
        assert captured["url"].endswith("/api/inference/status")
        assert captured["headers"] == {"Authorization": "Bearer sk-unsloth-test"}
        assert captured["json_ready_key"] == "active_model"
        assert captured["process_running"] is True
        assert lifecycle["readiness"]["json_ready_value"] == "loaded-model"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def test_load_test_wrapper_command_reads_non_comment_cmd_lines(tmp_path):
    wrapper_file = tmp_path / "wrap_each_test.cmd"
    wrapper_file.write_text(
        "\n".join(
            [
                "@echo off",
                "REM comment",
                ":: comment",
                "# comment",
                "bash -lc 'echo \"$result:$recipe_id\"'",
            ]
        ),
        encoding="utf-8",
    )

    assert load_test_wrapper_command(wrapper_file) == "bash -lc 'echo \"$result:$recipe_id\"'"


def test_run_shell_command_replaces_non_utf8_output(tmp_path):
    code = "import sys; sys.stdout.buffer.write(bytes([0x82, 0x6f, 0x6b])); sys.stderr.buffer.write(bytes([0x82]))"
    command = f'"{sys.executable}" -c {json.dumps(code)}'

    result = _run_shell_command(command, cwd=tmp_path, timeout_seconds=10)

    assert result["returncode"] == 0
    assert "ok" in result["stdout"]
    assert "\ufffd" in result["stdout"]
    assert "\ufffd" in result["stderr"]


def test_bakeoff_wrapper_runs_before_each_recipe_with_result_env(tmp_path):
    recipes_dir = tmp_path / "recipes"
    log_path = tmp_path / "lifecycle.log"
    recipe_ids = ["inference.fake-regular", "inference.fake-mtp"]
    for recipe_id in recipe_ids:
        _write_echo_recipe(recipes_dir, recipe_id, recipe_id.rsplit(".", 1)[-1], log_path)

    result = run_bakeoff_suite(
        recipe_ids,
        recipes_dir=recipes_dir,
        output_dir=tmp_path / "runs",
        run_id="wrapper-suite",
        lifecycle=True,
        test_wrapper_command=_wrapper_command(log_path),
    )

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "wrapper:inference.fake-regular",
        "start:inference.fake-regular",
        "stop:inference.fake-regular",
        "wrapper:inference.fake-mtp",
        "start:inference.fake-mtp",
        "stop:inference.fake-mtp",
    ]
    wrapper_env_lines = (result.suite_dir / "wrapper-env.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split("|", 1)[0] for line in wrapper_env_lines] == recipe_ids
    assert all(str(result.suite_dir) in line for line in wrapper_env_lines)
    for recipe_run in result.recipe_runs:
        assert recipe_run.lifecycle["pre_test_wrapper"]["status"] == "ok"
        assert recipe_run.lifecycle["pre_test_wrapper"]["result"] == str(result.suite_dir)


def test_cli_bakeoff_writes_comparison_artifacts(tmp_path, capsys):
    recipes_dir = tmp_path / "recipes"
    log_path = tmp_path / "lifecycle.log"
    _write_echo_recipe(recipes_dir, "inference.fake-regular", "regular", log_path)
    _write_echo_recipe(recipes_dir, "inference.fake-mtp", "mtp", log_path)

    exit_code = main(
        [
            "bakeoff",
            "--recipes-dir",
            str(recipes_dir),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "cli-suite",
            "--recipe",
            "inference.fake-regular",
            "--recipe",
            "inference.fake-mtp",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "comparison.json" in out
    assert "comparison.html" in out
    assert next((tmp_path / "runs").glob("cli-suite-*/comparison.json")).exists()
    assert next((tmp_path / "runs").glob("cli-suite-*/comparison.html")).exists()
