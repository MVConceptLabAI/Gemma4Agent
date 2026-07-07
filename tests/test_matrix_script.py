import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_matrix_module():
    module_path = Path("scripts/run_gemma_matrix.py").resolve()
    module_name = "run_gemma_matrix_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_script_dry_run_generates_plan_and_recipes(tmp_path):
    output_dir = tmp_path / "runs"
    command = [
        sys.executable,
        "scripts/run_gemma_matrix.py",
        "--dry-run",
        "--allow-missing",
        "--output-dir",
        str(output_dir),
        "--run-id",
        "matrix-test",
        "--model",
        "12b",
        "--mode",
        "regular",
        "--context",
        "32768",
        "--reasoning",
        "off",
        "--gemma12-model-gguf",
        str(tmp_path / "fake-primary.gguf"),
        "--gemma12-mtp-draft-gguf",
        str(tmp_path / "fake-draft.gguf"),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)

    assert "Matrix specs: 1" in completed.stdout
    plan_path = output_dir / "matrix-test-matrix-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["count"] == 1
    assert plan["specs"][0]["context_size"] == 32768
    assert plan["specs"][0]["reasoning"] == "off"
    recipe_path = Path(plan["specs"][0]["recipe_path"])
    assert recipe_path.exists()
    recipe_text = recipe_path.read_text(encoding="utf-8")
    assert "context_size = 32768" in recipe_text
    assert "reasoning = \"off\"" in recipe_text
    assert "--reasoning off" in recipe_text


def test_matrix_script_dry_run_default_72_specs_uses_result_and_wrapper_plan(tmp_path):
    output_dir = tmp_path / "custom-result"
    command = [
        sys.executable,
        "scripts/run_gemma_matrix.py",
        "--dry-run",
        "--allow-missing",
        "--result",
        str(output_dir),
        "--run-id",
        "matrix-72",
        "--mode",
        "regular",
        "--mode",
        "mtp",
        "--mode",
        "diffusion",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)

    assert "Matrix specs: 72" in completed.stdout
    plan_path = output_dir / "matrix-72-matrix-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["count"] == 72
    assert plan["result_root"] == str(output_dir)
    assert plan["orchestration_order"] == "lifecycle"
    assert plan["lifecycle_reuse"] is True
    assert plan["wrapper_command_file"].endswith("wrap_each_test.cmd")
    assert "$result" in plan["result_variable_note"]


def test_matrix_script_dry_run_unsloth_mode_generates_72_backend_specs(tmp_path):
    output_dir = tmp_path / "unsloth-result"
    command = [
        sys.executable,
        "scripts/run_gemma_matrix.py",
        "--dry-run",
        "--allow-missing",
        "--result",
        str(output_dir),
        "--run-id",
        "unsloth-72",
        "--mode",
        "unsloth",
        "--unsloth-base-url",
        "http://127.0.0.1:18084/v1",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)

    assert "Matrix specs: 72" in completed.stdout
    plan = json.loads((output_dir / "unsloth-72-matrix-plan.json").read_text(encoding="utf-8"))
    assert plan["count"] == 72
    assert {spec["backend"] for spec in plan["specs"]} == {"unsloth-studio-or-compatible"}
    assert {spec["mode"] for spec in plan["specs"]} == {"regular", "mtp", "diffusion-unsloth"}
    first_recipe = Path(plan["specs"][0]["recipe_path"]).read_text(encoding="utf-8")
    assert "/api/inference/status" in first_recipe
    assert "ready_json_key = \"active_model\"" in first_recipe
    assert "/v1/models" not in first_recipe


def test_matrix_script_dry_run_mtp_unsloth_mode_generates_only_mtp_backend_specs(tmp_path):
    output_dir = tmp_path / "mtp-unsloth-result"
    command = [
        sys.executable,
        "scripts/run_gemma_matrix.py",
        "--dry-run",
        "--allow-missing",
        "--result",
        str(output_dir),
        "--run-id",
        "mtp-unsloth-26b",
        "--model",
        "26b",
        "--mode",
        "mtp-unsloth",
        "--unsloth-base-url",
        "http://127.0.0.1:18084/v1",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)

    assert "Matrix specs: 24" in completed.stdout
    plan = json.loads((output_dir / "mtp-unsloth-26b-matrix-plan.json").read_text(encoding="utf-8"))
    assert plan["count"] == 24
    assert plan["modes"] == ["mtp-unsloth"]
    assert {spec["backend"] for spec in plan["specs"]} == {"unsloth-studio-or-compatible"}
    assert {spec["mode"] for spec in plan["specs"]} == {"mtp"}
    assert {spec["mtp_draft_n"] for spec in plan["specs"]} == {2, 4, 6}
    first_recipe = Path(plan["specs"][0]["recipe_path"]).read_text(encoding="utf-8")
    assert 'backend = "unsloth-studio-or-compatible"' in first_recipe
    assert 'mode = "mtp"' in first_recipe
    assert 'variant = "mtp-draft-unsloth-endpoint"' in first_recipe
    assert "--spec-type draft-mtp" in first_recipe
    assert "--spec-draft-n-max 2" in first_recipe
    assert "--model-draft" in first_recipe
    assert "mtp-gemma-4-26B-A4B-it.gguf" in first_recipe
    assert "--spec-draft-ngl 99" in first_recipe
    assert 'model_memory_key = "unsloth:26b:mtp-n2:ctx32768"' in first_recipe
    assert "/api/inference/status" in first_recipe
    assert "ready_json_key = \"active_model\"" in first_recipe
    assert "/v1/models" not in first_recipe


def test_unsloth_missing_dependencies_are_reported_without_install(monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.delenv("UNSLOTH_PYTHON", raising=False)
    monkeypatch.setattr(matrix, "_missing_python_modules", lambda python, modules: ["unsloth", "torch"])
    args = argparse.Namespace(mode=["diffusion-unsloth"], install_unsloth_if_required=False)

    report = matrix._ensure_unsloth_if_requested(args)

    assert report["status"] == "missing_not_installed"
    assert report["install_requested"] is False
    assert report["missing_modules"] == ["unsloth", "torch"]


def test_unsloth_install_if_required_runs_pip_and_rechecks(monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.setenv("UNSLOTH_PYTHON", sys.executable)
    missing_responses = iter([["unsloth"], []])
    monkeypatch.setattr(matrix, "_missing_python_modules", lambda python, modules: next(missing_responses))
    captured = {}

    def fake_run(command, text, capture_output, timeout, **kwargs):
        captured["command"] = command
        captured["timeout"] = timeout
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="installed ok", stderr="")

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    args = argparse.Namespace(
        mode=["diffusion-unsloth"],
        install_unsloth_if_required=True,
        unsloth_install_package=["fake-unsloth-package"],
        unsloth_install_timeout_seconds=123,
    )

    report = matrix._ensure_unsloth_if_requested(args)

    assert captured["command"] == [sys.executable, "-m", "pip", "install", "--upgrade", "fake-unsloth-package"]
    assert captured["timeout"] == 123
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
    assert report["status"] == "installed"
    assert report["install_returncode"] == 0
    assert report["missing_modules_after_install"] == []


def test_unsloth_install_report_is_added_to_comparison_preflight_checks():
    matrix = _load_matrix_module()
    comparison = {"models": []}
    report = {
        "status": "missing_not_installed",
        "python": sys.executable,
        "checked_modules": ["unsloth", "torch", "fastapi", "uvicorn"],
        "missing_modules": ["unsloth", "torch"],
        "install_requested": False,
    }

    matrix._add_unsloth_preflight_to_comparison(comparison, report)

    assert comparison["preflight_checks"][0]["name"] == "Unsloth backend dependency test"
    assert comparison["preflight_checks"][0]["status"] == "missing_not_installed"
    assert comparison["preflight_checks"][0]["missing_modules"] == ["unsloth", "torch"]


def test_missing_unsloth_and_unreachable_endpoint_marks_benchmark_specs_skipped(monkeypatch):
    matrix = _load_matrix_module()
    args = argparse.Namespace(
        mode=["unsloth"],
        unsloth_base_url="http://127.0.0.1:9/v1",
        unsloth_diffusion_base_url=None,
        force_unsloth_run_on_missing=False,
        unsloth_endpoint_timeout_seconds=0.01,
    )
    report = {"status": "missing_not_installed", "missing_modules": ["unsloth"]}
    monkeypatch.setattr(matrix, "_unsloth_endpoint_reachable", lambda base_url, timeout_seconds: False)

    matrix._drop_unrunnable_unsloth_specs(args, report)

    assert args.mode == ["unsloth"]
    assert report["benchmark_specs_skipped"] is True
    assert report["endpoint_status"] == "unreachable"


def test_missing_unsloth_but_reachable_endpoint_keeps_benchmark_specs(monkeypatch):
    matrix = _load_matrix_module()
    args = argparse.Namespace(
        mode=["unsloth"],
        unsloth_base_url="http://127.0.0.1:18084/v1",
        unsloth_diffusion_base_url=None,
        force_unsloth_run_on_missing=False,
        unsloth_endpoint_timeout_seconds=0.01,
    )
    report = {"status": "missing_not_installed", "missing_modules": ["unsloth"]}
    monkeypatch.setattr(matrix, "_unsloth_endpoint_reachable", lambda base_url, timeout_seconds: True)

    matrix._drop_unrunnable_unsloth_specs(args, report)

    assert args.mode == ["unsloth"]
    assert report["endpoint_status"] == "reachable"
    assert "benchmark_specs_skipped" not in report


def test_diffusion_unsloth_recipe_marks_missing_visual_server_as_preflight_skip(tmp_path, monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.setattr(matrix, "_default_dg_visual_bin", lambda: None)
    monkeypatch.setattr(matrix, "_default_unsloth_dg_shim", lambda: Path("C:/fake/shim.py"))
    args = argparse.Namespace(
        recipes_dir=matrix.DEFAULT_RECIPES_DIR,
        allow_missing=True,
        model=["12b"],
        mode=["diffusion-unsloth"],
        context=[32768],
        reasoning=["off"],
        mtp_draft_n=matrix.MTP_DRAFT_N,
        gemma12_model_gguf=None,
        gemma12_mtp_draft_gguf=None,
        gemma26_model_gguf=None,
        gemma26_mtp_draft_gguf=None,
        target_gpu="A100",
        timeout_seconds=1,
        diffusion_timeout_seconds=1,
        unsloth_base_url="http://127.0.0.1:18084/v1",
        unsloth_diffusion_base_url=None,
        unsloth_model_prefix=None,
        unsloth_diffusion_model=None,
        unsloth_api_key="sk-unsloth-test",
        unsloth_port=18084,
        unsloth_server="unsloth",
        unsloth_llama_server_path=None,
        cuda_visible_devices="2",
        dg_visual_bin=None,
        unsloth_dg_shim=None,
        unsloth_extra_args=None,
        max_tokens=None,
        thinking_max_tokens=None,
        orchestration_order="lifecycle",
        limit=None,
    )

    specs = matrix._build_matrix(args, tmp_path / "recipes")

    assert len(specs) == 1
    recipe_text = Path(specs[0].recipe_path).read_text(encoding="utf-8")
    assert 'skip_preflight_reason = "missing_diffusiongemma_visual_server"' in recipe_text
    assert "DiffusionGemma visual-server binary was not found" in recipe_text


def test_unsloth_start_command_exports_diffusion_shim_and_visual_bin(monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.setattr(matrix, "_default_dg_visual_bin", lambda: Path("C:/dg/visual.exe"))
    monkeypatch.setattr(matrix, "_default_unsloth_dg_shim", lambda: Path("C:/dg/shim.py"))
    args = argparse.Namespace(
        unsloth_llama_server_path=None,
        llama_server=None,
        cuda_visible_devices="2",
        dg_visual_bin=None,
        unsloth_dg_shim=None,
        unsloth_port=18084,
        unsloth_server="C:/unsloth.exe",
        unsloth_extra_args=None,
    )

    command = matrix._unsloth_start_command(Path("D:/models/diffusion.gguf"), 32768, args)

    assert "DG_VISUAL_BIN=C:/dg/visual.exe" in command
    assert "UNSLOTH_DG_SHIM=C:/dg/shim.py" in command


def test_unsloth_start_command_auto_exports_default_llama_server(monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setattr(matrix, "DEFAULT_LLAMA_SERVER_CANDIDATES", [Path("F:/llama-build-cuda/bin/llama-server.exe")], raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: str(self).replace("\\", "/") == "F:/llama-build-cuda/bin/llama-server.exe")
    args = argparse.Namespace(
        unsloth_llama_server_path=None,
        llama_server=None,
        cuda_visible_devices="1",
        dg_visual_bin=None,
        unsloth_dg_shim=None,
        unsloth_port=18084,
        unsloth_server="C:/unsloth.exe",
        unsloth_extra_args=None,
    )

    command = matrix._unsloth_start_command(Path("F:/models/model.gguf"), 32768, args)

    assert "LLAMA_SERVER_PATH=F:/llama-build-cuda/bin/llama-server.exe" in command


def test_skipped_unsloth_matrix_writes_72_comparison_rows(tmp_path):
    matrix = _load_matrix_module()
    args = argparse.Namespace(
        recipes_dir=matrix.DEFAULT_RECIPES_DIR,
        allow_missing=True,
        model=["12b", "26b"],
        mode=["unsloth"],
        context=matrix.CONTEXTS,
        reasoning=matrix.REASONING_MODES,
        mtp_draft_n=matrix.MTP_DRAFT_N,
        orchestration_order="lifecycle",
        limit=None,
        gemma12_model_gguf=None,
        gemma12_mtp_draft_gguf=None,
        gemma26_model_gguf=None,
        gemma26_mtp_draft_gguf=None,
        target_gpu="GPU",
        timeout_seconds=1,
        diffusion_timeout_seconds=1,
        unsloth_base_url="http://127.0.0.1:9/v1",
        unsloth_diffusion_base_url=None,
        unsloth_model_prefix=None,
        unsloth_diffusion_model=None,
        max_tokens=None,
        thinking_max_tokens=None,
    )
    recipes_dir = tmp_path / "unsloth-recipes"
    specs = matrix._build_matrix(args, recipes_dir)
    assert len(specs) == 72

    suite_dir, comparison_json_path, comparison_html_path = matrix._run_matrix_suite(
        specs,
        recipes_dir,
        tmp_path / "runs",
        "unsloth-skip",
        lifecycle=False,
        unsloth_install_report={
            "status": "missing_not_installed",
            "missing_modules": ["unsloth", "torch"],
            "checked_modules": ["unsloth", "torch", "fastapi", "uvicorn"],
            "benchmark_specs_skipped": True,
            "benchmark_specs_skip_reason": "missing_unsloth_dependencies_and_unreachable_endpoint",
        },
    )

    comparison = json.loads(comparison_json_path.read_text(encoding="utf-8"))
    assert suite_dir.exists()
    assert comparison["model_count"] == 72
    assert len(comparison["models"]) == 72
    assert {model["backend"] for model in comparison["models"]} == {"unsloth-studio-or-compatible"}
    assert {model["lifecycle"]["status"] for model in comparison["models"]} == {"skipped_preflight"}
    assert "Backend preflight checks" in comparison_html_path.read_text(encoding="utf-8")
