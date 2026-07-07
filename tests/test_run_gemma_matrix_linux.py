import argparse
import importlib.util
import sys
from pathlib import Path


DIFFUSION_FILENAME = "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"


def _load_matrix_module():
    module_path = Path("scripts/run_gemma_matrix.py").resolve()
    module_name = "run_gemma_matrix_linux_tests"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_linux_platform_defaults_use_linux_paths(monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.setattr(matrix.sys, "platform", "linux")

    assert matrix._host_platform() == "linux"
    assert matrix._default_wrapper_command_file().as_posix().endswith("scripts/wrap_each_test.sh")
    assert matrix._default_diffusion_gguf().as_posix() == f"/mnt/d/models/{DIFFUSION_FILENAME}"

    llama_candidates = [candidate.as_posix() for candidate in matrix._default_llama_server_candidates()]
    assert "/usr/local/bin/llama-server" in llama_candidates
    assert "/usr/bin/llama-server" in llama_candidates
    assert all(not candidate.endswith(".exe") for candidate in llama_candidates)


def test_linux_unsloth_start_command_uses_posix_env_prefix(monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.setattr(matrix.sys, "platform", "linux")
    monkeypatch.setenv("LLAMA_SERVER_PATH", "/usr/local/bin/llama-server")
    monkeypatch.setattr(matrix, "_default_dg_visual_bin", lambda: Path("/usr/local/bin/llama-diffusion-gemma-visual-server"))
    monkeypatch.setattr(matrix, "_default_unsloth_dg_shim", lambda: Path("/home/user/AiReceipes/scripts/dg_shim_module_wrapper.py"))
    args = argparse.Namespace(
        unsloth_llama_server_path=None,
        llama_server=None,
        cuda_visible_devices="0",
        dg_visual_bin=None,
        unsloth_dg_shim=None,
        unsloth_port=18084,
        unsloth_server="/home/user/.local/bin/unsloth",
        unsloth_extra_args=None,
    )

    command = matrix._unsloth_start_command(Path("/mnt/d/models/gemma-4-26B-A4B-it.gguf"), 32768, args)

    assert command.startswith(
        "CUDA_VISIBLE_DEVICES=0 LLAMA_SERVER_PATH=/usr/local/bin/llama-server "
        "DG_VISUAL_BIN=/usr/local/bin/llama-diffusion-gemma-visual-server "
        "UNSLOTH_DG_SHIM=/home/user/AiReceipes/scripts/dg_shim_module_wrapper.py "
    )
    assert 'set "' not in command
    assert '"/home/user/.local/bin/unsloth" run --model "/mnt/d/models/gemma-4-26B-A4B-it.gguf"' in command


def test_linux_matrix_command_accepts_linux_paths(tmp_path, monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.setattr(matrix.sys, "platform", "linux")
    args = argparse.Namespace(
        recipes_dir=matrix.DEFAULT_RECIPES_DIR,
        allow_missing=True,
        model=["26b"],
        mode=["mtp"],
        context=[32768],
        reasoning=["off"],
        mtp_draft_n=[2],
        orchestration_order="lifecycle",
        limit=None,
        gemma12_model_gguf=None,
        gemma12_mtp_draft_gguf=None,
        gemma26_model_gguf="/mnt/f/hemes/AiReceipes/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        gemma26_mtp_draft_gguf="/mnt/f/hemes/AiReceipes/models/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it.gguf",
        target_gpu="A100-SXM4-80GB",
        timeout_seconds=1,
        diffusion_timeout_seconds=1,
        llama_server="/usr/local/bin/llama-server",
        diffusion_server="/usr/local/bin/llama-diffusion-gemma-server",
        regular_port=18082,
        mtp_port=18083,
        diffusion_port=18081,
        gpu_device="CUDA0",
        batch_size=8192,
        ubatch_size=2048,
        regular_extra_args=None,
        mtp_extra_args=None,
        diffusion_extra_args=None,
        diffusion_gguf=None,
        diffusion_steps=8,
        diffusion_block_length=256,
        max_tokens=None,
        thinking_max_tokens=None,
    )

    specs = matrix._build_matrix(args, tmp_path / "recipes")

    assert len(specs) == 1
    command = specs[0].command
    assert command is not None
    assert command.startswith('"/usr/local/bin/llama-server" -m "/mnt/f/hemes/AiReceipes/models/')
    assert "--model-draft \"/mnt/f/hemes/AiReceipes/models/" in command
    assert "--spec-type draft-mtp" in command
    assert "F:/" not in command
    assert "D:/" not in command
    assert ".exe" not in command

    recipe_text = Path(specs[0].recipe_path).read_text(encoding="utf-8")
    assert "/usr/local/bin/llama-server" in recipe_text
    assert "/mnt/f/hemes/AiReceipes/models/" in recipe_text
