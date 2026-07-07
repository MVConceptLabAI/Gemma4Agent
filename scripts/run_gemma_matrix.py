from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aireceipes.catalog import Catalog, DEFAULT_RECIPES_DIR  # noqa: E402
from aireceipes.runner import RunResult, run_recipe  # noqa: E402
from aireceipes.suite import (  # noqa: E402
    RecipeRunReport,
    build_comparison_html,
    build_comparison_payload,
    load_test_wrapper_command,
    _lifecycle_key,
    _run_test_wrapper,
    _should_keep_lifecycle,
    _start_lifecycle,
    _stop_lifecycle,
    _write_metrics_with_lifecycle,
)

CONTEXTS = [32768, 65536, 131072, 262144]
REASONING_MODES = ["off", "on"]
MTP_DRAFT_N = [2, 4, 6]
TEXT_BASE_RECIPE = "inference.gemma4-regular-bakeoff"
DIFFUSION_BASE_RECIPE = "inference.diffusiongemma-bakeoff"

DEFAULT_12B_MODEL = ROOT / "models/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
DEFAULT_12B_DRAFT = ROOT / "models/gemma-4-12B-it-qat-GGUF/mtp-gemma-4-12B-it.gguf"
DEFAULT_26B_MODEL = ROOT / "models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
DEFAULT_26B_DRAFT = ROOT / "models/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it.gguf"
DEFAULT_DIFFUSION = Path("D:/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
DEFAULT_LINUX_DIFFUSION = Path("/mnt/d/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
DEFAULT_WRAPPER_COMMAND_FILE = ROOT / "scripts" / "wrap_each_test.cmd"
DEFAULT_LINUX_WRAPPER_COMMAND_FILE = ROOT / "scripts" / "wrap_each_test.sh"
DEFAULT_UNSLOTH_STUDIO_ROOT = Path.home() / ".unsloth" / "studio"
DEFAULT_UNSLOTH_STUDIO_SCRIPTS = DEFAULT_UNSLOTH_STUDIO_ROOT / "unsloth_studio" / "Scripts"
DEFAULT_UNSLOTH_SITE_PACKAGES = DEFAULT_UNSLOTH_STUDIO_ROOT / "unsloth_studio" / "Lib" / "site-packages"
DEFAULT_UNSLOTH_SERVER = DEFAULT_UNSLOTH_STUDIO_SCRIPTS / "unsloth.exe"
DEFAULT_UNSLOTH_PYTHON = DEFAULT_UNSLOTH_STUDIO_SCRIPTS / "python.exe"
DEFAULT_UNSLOTH_DG_SHIM = ROOT / "scripts" / "dg_shim_module_wrapper.py"
INSTALLED_UNSLOTH_DG_SHIM = DEFAULT_UNSLOTH_SITE_PACKAGES / "unsloth_zoo" / "diffusion_studio" / "shim.py"
DEFAULT_LLAMA_SERVER_CANDIDATES = [
    Path("F:/llama-build-cuda/bin/llama-server.exe"),
    Path("F:/hemes/llama-build-cuda/bin/llama-server.exe"),
]
DEFAULT_LINUX_LLAMA_SERVER_CANDIDATES = [
    Path("/usr/local/bin/llama-server"),
    Path("/usr/bin/llama-server"),
    Path.home() / ".local" / "bin" / "llama-server",
]
DEFAULT_DG_VISUAL_BIN_CANDIDATES = [
    DEFAULT_UNSLOTH_STUDIO_ROOT / "bin" / "llama-diffusion-gemma-visual-server.exe",
    Path("F:/hemes/llama-dg-visual-build/bin/llama-diffusion-gemma-visual-server.exe"),
    Path("F:/hemes/llama-dg-visual-build-fast/bin/llama-diffusion-gemma-visual-server.exe"),
    Path("F:/llama-build-cuda/bin/llama-diffusion-gemma-visual-server.exe"),
]
DEFAULT_LINUX_DG_VISUAL_BIN_CANDIDATES = [
    DEFAULT_UNSLOTH_STUDIO_ROOT / "bin" / "llama-diffusion-gemma-visual-server",
    Path("/usr/local/bin/llama-diffusion-gemma-visual-server"),
    Path("/usr/bin/llama-diffusion-gemma-visual-server"),
    Path.home() / ".local" / "bin" / "llama-diffusion-gemma-visual-server",
]
DEFAULT_UNSLOTH_INSTALL_PACKAGES = [
    "unsloth",
    "torch",
    "uvicorn",
    "starlette",
    "fastapi",
    "structlog",
    "matplotlib",
    "pillow",
    "scipy",
    "pyjwt",
    "cryptography",
    "diceware",
    "asyncpg",
    "aiosqlite",
    "sqlalchemy",
    "alembic",
]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    family: str
    primary_gguf: Path
    draft_gguf: Path | None = None


@dataclass(frozen=True)
class MatrixSpec:
    recipe_id: str
    name: str
    mode: str
    model_key: str
    context_size: int
    reasoning: str
    mtp_draft_n: int | None
    backend: str
    recipe_path: Path
    command: str | None


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def _host_platform() -> str:
    return "windows" if sys.platform == "win32" else "linux" if sys.platform.startswith("linux") else sys.platform


def _is_windows_host() -> bool:
    return _host_platform() == "windows"


def _default_diffusion_gguf() -> Path:
    return DEFAULT_DIFFUSION if _is_windows_host() else DEFAULT_LINUX_DIFFUSION


def _default_wrapper_command_file() -> Path:
    return DEFAULT_WRAPPER_COMMAND_FILE if _is_windows_host() else DEFAULT_LINUX_WRAPPER_COMMAND_FILE


def _default_llama_server_candidates() -> list[Path]:
    return DEFAULT_LLAMA_SERVER_CANDIDATES if _is_windows_host() else DEFAULT_LINUX_LLAMA_SERVER_CANDIDATES


def _default_dg_visual_bin_candidates() -> list[Path]:
    return DEFAULT_DG_VISUAL_BIN_CANDIDATES if _is_windows_host() else DEFAULT_LINUX_DG_VISUAL_BIN_CANDIDATES


def _default_unsloth_scripts_dir() -> Path:
    return DEFAULT_UNSLOTH_STUDIO_ROOT / "unsloth_studio" / ("Scripts" if _is_windows_host() else "bin")


def _default_unsloth_site_packages_dir() -> Path:
    if _is_windows_host():
        return DEFAULT_UNSLOTH_STUDIO_ROOT / "unsloth_studio" / "Lib" / "site-packages"
    return DEFAULT_UNSLOTH_STUDIO_ROOT / "unsloth_studio" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _default_unsloth_server_path() -> Path:
    return _default_unsloth_scripts_dir() / ("unsloth.exe" if _is_windows_host() else "unsloth")


def _default_unsloth_python_path() -> Path:
    return _default_unsloth_scripts_dir() / ("python.exe" if _is_windows_host() else "python")


def _default_unsloth_python() -> str:
    configured = os.environ.get("UNSLOTH_PYTHON")
    if configured:
        return configured
    default_python = _default_unsloth_python_path()
    if default_python.exists():
        return str(default_python)
    return sys.executable


def _default_unsloth_server() -> str:
    configured = os.environ.get("UNSLOTH_SERVER")
    if configured:
        return configured
    default_server = _default_unsloth_server_path()
    if default_server.exists():
        return str(default_server)
    return "unsloth"


def _existing_file(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_file() else None


def _default_unsloth_dg_shim() -> Path | None:
    configured = _existing_file(os.environ.get("UNSLOTH_DG_SHIM"))
    if configured:
        return configured
    installed_shim = _default_unsloth_site_packages_dir() / "unsloth_zoo" / "diffusion_studio" / "shim.py"
    if installed_shim.is_file():
        return installed_shim
    return DEFAULT_UNSLOTH_DG_SHIM if DEFAULT_UNSLOTH_DG_SHIM.is_file() else None


def _default_llama_server_path() -> Path | None:
    configured = _existing_file(os.environ.get("LLAMA_SERVER_PATH"))
    if configured:
        return configured
    for candidate in _default_llama_server_candidates():
        if candidate.is_file():
            return candidate
    return None


def _default_dg_visual_bin() -> Path | None:
    configured = _existing_file(os.environ.get("DG_VISUAL_BIN"))
    if configured:
        return configured
    for candidate in _default_dg_visual_bin_candidates():
        if candidate.is_file():
            return candidate
    return None


def _configured_dg_visual_bin(args: argparse.Namespace) -> Path | None:
    return _existing_file(getattr(args, "dg_visual_bin", None)) or _default_dg_visual_bin()


def _configured_unsloth_dg_shim(args: argparse.Namespace) -> Path | None:
    return _existing_file(getattr(args, "unsloth_dg_shim", None)) or _default_unsloth_dg_shim()


def _cmd_env_prefix(assignments: dict[str, str | Path | None]) -> str:
    parts = []
    for key, value in assignments.items():
        if value not in (None, ""):
            clean_value = str(value).replace("\\", "/")
            if _is_windows_host():
                parts.append(f'set "{key}={clean_value.replace(chr(34), "")}"')
            else:
                parts.append(f"{key}={shlex.quote(clean_value)}")
    separator = " && " if _is_windows_host() else " "
    return separator.join(parts) + (separator if parts else "")


def _q(value: str | Path) -> str:
    text = str(value).replace("\\", "/").replace('"', '\\"')
    return f'"{text}"'


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml(item)}" for key, item in value.items()) + " }"
    return json.dumps(str(value), ensure_ascii=False)


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _ctx_label(ctx: int) -> str:
    return f"{ctx // 1024}k"


def _stem(path: Path) -> str:
    return path.name[:-5] if path.name.lower().endswith(".gguf") else path.name


def _base_recipe(catalog: Catalog, recipe_id: str) -> Any:
    return catalog.get(recipe_id)


def _write_recipe(path: Path, *, base: Any, recipe_id: str, name: str, description: str, tags: list[str], classification: dict[str, Any], runtime: dict[str, Any], parameters: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"id = {_toml(recipe_id)}",
        f"category = {_toml(base.category)}",
        f"name = {_toml(name)}",
        f"description = {_toml(description)}",
        f"tags = {_toml(tags)}",
        "",
        "[classification]",
    ]
    for key, value in classification.items():
        lines.append(f"{key} = {_toml(value)}")
    lines.extend(["", "[runtime]"])
    for key, value in runtime.items():
        if value is not None:
            lines.append(f"{key} = {_toml(value)}")
    lines.extend(["", "[parameters]"])
    for key, value in parameters.items():
        if value is not None:
            lines.append(f"{key} = {_toml(value)}")

    for case in base.dataset.get("cases", []):
        lines.extend(["", "[[dataset.cases]]"])
        for key, value in case.items():
            lines.append(f"{key} = {_toml(value)}")

    lines.extend(["", "[kpis]"])
    for key, value in base.kpis.items():
        lines.append(f"{key} = {_toml(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_models(args: argparse.Namespace) -> list[ModelConfig]:
    candidates = {
        "12b": ModelConfig(
            key="12b",
            label="Gemma 4 12B QAT",
            family="gemma-4-12b-qat",
            primary_gguf=Path(args.gemma12_model_gguf) if args.gemma12_model_gguf else _env_path("GEMMA12_MODEL_GGUF", DEFAULT_12B_MODEL),
            draft_gguf=Path(args.gemma12_mtp_draft_gguf) if args.gemma12_mtp_draft_gguf else _env_path("GEMMA12_MTP_DRAFT_GGUF", DEFAULT_12B_DRAFT),
        ),
        "26b": ModelConfig(
            key="26b",
            label="Gemma 4 26B-A4B",
            family="gemma-4-26b-a4b",
            primary_gguf=Path(args.gemma26_model_gguf) if args.gemma26_model_gguf else _env_path("GEMMA26_MODEL_GGUF", DEFAULT_26B_MODEL),
            draft_gguf=Path(args.gemma26_mtp_draft_gguf) if args.gemma26_mtp_draft_gguf else _env_path("GEMMA26_MTP_DRAFT_GGUF", DEFAULT_26B_DRAFT),
        ),
    }
    models: list[ModelConfig] = []
    for key in args.model:
        model = candidates[key]
        required = [model.primary_gguf]
        if any(mode in args.mode for mode in ("mtp", "unsloth", "mtp-unsloth")):
            if model.draft_gguf is not None:
                required.append(model.draft_gguf)
        if args.allow_missing or all(path.exists() for path in required):
            models.append(model)
        else:
            print(f"skip {key}: missing {', '.join(str(path) for path in required if not path.exists())}", file=sys.stderr)
    return models


def _common_runtime(target_gpu: str, base_url: str, model_name: str, start_command: str | None, stop_command: str | None, ready_url: str | None, timeout_seconds: int) -> dict[str, Any]:
    return {
        "adapter": "openai_compatible_chat",
        "target_gpu": target_gpu,
        "base_url": base_url,
        "model": model_name,
        "start_command": start_command,
        "stop_command": stop_command,
        "ready_url": ready_url,
        "ready_timeout_seconds": 1200,
        "ready_poll_seconds": 2,
        "startup_grace_seconds": 2,
        "unload_wait_seconds": 10,
        "system_prompt": "You are a precise benchmark participant. Follow the user's formatting constraints exactly. Do not include benchmark commentary.",
        "timeout_seconds": timeout_seconds,
    }


def _unsloth_base_url(args: argparse.Namespace) -> str:
    return (
        getattr(args, "unsloth_base_url", None)
        or getattr(args, "unsloth_diffusion_base_url", None)
        or os.environ.get("UNSLOTH_BASE_URL")
        or os.environ.get("UNSLOTH_DIFFUSION_BASE_URL")
        or f"http://127.0.0.1:{getattr(args, 'unsloth_port', 18084)}/v1"
    )


def _unsloth_api_root(args: argparse.Namespace) -> str:
    normalized = _unsloth_base_url(args).rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def _unsloth_ready_url(args: argparse.Namespace) -> str:
    return f"{_unsloth_api_root(args)}/api/inference/status"


def _unsloth_stop_command(args: argparse.Namespace) -> str:
    return f"{_q(sys.executable)} {_q(ROOT / 'scripts/kill_port.py')} {getattr(args, 'unsloth_port', 18084)}"


def _unsloth_start_command(model_path: Path, ctx: int, args: argparse.Namespace, extra_args: str | None = None) -> str:
    llama_server_path = getattr(args, "unsloth_llama_server_path", None) or os.environ.get("LLAMA_SERVER_PATH")
    if not llama_server_path and getattr(args, "llama_server", None):
        candidate = Path(str(args.llama_server)).expanduser()
        if candidate.exists():
            llama_server_path = str(candidate)
    if not llama_server_path:
        default_llama_server_path = _default_llama_server_path()
        if default_llama_server_path is not None:
            llama_server_path = str(default_llama_server_path)
    env_prefix = _cmd_env_prefix(
        {
            "CUDA_VISIBLE_DEVICES": getattr(args, "cuda_visible_devices", None) or os.environ.get("UNSLOTH_CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LLAMA_SERVER_PATH": llama_server_path,
            "DG_VISUAL_BIN": _configured_dg_visual_bin(args),
            "UNSLOTH_DG_SHIM": _configured_unsloth_dg_shim(args),
        }
    )
    port = getattr(args, "unsloth_port", 18084)
    server = getattr(args, "unsloth_server", None) or _default_unsloth_server()
    passthrough_args = _join_extra_args(getattr(args, "unsloth_extra_args", None), extra_args)
    return (
        f"{env_prefix}{_q(server)} run --model {_q(model_path)} "
        f"--port {port} --host 127.0.0.1 --max-seq-length {ctx} "
        f"--no-cloudflare --disable-tools{_extra_args(passthrough_args)}"
    )


def _unsloth_runtime(model_path: Path, model_name: str, ctx: int, memory_key: str, args: argparse.Namespace, timeout_seconds: int, extra_args: str | None = None) -> dict[str, Any]:
    runtime = _common_runtime(
        args.target_gpu,
        _unsloth_base_url(args),
        model_name,
        _unsloth_start_command(model_path, ctx, args, extra_args),
        _unsloth_stop_command(args),
        _unsloth_ready_url(args),
        timeout_seconds,
    )
    runtime["model_memory_key"] = memory_key
    runtime["ready_json_key"] = "active_model"
    api_key = getattr(args, "unsloth_api_key", None)
    if api_key:
        os.environ["AIRECEIPES_UNSLOTH_API_KEY"] = str(api_key)
        runtime["api_key_env"] = "AIRECEIPES_UNSLOTH_API_KEY"
        runtime["ready_api_key_env"] = "AIRECEIPES_UNSLOTH_API_KEY"
    return runtime


def _diffusion_unsloth_runtime_overrides(args: argparse.Namespace) -> dict[str, Any]:
    visual_bin = _configured_dg_visual_bin(args)
    shim = _configured_unsloth_dg_shim(args)
    overrides: dict[str, Any] = {
        "dg_visual_bin": str(visual_bin) if visual_bin else None,
        "unsloth_dg_shim": str(shim) if shim else None,
    }
    if visual_bin and shim:
        return overrides
    missing = []
    if not visual_bin:
        missing.append("DiffusionGemma visual-server binary")
    if not shim:
        missing.append("Unsloth DiffusionGemma shim.py")
    overrides.update(
        {
            "skip_preflight_reason": "missing_diffusiongemma_visual_server" if not visual_bin else "missing_diffusiongemma_shim",
            "skip_preflight_detail": (
                f"{', '.join(missing)} was not found. "
                "Install/build llama-diffusion-gemma-visual-server.exe and set DG_VISUAL_BIN, "
                "and ensure UNSLOTH_DG_SHIM points to unsloth_zoo/diffusion_studio/shim.py."
            ),
        }
    )
    return overrides


def _unsloth_model_name(model_key: str, mode: str, reasoning: str, ctx: int, args: argparse.Namespace) -> str:
    prefix = getattr(args, "unsloth_model_prefix", None) or os.environ.get("UNSLOTH_MODEL_PREFIX", "unsloth")
    return f"{prefix}-{model_key}-{mode}-{reasoning}-ctx{_ctx_label(ctx)}"


def _parameters(base: Any, reasoning: str, args: argparse.Namespace) -> dict[str, Any]:
    params = dict(base.parameters)
    params["reasoning"] = reasoning == "on"
    if args.thinking_max_tokens is not None and reasoning == "on":
        params["max_tokens"] = args.thinking_max_tokens
    elif args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens
    return params


def _server_reasoning_flag(reasoning: str) -> str:
    return f" --reasoning {reasoning}"


def _extra_args(value: str | None) -> str:
    return f" {value.strip()}" if value and value.strip() else ""


def _join_extra_args(*values: str | None) -> str | None:
    parts = [str(value).strip() for value in values if value and str(value).strip()]
    return " ".join(parts) if parts else None


def _make_regular(model: ModelConfig, ctx: int, reasoning: str, args: argparse.Namespace, recipes_dir: Path, catalog: Catalog) -> MatrixSpec:
    base = _base_recipe(catalog, TEXT_BASE_RECIPE)
    port = args.regular_port
    model_name = f"{model.key}-regular-{reasoning}-ctx{_ctx_label(ctx)}"
    recipe_id = f"inference.matrix-{_safe_id(model_name)}"
    command = (
        f"{_q(args.llama_server)} -m {_q(model.primary_gguf)} --alias {_q(model_name)} --host 127.0.0.1 --port {port} "
        f"-c {ctx} -ngl 99 --device {args.gpu_device} --split-mode none --flash-attn on -b {args.batch_size} -ub {args.ubatch_size}"
        f"{_server_reasoning_flag(reasoning)}{_extra_args(args.regular_extra_args)}"
    )
    path = recipes_dir / "inference" / f"{recipe_id.split('.')[-1]}.toml"
    _write_recipe(
        path,
        base=base,
        recipe_id=recipe_id,
        name=f"{model.label} regular {reasoning} ctx {_ctx_label(ctx)}",
        description="Matrix benchmark generated for regular autoregressive Gemma 4.",
        tags=["matrix", "gemma4", model.key, "regular", reasoning, f"ctx-{_ctx_label(ctx)}"],
        classification={
            "task": "gemma4-matrix-bakeoff",
            "modality": "text",
            "size": "vertical-smoke",
            "model_family": model.family,
            "variant": "regular-autoregressive",
            "mode": "regular",
            "reasoning": reasoning,
            "context_size": ctx,
            "backend": "llama.cpp",
            "primary_model_filename": model.primary_gguf.name,
        },
        runtime=_common_runtime(args.target_gpu, f"http://127.0.0.1:{port}/v1", model_name, command, f"{_q(sys.executable)} {_q(ROOT / 'scripts/kill_port.py')} {port}", f"http://127.0.0.1:{port}/v1/models", args.timeout_seconds),
        parameters=_parameters(base, reasoning, args),
    )
    return MatrixSpec(recipe_id, f"{model.label} regular {reasoning} ctx {_ctx_label(ctx)}", "regular", model.key, ctx, reasoning, None, "llama.cpp", path, command)


def _make_mtp(model: ModelConfig, ctx: int, reasoning: str, draft_n: int, args: argparse.Namespace, recipes_dir: Path, catalog: Catalog) -> MatrixSpec:
    base = _base_recipe(catalog, TEXT_BASE_RECIPE)
    port = args.mtp_port
    if model.draft_gguf is None:
        raise ValueError(f"No draft GGUF configured for {model.key}")
    model_name = f"{model.key}-mtp-n{draft_n}-{reasoning}-ctx{_ctx_label(ctx)}"
    recipe_id = f"inference.matrix-{_safe_id(model_name)}"
    command = (
        f"{_q(args.llama_server)} -m {_q(model.primary_gguf)} --alias {_q(model_name)} --host 127.0.0.1 --port {port} "
        f"-c {ctx} -ngl 99 --device {args.gpu_device} --split-mode none --flash-attn on -b {args.batch_size} -ub {args.ubatch_size} "
        f"--spec-type draft-mtp --spec-draft-n-max {draft_n} --model-draft {_q(model.draft_gguf)} --spec-draft-ngl 99 --spec-draft-device {args.gpu_device}"
        f"{_server_reasoning_flag(reasoning)}{_extra_args(args.mtp_extra_args)}"
    )
    path = recipes_dir / "inference" / f"{recipe_id.split('.')[-1]}.toml"
    _write_recipe(
        path,
        base=base,
        recipe_id=recipe_id,
        name=f"{model.label} MTP n={draft_n} {reasoning} ctx {_ctx_label(ctx)}",
        description="Matrix benchmark generated for Gemma 4 MTP speculative decoding.",
        tags=["matrix", "gemma4", model.key, "mtp", f"n-{draft_n}", reasoning, f"ctx-{_ctx_label(ctx)}"],
        classification={
            "task": "gemma4-matrix-bakeoff",
            "modality": "text",
            "size": "vertical-smoke",
            "model_family": model.family,
            "variant": "mtp-draft",
            "mode": "mtp",
            "mtp_draft_n": draft_n,
            "reasoning": reasoning,
            "context_size": ctx,
            "backend": "llama.cpp",
            "primary_model_filename": model.primary_gguf.name,
            "draft_model_filename": model.draft_gguf.name,
        },
        runtime=_common_runtime(args.target_gpu, f"http://127.0.0.1:{port}/v1", model_name, command, f"{_q(sys.executable)} {_q(ROOT / 'scripts/kill_port.py')} {port}", f"http://127.0.0.1:{port}/v1/models", args.timeout_seconds),
        parameters=_parameters(base, reasoning, args),
    )
    return MatrixSpec(recipe_id, f"{model.label} MTP n={draft_n} {reasoning} ctx {_ctx_label(ctx)}", "mtp", model.key, ctx, reasoning, draft_n, "llama.cpp", path, command)


def _make_unsloth_regular(model: ModelConfig, ctx: int, reasoning: str, args: argparse.Namespace, recipes_dir: Path, catalog: Catalog) -> MatrixSpec:
    base = _base_recipe(catalog, TEXT_BASE_RECIPE)
    model_name = _unsloth_model_name(model.key, "regular", reasoning, ctx, args)
    recipe_id = f"inference.matrix-unsloth-{_safe_id(model.key)}-regular-{reasoning}-ctx{_ctx_label(ctx)}"
    path = recipes_dir / "inference" / f"{recipe_id.split('.')[-1]}.toml"
    _write_recipe(
        path,
        base=base,
        recipe_id=recipe_id,
        name=f"{model.label} Unsloth regular {reasoning} ctx {_ctx_label(ctx)}",
        description="Matrix benchmark generated for regular Gemma 4 through an Unsloth/OpenAI-compatible endpoint.",
        tags=["matrix", "gemma4", model.key, "regular", "unsloth", reasoning, f"ctx-{_ctx_label(ctx)}"],
        classification={
            "task": "gemma4-matrix-bakeoff",
            "modality": "text",
            "size": "vertical-smoke",
            "model_family": model.family,
            "variant": "regular-autoregressive-unsloth-endpoint",
            "mode": "regular",
            "reasoning": reasoning,
            "context_size": ctx,
            "backend": "unsloth-studio-or-compatible",
            "primary_model_filename": model.primary_gguf.name,
        },
        runtime=_unsloth_runtime(model.primary_gguf, model_name, ctx, f"unsloth:{model.key}:ctx{ctx}", args, args.timeout_seconds),
        parameters=_parameters(base, reasoning, args),
    )
    return MatrixSpec(recipe_id, f"{model.label} Unsloth regular {reasoning} ctx {_ctx_label(ctx)}", "regular", model.key, ctx, reasoning, None, "unsloth-studio-or-compatible", path, None)


def _make_unsloth_mtp(model: ModelConfig, ctx: int, reasoning: str, draft_n: int, args: argparse.Namespace, recipes_dir: Path, catalog: Catalog) -> MatrixSpec:
    base = _base_recipe(catalog, TEXT_BASE_RECIPE)
    if model.draft_gguf is None:
        raise ValueError(f"No draft GGUF configured for {model.key}")
    model_name = _unsloth_model_name(model.key, f"mtp-n{draft_n}", reasoning, ctx, args)
    recipe_id = f"inference.matrix-unsloth-{_safe_id(model.key)}-mtp-n{draft_n}-{reasoning}-ctx{_ctx_label(ctx)}"
    path = recipes_dir / "inference" / f"{recipe_id.split('.')[-1]}.toml"
    gpu_device = getattr(args, "gpu_device", "CUDA0")
    mtp_passthrough_args = (
        f"--spec-type draft-mtp --spec-draft-n-max {draft_n} --model-draft {_q(model.draft_gguf)} "
        f"--spec-draft-ngl 99 --spec-draft-device {gpu_device}"
    )
    _write_recipe(
        path,
        base=base,
        recipe_id=recipe_id,
        name=f"{model.label} Unsloth MTP n={draft_n} {reasoning} ctx {_ctx_label(ctx)}",
        description="Matrix benchmark generated for the MTP slice through an Unsloth/OpenAI-compatible endpoint with llama-server MTP pass-through flags.",
        tags=["matrix", "gemma4", model.key, "mtp", "unsloth", f"n-{draft_n}", reasoning, f"ctx-{_ctx_label(ctx)}"],
        classification={
            "task": "gemma4-matrix-bakeoff",
            "modality": "text",
            "size": "vertical-smoke",
            "model_family": model.family,
            "variant": "mtp-draft-unsloth-endpoint",
            "mode": "mtp",
            "mtp_draft_n": draft_n,
            "reasoning": reasoning,
            "context_size": ctx,
            "backend": "unsloth-studio-or-compatible",
            "primary_model_filename": model.primary_gguf.name,
            "draft_model_filename": model.draft_gguf.name,
        },
        runtime=_unsloth_runtime(
            model.primary_gguf,
            model_name,
            ctx,
            f"unsloth:{model.key}:mtp-n{draft_n}:ctx{ctx}",
            args,
            args.timeout_seconds,
            mtp_passthrough_args,
        ),
        parameters=_parameters(base, reasoning, args),
    )
    return MatrixSpec(recipe_id, f"{model.label} Unsloth MTP n={draft_n} {reasoning} ctx {_ctx_label(ctx)}", "mtp", model.key, ctx, reasoning, draft_n, "unsloth-studio-or-compatible", path, None)


def _make_diffusion(ctx: int, reasoning: str, args: argparse.Namespace, recipes_dir: Path, catalog: Catalog) -> MatrixSpec | None:
    base = _base_recipe(catalog, DIFFUSION_BASE_RECIPE)
    diffusion_arg = getattr(args, "diffusion_gguf", None)
    diffusion_gguf = Path(diffusion_arg) if diffusion_arg else _env_path("DIFFUSION_GGUF", _default_diffusion_gguf())
    if not args.allow_missing and not diffusion_gguf.exists():
        print(f"skip diffusion: missing {diffusion_gguf}", file=sys.stderr)
        return None
    port = args.diffusion_port
    model_name = _stem(diffusion_gguf)
    recipe_name_part = f"diffusiongemma-llamacpp-{reasoning}-ctx{_ctx_label(ctx)}"
    recipe_id = f"inference.matrix-{_safe_id(recipe_name_part)}"
    command = (
        f"{_q(args.diffusion_server)} -m {_q(diffusion_gguf)} --host 127.0.0.1 --port {port} "
        f"-c {ctx} -ngl 99 --device {args.gpu_device} --split-mode none --flash-attn on -b {args.batch_size} -ub {args.ubatch_size} "
        f"--diffusion-steps {args.diffusion_steps} --diffusion-block-length {args.diffusion_block_length}{_extra_args(args.diffusion_extra_args)}"
    )
    path = recipes_dir / "inference" / f"{recipe_id.split('.')[-1]}.toml"
    _write_recipe(
        path,
        base=base,
        recipe_id=recipe_id,
        name=f"DiffusionGemma llama.cpp {reasoning} ctx {_ctx_label(ctx)}",
        description="Matrix benchmark generated for DiffusionGemma through llama-diffusion-gemma-server. Reasoning is sent as a payload flag when requested; this server may ignore it.",
        tags=["matrix", "diffusiongemma", "llama.cpp", reasoning, f"ctx-{_ctx_label(ctx)}"],
        classification={
            "task": "gemma4-matrix-bakeoff",
            "modality": "text",
            "size": "vertical-smoke",
            "model_family": "diffusiongemma-26b-a4b",
            "variant": "diffusion-block-denoising",
            "mode": "diffusion",
            "reasoning": reasoning,
            "reasoning_application": "payload_only",
            "context_size": ctx,
            "backend": "llama.cpp-diffusion",
            "primary_model_filename": diffusion_gguf.name,
            "source_urls": ["https://unsloth.ai/docs/models/diffusiongemma", "https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF"],
            "reference_claim": "Unsloth docs state DiffusionGemma can reach 2000+ tokens/s on an RTX 6000 via Unsloth Studio or llama.cpp.",
        },
        runtime=_common_runtime(args.target_gpu, f"http://127.0.0.1:{port}/v1", model_name, command, f"{_q(sys.executable)} {_q(ROOT / 'scripts/kill_port.py')} {port}", f"http://127.0.0.1:{port}/v1/models", args.diffusion_timeout_seconds),
        parameters=_parameters(base, reasoning, args),
    )
    return MatrixSpec(recipe_id, f"DiffusionGemma llama.cpp {reasoning} ctx {_ctx_label(ctx)}", "diffusion", "26b-a4b", ctx, reasoning, None, "llama.cpp-diffusion", path, command)


def _make_unsloth_diffusion(ctx: int, reasoning: str, args: argparse.Namespace, recipes_dir: Path, catalog: Catalog) -> MatrixSpec | None:
    diffusion_arg = getattr(args, "diffusion_gguf", None)
    diffusion_gguf = Path(diffusion_arg) if diffusion_arg else _env_path("DIFFUSION_GGUF", _default_diffusion_gguf())
    if not args.allow_missing and not diffusion_gguf.exists():
        print(f"skip diffusion unsloth: missing {diffusion_gguf}", file=sys.stderr)
        return None
    base_url = _unsloth_base_url(args)
    base = _base_recipe(catalog, DIFFUSION_BASE_RECIPE)
    model_name = args.unsloth_diffusion_model or os.environ.get("UNSLOTH_DIFFUSION_MODEL", "diffusiongemma-26B-A4B-it")
    recipe_name_part = f"diffusiongemma-unsloth-{reasoning}-ctx{_ctx_label(ctx)}"
    recipe_id = f"inference.matrix-{_safe_id(recipe_name_part)}"
    path = recipes_dir / "inference" / f"{recipe_id.split('.')[-1]}.toml"
    runtime = _unsloth_runtime(diffusion_gguf, model_name, ctx, f"unsloth:diffusiongemma:ctx{ctx}", args, args.diffusion_timeout_seconds)
    runtime.update(_diffusion_unsloth_runtime_overrides(args))
    _write_recipe(
        path,
        base=base,
        recipe_id=recipe_id,
        name=f"DiffusionGemma Unsloth endpoint {reasoning} ctx {_ctx_label(ctx)}",
        description="Matrix benchmark generated for an already-running Unsloth Studio/OpenAI-compatible DiffusionGemma endpoint.",
        tags=["matrix", "diffusiongemma", "unsloth", reasoning, f"ctx-{_ctx_label(ctx)}"],
        classification={
            "task": "gemma4-matrix-bakeoff",
            "modality": "text",
            "size": "vertical-smoke",
            "model_family": "diffusiongemma-26b-a4b",
            "variant": "diffusion-block-denoising-unsloth-endpoint",
            "mode": "diffusion-unsloth",
            "reasoning": reasoning,
            "reasoning_application": "payload_only",
            "context_size": ctx,
            "backend": "unsloth-studio-or-compatible",
            "primary_model_filename": diffusion_gguf.name,
            "source_urls": ["https://unsloth.ai/docs/models/diffusiongemma"],
            "reference_claim": "Unsloth docs state DiffusionGemma can reach 2000+ tokens/s on an RTX 6000 via Unsloth Studio or llama.cpp.",
        },
        runtime=runtime,
        parameters=_parameters(base, reasoning, args),
    )
    return MatrixSpec(recipe_id, f"DiffusionGemma Unsloth endpoint {reasoning} ctx {_ctx_label(ctx)}", "diffusion-unsloth", "26b-a4b", ctx, reasoning, None, "unsloth-studio-or-compatible", path, None)


def _build_matrix(args: argparse.Namespace, recipes_dir: Path) -> list[MatrixSpec]:
    catalog = Catalog(args.recipes_dir)
    specs: list[MatrixSpec] = []
    models = _build_models(args)
    for ctx in args.context:
        for reasoning in args.reasoning:
            for model in models:
                if "regular" in args.mode:
                    specs.append(_make_regular(model, ctx, reasoning, args, recipes_dir, catalog))
                if "mtp" in args.mode:
                    for draft_n in args.mtp_draft_n:
                        specs.append(_make_mtp(model, ctx, reasoning, draft_n, args, recipes_dir, catalog))
                if "unsloth" in args.mode:
                    specs.append(_make_unsloth_regular(model, ctx, reasoning, args, recipes_dir, catalog))
                    for draft_n in args.mtp_draft_n:
                        specs.append(_make_unsloth_mtp(model, ctx, reasoning, draft_n, args, recipes_dir, catalog))
                elif "mtp-unsloth" in args.mode:
                    for draft_n in args.mtp_draft_n:
                        specs.append(_make_unsloth_mtp(model, ctx, reasoning, draft_n, args, recipes_dir, catalog))
            if "diffusion" in args.mode:
                spec = _make_diffusion(ctx, reasoning, args, recipes_dir, catalog)
                if spec is not None:
                    specs.append(spec)
            if "diffusion-unsloth" in args.mode or "unsloth" in args.mode:
                spec = _make_unsloth_diffusion(ctx, reasoning, args, recipes_dir, catalog)
                if spec is not None:
                    specs.append(spec)
    if args.orchestration_order == "lifecycle":
        specs = sorted(specs, key=_orchestration_sort_key)
    if args.limit is not None:
        specs = specs[: args.limit]
    return specs


def _orchestration_sort_key(spec: MatrixSpec) -> tuple[Any, ...]:
    backend_order = {
        "llama.cpp": 0,
        "llama.cpp-diffusion": 1,
        "unsloth-studio-or-compatible": 2,
    }
    mode_order = {"regular": 0, "mtp": 1, "diffusion": 2, "diffusion-unsloth": 3}
    reasoning_order = {"off": 0, "on": 1}
    if spec.backend == "unsloth-studio-or-compatible":
        return (
            backend_order.get(spec.backend, 99),
            spec.model_key,
            spec.context_size,
            mode_order.get(spec.mode, 99),
            spec.mtp_draft_n or 0,
            reasoning_order.get(spec.reasoning, 99),
            spec.recipe_id,
        )
    return (
        backend_order.get(spec.backend, 99),
        spec.model_key,
        mode_order.get(spec.mode, 99),
        spec.mtp_draft_n or 0,
        spec.context_size,
        reasoning_order.get(spec.reasoning, 99),
        spec.recipe_id,
    )


def _missing_python_modules(python_executable: str, modules: list[str]) -> list[str]:
    check_code = """
import importlib.util
import json
import sys
modules = sys.argv[1:]
missing = [module for module in modules if importlib.util.find_spec(module) is None]
print(json.dumps(missing))
"""
    completed = subprocess.run(
        [python_executable, "-c", check_code, *modules],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to inspect Unsloth Python environment: {completed.stderr.strip()}")
    return json.loads(completed.stdout or "[]")


def _create_unsloth_api_key(python_executable: str, name: str) -> str:
    code = """
import sys
from pathlib import Path
import studio.backend
backend_dir = Path(studio.backend.__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
from auth.storage import create_api_key
raw_key, _row = create_api_key("cli", sys.argv[1], internal=True)
print(raw_key)
"""
    completed = subprocess.run(
        [python_executable, "-c", code, name],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to create Unsloth Studio API key: {completed.stderr.strip()}")
    api_key = completed.stdout.strip().splitlines()[-1].strip()
    if not api_key.startswith("sk-unsloth-"):
        raise RuntimeError("Unsloth Studio API key creation returned an unexpected value")
    return api_key


def _ensure_unsloth_api_key(args: argparse.Namespace, report: dict[str, Any] | None) -> None:
    if not any("unsloth" in mode for mode in args.mode):
        return
    if getattr(args, "unsloth_api_key", None):
        if report is not None:
            report["api_key_source"] = "provided"
        return
    python_executable = str(report.get("python") if report else _default_unsloth_python())
    api_key = _create_unsloth_api_key(python_executable, f"aireceipes-{args.run_id}")
    args.unsloth_api_key = api_key
    if report is not None:
        report["api_key_source"] = "generated"
        report["api_key_prefix"] = api_key[:20]


def _ensure_unsloth_if_requested(args: argparse.Namespace) -> dict[str, Any] | None:
    if not any("unsloth" in mode for mode in args.mode):
        return None
    python_executable = _default_unsloth_python()
    module_names = ["unsloth", "torch", "fastapi", "uvicorn"]
    missing_modules = _missing_python_modules(python_executable, module_names)
    report: dict[str, Any] = {
        "python": python_executable,
        "checked_modules": module_names,
        "missing_modules": missing_modules,
        "install_requested": bool(args.install_unsloth_if_required),
    }
    if not missing_modules:
        report["status"] = "available"
        return report
    if not args.install_unsloth_if_required:
        report["status"] = "missing_not_installed"
        print(
            "Unsloth backend packages are missing "
            f"({', '.join(missing_modules)}). Pass --install-unsloth-if-required to install them, "
            "or point --unsloth-diffusion-base-url at an already-running endpoint.",
            file=sys.stderr,
        )
        return report

    packages = args.unsloth_install_package or DEFAULT_UNSLOTH_INSTALL_PACKAGES
    install_command = [python_executable, "-m", "pip", "install", "--upgrade", *packages]
    completed = subprocess.run(
        install_command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=args.unsloth_install_timeout_seconds,
    )
    report.update(
        {
            "status": "installed" if completed.returncode == 0 else "install_failed",
            "install_command": install_command,
            "install_returncode": completed.returncode,
            "install_stdout_tail": completed.stdout[-4000:],
            "install_stderr_tail": completed.stderr[-4000:],
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Unsloth dependency installation failed. "
            f"Command: {' '.join(install_command)}\n{completed.stderr[-2000:]}"
        )
    report["missing_modules_after_install"] = _missing_python_modules(python_executable, module_names)
    return report


def _add_unsloth_preflight_to_comparison(comparison: dict[str, Any], report: dict[str, Any] | None) -> None:
    if not report:
        return
    missing_modules = report.get("missing_modules_after_install") or report.get("missing_modules") or []
    status = str(report.get("status") or "unknown")
    if missing_modules and status == "installed":
        status = "installed_but_still_missing"
    message = "Unsloth dependency preflight passed."
    if report.get("benchmark_specs_skipped"):
        message = "Unsloth benchmark specs were skipped because local dependencies are missing and the configured endpoint is unreachable."
    elif missing_modules:
        message = "Missing Unsloth backend dependencies are recorded here so the backend test is visible in comparison.html."
    elif report.get("install_requested"):
        message = "Unsloth dependencies were installed/rechecked before running backend tests."
    check = {
        "name": "Unsloth backend dependency test",
        "backend": "unsloth-studio-or-compatible",
        "status": status,
        "python": report.get("python"),
        "checked_modules": report.get("checked_modules", []),
        "missing_modules": missing_modules,
        "install_requested": report.get("install_requested"),
        "install_command": report.get("install_command"),
        "message": message,
    }
    if report.get("endpoint_status"):
        check["endpoint_status"] = report.get("endpoint_status")
    if report.get("benchmark_specs_skipped"):
        check["benchmark_specs_skipped"] = True
    comparison.setdefault("preflight_checks", []).append(check)


def _unsloth_endpoint_reachable(base_url: str | None, timeout_seconds: float) -> bool:
    if not base_url:
        return False
    normalized = base_url.rstrip("/")
    url = normalized if normalized.endswith("/models") else f"{normalized}/models"
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= int(getattr(response, "status", 200)) < 500
    except Exception:
        return False


def _drop_unrunnable_unsloth_specs(args: argparse.Namespace, report: dict[str, Any] | None) -> None:
    if not report or not any("unsloth" in mode for mode in args.mode):
        return
    missing_modules = report.get("missing_modules_after_install") or report.get("missing_modules") or []
    if not missing_modules or getattr(args, "force_unsloth_run_on_missing", False):
        return
    base_url = _unsloth_base_url(args)
    timeout_seconds = float(getattr(args, "unsloth_endpoint_timeout_seconds", 2.0))
    if _unsloth_endpoint_reachable(base_url, timeout_seconds):
        report["endpoint_status"] = "reachable"
        return
    report["endpoint_status"] = "unreachable"
    report["benchmark_specs_skipped"] = True
    report["benchmark_specs_skip_reason"] = "missing_unsloth_dependencies_and_unreachable_endpoint"
    print(
        "Marking Unsloth benchmark specs as skipped: local Unsloth dependencies are missing "
        "and the configured endpoint is unreachable. The requested Unsloth matrix will still appear in comparison.html.",
        file=sys.stderr,
    )



def _write_plan(specs: list[MatrixSpec], output_dir: Path, run_id: str, args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / f"{run_id}-matrix-plan.json"
    plan = {
        "run_id": run_id,
        "count": len(specs),
        "contexts": args.context,
        "reasoning": args.reasoning,
        "mtp_draft_n": args.mtp_draft_n,
        "modes": args.mode,
        "models": args.model,
        "orchestration_order": args.orchestration_order,
        "lifecycle_reuse": not args.no_lifecycle_reuse,
        "wrapper_command_file": str(args.wrapper_command_file) if args.wrapper_command_file else None,
        "result_root": str(output_dir),
        "result_variable_note": "$result is set to the concrete suite directory before each wrapper/start/stop command; $test_result is set to the per-test metrics directory.",
        "unsloth_install_requested": bool(args.install_unsloth_if_required),
        "unsloth_install_report": getattr(args, "unsloth_install_report", None),
        "specs": [spec.__dict__ | {"recipe_path": str(spec.recipe_path)} for spec in specs],
    }
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return plan_path


def _run_dir_name(index: int, recipe_id: str) -> str:
    return f"{index:02d}-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", recipe_id)


def _failure_result(recipe: Any, suite_dir: Path, index: int, exc: Exception, lifecycle: dict[str, Any]) -> RunResult:
    run_dir = suite_dir / _run_dir_name(index, recipe.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "recipe_id": recipe.id,
        "category": recipe.category,
        "classification": recipe.classification,
        "adapter": recipe.runtime.get("adapter", ""),
        "started_at": lifecycle.get("started_at"),
        "finished_at": lifecycle.get("finished_at"),
        "prompt_count": 0,
        "success_count": 0,
        "error_count": 1,
        "kpis": {},
        "verticals": {},
        "results": [],
        "error": str(exc),
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return RunResult(run_dir=run_dir, metrics_path=metrics_path, metrics=metrics)


def _skipped_result(recipe: Any, suite_dir: Path, index: int, reason: str, lifecycle: dict[str, Any]) -> RunResult:
    run_dir = suite_dir / _run_dir_name(index, recipe.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_metrics = recipe.kpis.get("metrics")
    kpis = {metric: None for metric in requested_metrics} if isinstance(requested_metrics, list) else {}
    metrics = {
        "recipe_id": recipe.id,
        "category": recipe.category,
        "classification": recipe.classification,
        "adapter": recipe.runtime.get("adapter", ""),
        "started_at": lifecycle.get("started_at"),
        "finished_at": lifecycle.get("finished_at"),
        "prompt_count": 0,
        "success_count": 0,
        "error_count": 0,
        "skipped_count": 1,
        "skip_reason": reason,
        "kpis": kpis,
        "verticals": {},
        "results": [],
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return RunResult(run_dir=run_dir, metrics_path=metrics_path, metrics=metrics)


def _run_matrix_suite(
    specs: list[MatrixSpec],
    recipes_dir: Path,
    output_dir: Path,
    run_id: str,
    *,
    lifecycle: bool = True,
    test_wrapper_command: str | None = None,
    reuse_lifecycle: bool = True,
    unsloth_install_report: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    catalog = Catalog(recipes_dir)
    suite_dir = output_dir / f"{run_id}-gemma-matrix-bakeoff"
    suite_dir.mkdir(parents=True, exist_ok=True)
    recipe_runs: list[RecipeRunReport] = []
    active_process: subprocess.Popen[str] | None = None
    active_key: str | None = None
    active_recipe: Any | None = None
    active_lifecycle: dict[str, Any] | None = None

    for index, spec in enumerate(specs, start=1):
        recipe = catalog.get(spec.recipe_id)
        next_spec = specs[index] if index < len(specs) else None
        next_recipe = catalog.get(next_spec.recipe_id) if next_spec is not None else None
        process: subprocess.Popen[str] | None = None
        lifecycle_info: dict[str, Any] = {
            "target_gpu": recipe.runtime.get("target_gpu"),
            "status": "not_started" if lifecycle else "disabled",
            "suite_dir": str(suite_dir),
            "result": str(suite_dir),
            "lifecycle_reuse_enabled": reuse_lifecycle,
        }
        pre_test_wrapper: dict[str, Any] | None = None
        runtime_skip_reason = recipe.runtime.get("skip_preflight_reason")
        if runtime_skip_reason:
            skip_reason = str(runtime_skip_reason)
            lifecycle_info.update(
                {
                    "status": "skipped_preflight",
                    "skip_reason": skip_reason,
                    "skip_detail": recipe.runtime.get("skip_preflight_detail"),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            result = _skipped_result(recipe, suite_dir, index, skip_reason, lifecycle_info)
            metrics = _write_metrics_with_lifecycle(result, lifecycle_info)
            recipe_runs.append(
                RecipeRunReport(
                    recipe_id=recipe.id,
                    recipe_name=recipe.name,
                    run_dir=result.run_dir,
                    metrics_path=result.metrics_path,
                    metrics=metrics,
                    lifecycle=lifecycle_info,
                )
            )
            continue
        if (
            unsloth_install_report
            and unsloth_install_report.get("benchmark_specs_skipped")
            and spec.backend == "unsloth-studio-or-compatible"
        ):
            skip_reason = str(unsloth_install_report.get("benchmark_specs_skip_reason") or "unsloth_preflight_unavailable")
            lifecycle_info.update(
                {
                    "status": "skipped_preflight",
                    "skip_reason": skip_reason,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            result = _skipped_result(recipe, suite_dir, index, skip_reason, lifecycle_info)
            metrics = _write_metrics_with_lifecycle(result, lifecycle_info)
            recipe_runs.append(
                RecipeRunReport(
                    recipe_id=recipe.id,
                    recipe_name=recipe.name,
                    run_dir=result.run_dir,
                    metrics_path=result.metrics_path,
                    metrics=metrics,
                    lifecycle=lifecycle_info,
                )
            )
            continue
        try:
            run_dir = suite_dir / _run_dir_name(index, recipe.id)
            pre_test_wrapper = _run_test_wrapper(
                recipe,
                suite_dir=suite_dir,
                run_dir=run_dir,
                index=index,
                wrapper_command=test_wrapper_command,
            )
            lifecycle_info["pre_test_wrapper"] = pre_test_wrapper
            if lifecycle:
                key = _lifecycle_key(recipe)
                if reuse_lifecycle and key is not None and key == active_key and active_process is not None and active_process.poll() is None:
                    process = active_process
                    lifecycle_info.update(
                        {
                            "status": "reused_loaded_model",
                            "lifecycle_key": key,
                            "reused_from_recipe_id": active_recipe.id if active_recipe is not None else None,
                            "command": active_lifecycle.get("command") if active_lifecycle else None,
                            "log_path": active_lifecycle.get("log_path") if active_lifecycle else None,
                            "ready_url": active_lifecycle.get("ready_url") if active_lifecycle else None,
                            "started_at": active_lifecycle.get("started_at") if active_lifecycle else None,
                        }
                    )
                else:
                    if active_process is not None and active_recipe is not None and active_lifecycle is not None:
                        _stop_lifecycle(active_recipe, active_process, active_lifecycle)
                    process, lifecycle_info = _start_lifecycle(recipe, suite_dir)
                    lifecycle_info.update(
                        {
                            "suite_dir": str(suite_dir),
                            "result": str(suite_dir),
                            "lifecycle_reuse_enabled": reuse_lifecycle,
                            "lifecycle_key": key,
                            "pre_test_wrapper": pre_test_wrapper,
                        }
                    )
                    active_process = process
                    active_key = key
                    active_recipe = recipe
                    active_lifecycle = lifecycle_info
            result = run_recipe(recipe.id, recipes_dir=recipes_dir, output_dir=suite_dir, run_id=f"{index:02d}")
            if lifecycle:
                if _should_keep_lifecycle(recipe, next_recipe, process, reuse_lifecycle=reuse_lifecycle):
                    lifecycle_info["kept_loaded_for_next_recipe"] = next_recipe.id if next_recipe is not None else None
                    lifecycle_info["finished_at"] = datetime.now(timezone.utc).isoformat()
                    active_process = process
                    active_key = _lifecycle_key(recipe)
                    active_recipe = recipe
                    active_lifecycle = lifecycle_info
                else:
                    lifecycle_info = _stop_lifecycle(recipe, process, lifecycle_info)
                    if process is active_process:
                        active_process = None
                        active_key = None
                        active_recipe = None
                        active_lifecycle = None
            metrics = _write_metrics_with_lifecycle(result, lifecycle_info)
        except Exception as exc:
            lifecycle_info["pre_test_wrapper"] = pre_test_wrapper or lifecycle_info.get("pre_test_wrapper")
            lifecycle_info["error"] = str(exc)
            try:
                if lifecycle:
                    lifecycle_info = _stop_lifecycle(recipe, process, lifecycle_info)
            except Exception as stop_exc:  # keep the original error and capture stop failure
                lifecycle_info["stop_error"] = str(stop_exc)
            if process is active_process:
                active_process = None
                active_key = None
                active_recipe = None
                active_lifecycle = None
            result = _failure_result(recipe, suite_dir, index, exc, lifecycle_info)
            metrics = _write_metrics_with_lifecycle(result, lifecycle_info)
            print(f"FAILED {recipe.id}: {exc}", file=sys.stderr)
        recipe_runs.append(
            RecipeRunReport(
                recipe_id=recipe.id,
                recipe_name=recipe.name,
                run_dir=result.run_dir,
                metrics_path=result.metrics_path,
                metrics=metrics,
                lifecycle=lifecycle_info,
            )
        )

    comparison = build_comparison_payload(recipe_runs, run_id=run_id)
    _add_unsloth_preflight_to_comparison(comparison, unsloth_install_report)
    comparison_json_path = suite_dir / "comparison.json"
    comparison_html_path = suite_dir / "comparison.html"
    comparison_json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    comparison_html_path.write_text(build_comparison_html(comparison), encoding="utf-8")
    return suite_dir, comparison_json_path, comparison_html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or plan the full Gemma 4 / MTP / DiffusionGemma matrix benchmark.")
    parser.add_argument("--recipes-dir", type=Path, default=DEFAULT_RECIPES_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--result", type=Path, help="Alias for --output-dir: root folder for the matrix plan, generated recipes, logs, and results.")
    parser.add_argument("--run-id", default="gemma-matrix")
    parser.add_argument("--dry-run", action="store_true", help="Generate recipes and a matrix plan but do not start model servers.")
    parser.add_argument("--no-lifecycle", action="store_true", help="Do not start/stop model servers; use already-running endpoints.")
    parser.add_argument("--no-lifecycle-reuse", action="store_true", help="Disable reuse when adjacent recipes have the same lifecycle_key/model_memory_key.")
    parser.add_argument("--orchestration-order", choices=["lifecycle", "matrix"], default=os.environ.get("ORCHESTRATION_ORDER", "lifecycle"), help="lifecycle groups by backend/model/mode to reduce load/unload churn; matrix keeps generation order.")
    parser.add_argument("--wrapper-command-file", "--test-wrapper-cmd", type=Path, default=Path(os.environ.get("TEST_WRAPPER_CMD", str(_default_wrapper_command_file()))), help="Wrapper file whose non-comment lines are executed before every test with $result and $test_result set. Defaults to scripts/wrap_each_test.cmd on Windows and scripts/wrap_each_test.sh on Linux.")
    parser.add_argument("--limit", type=int, help="Limit number of generated/running specs, useful for smoke tests.")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing model files in dry-run plans/tests.")
    parser.add_argument("--model", action="append", choices=["12b", "26b"], default=[])
    parser.add_argument("--mode", action="append", choices=["regular", "mtp", "diffusion", "diffusion-unsloth", "mtp-unsloth", "unsloth"], default=[])
    parser.add_argument("--context", action="append", type=int, default=[])
    parser.add_argument("--reasoning", action="append", choices=REASONING_MODES, default=[])
    parser.add_argument("--mtp-draft-n", action="append", type=int, default=[])
    parser.add_argument("--gpu-device", default=os.environ.get("GPU_DEVICE", "CUDA0"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"), help="CUDA_VISIBLE_DEVICES value injected into Unsloth Studio lifecycle commands; use nvidia-smi GPU indexes such as 0 for RTX A6000 or 2 for an A100 on this host.")
    parser.add_argument("--target-gpu", default=os.environ.get("TARGET_GPU", "GPU"))
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server"))
    parser.add_argument("--diffusion-server", default=os.environ.get("DIFFUSION_SERVER", "llama-diffusion-gemma-server"))
    parser.add_argument("--regular-port", type=int, default=int(os.environ.get("REGULAR_PORT", "18082")))
    parser.add_argument("--mtp-port", type=int, default=int(os.environ.get("MTP_PORT", "18083")))
    parser.add_argument("--diffusion-port", type=int, default=int(os.environ.get("DIFFUSION_PORT", "18081")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8192")))
    parser.add_argument("--ubatch-size", type=int, default=int(os.environ.get("UBATCH_SIZE", "2048")))
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--thinking-max-tokens", type=int, help="Optional max_tokens override only for reasoning=on recipes.")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--diffusion-timeout-seconds", type=int, default=300)
    parser.add_argument("--diffusion-steps", type=int, default=int(os.environ.get("DIFFUSION_STEPS", "8")))
    parser.add_argument("--diffusion-block-length", type=int, default=int(os.environ.get("DIFFUSION_BLOCK_LENGTH", "256")))
    parser.add_argument("--regular-extra-args", default=os.environ.get("REGULAR_EXTRA_ARGS"), help="Extra raw flags appended to regular llama-server commands.")
    parser.add_argument("--mtp-extra-args", default=os.environ.get("MTP_EXTRA_ARGS"), help="Extra raw flags appended to MTP llama-server commands.")
    parser.add_argument("--diffusion-extra-args", default=os.environ.get("DIFFUSION_EXTRA_ARGS"), help="Extra raw flags appended to llama-diffusion-gemma-server commands, e.g. KV cache type overrides.")
    parser.add_argument("--gemma12-model-gguf")
    parser.add_argument("--gemma12-mtp-draft-gguf")
    parser.add_argument("--gemma26-model-gguf")
    parser.add_argument("--gemma26-mtp-draft-gguf")
    parser.add_argument("--diffusion-gguf")
    parser.add_argument("--unsloth-base-url", help="OpenAI-compatible Unsloth endpoint for the full 72-spec Unsloth matrix. Defaults to UNSLOTH_BASE_URL, UNSLOTH_DIFFUSION_BASE_URL, then http://127.0.0.1:<unsloth-port>/v1.")
    parser.add_argument("--unsloth-port", type=int, default=int(os.environ.get("UNSLOTH_PORT", "18084")), help="Port used by auto-started Unsloth Studio lifecycle servers.")
    parser.add_argument("--unsloth-server", default=_default_unsloth_server(), help="Path to unsloth.exe/unsloth command used for auto-started Unsloth Studio lifecycle servers.")
    parser.add_argument("--unsloth-llama-server-path", default=os.environ.get("UNSLOTH_LLAMA_SERVER_PATH") or os.environ.get("LLAMA_SERVER_PATH"), help="Existing llama-server.exe path exported as LLAMA_SERVER_PATH for Unsloth Studio GGUF serving.")
    parser.add_argument("--dg-visual-bin", default=os.environ.get("DG_VISUAL_BIN"), help="Optional DiffusionGemma visual runner path exported as DG_VISUAL_BIN for Unsloth Studio.")
    parser.add_argument("--unsloth-dg-shim", default=os.environ.get("UNSLOTH_DG_SHIM"), help="Optional unsloth_zoo DiffusionGemma shim.py path exported as UNSLOTH_DG_SHIM for Unsloth Studio.")
    parser.add_argument("--unsloth-extra-args", default=os.environ.get("UNSLOTH_EXTRA_ARGS"), help="Extra raw flags appended to auto-started Unsloth Studio commands.")
    parser.add_argument("--unsloth-api-key", default=os.environ.get("UNSLOTH_API_KEY") or os.environ.get("UNSLOTH_E2E_API_KEY"), help="Bearer API key for Unsloth Studio. If omitted for a real run, the runner creates an internal key in the Unsloth Studio auth DB.")
    parser.add_argument("--unsloth-model-prefix", help="Prefix used to form model names for regular/MTP Unsloth matrix rows.")
    parser.add_argument("--unsloth-diffusion-base-url")
    parser.add_argument("--unsloth-diffusion-model")
    parser.add_argument("--install-unsloth-if-required", action="store_true", help="When diffusion-unsloth is selected, pip-install missing Unsloth/API packages into UNSLOTH_PYTHON or the current Python before running.")
    parser.add_argument("--force-unsloth-run-on-missing", action="store_true", help="Run diffusion-unsloth prompts even when local deps are missing and the configured endpoint is unreachable.")
    parser.add_argument("--unsloth-endpoint-timeout-seconds", type=float, default=float(os.environ.get("UNSLOTH_ENDPOINT_TIMEOUT_SECONDS", "2")), help="Timeout for the preflight /v1/models reachability probe before skipping unrunnable Unsloth specs.")
    parser.add_argument("--unsloth-install-package", action="append", default=[], help="Override/add a package spec for --install-unsloth-if-required. Repeatable; defaults install unsloth, torch, and API deps.")
    parser.add_argument("--unsloth-install-timeout-seconds", type=int, default=int(os.environ.get("UNSLOTH_INSTALL_TIMEOUT_SECONDS", "3600")))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.result is not None:
        args.output_dir = args.result
    args.output_dir = args.output_dir.expanduser()
    args.wrapper_command_file = args.wrapper_command_file.expanduser() if args.wrapper_command_file else None
    if not args.model:
        args.model = ["12b", "26b"]
    if not args.mode:
        args.mode = ["regular", "mtp", "diffusion"]
        if args.unsloth_diffusion_base_url or os.environ.get("UNSLOTH_DIFFUSION_BASE_URL"):
            args.mode.append("diffusion-unsloth")
    if not args.context:
        args.context = CONTEXTS
    if not args.reasoning:
        args.reasoning = REASONING_MODES
    if not args.mtp_draft_n:
        args.mtp_draft_n = MTP_DRAFT_N

    args.unsloth_install_report = _ensure_unsloth_if_requested(args)
    _drop_unrunnable_unsloth_specs(args, args.unsloth_install_report)
    if not args.dry_run:
        _ensure_unsloth_api_key(args, args.unsloth_install_report)
    wrapper_command = load_test_wrapper_command(args.wrapper_command_file)

    generated_recipes_dir = args.output_dir / f"{args.run_id}-matrix-recipes"
    specs = _build_matrix(args, generated_recipes_dir)
    plan_path = _write_plan(specs, args.output_dir, args.run_id, args)
    print(f"Matrix specs: {len(specs)}")
    print(f"Matrix plan: {plan_path}")
    print(f"Generated recipes: {generated_recipes_dir}")
    if args.dry_run:
        return 0
    if not specs:
        if args.unsloth_install_report and args.unsloth_install_report.get("benchmark_specs_skipped"):
            suite_dir, comparison_json_path, comparison_html_path = _run_matrix_suite(
                specs,
                generated_recipes_dir,
                args.output_dir,
                args.run_id,
                lifecycle=not args.no_lifecycle,
                test_wrapper_command=wrapper_command,
                reuse_lifecycle=not args.no_lifecycle_reuse,
                unsloth_install_report=args.unsloth_install_report,
            )
            print(f"Suite directory: {suite_dir}")
            print(f"Comparison JSON: {comparison_json_path}")
            print(f"Comparison HTML: {comparison_html_path}")
            return 0
        raise SystemExit("No matrix specs generated")
    suite_dir, comparison_json_path, comparison_html_path = _run_matrix_suite(
        specs,
        generated_recipes_dir,
        args.output_dir,
        args.run_id,
        lifecycle=not args.no_lifecycle,
        test_wrapper_command=wrapper_command,
        reuse_lifecycle=not args.no_lifecycle_reuse,
        unsloth_install_report=args.unsloth_install_report,
    )
    print(f"Suite directory: {suite_dir}")
    print(f"Comparison JSON: {comparison_json_path}")
    print(f"Comparison HTML: {comparison_html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
