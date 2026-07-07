from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_probe(command: list[str], timeout_seconds: float = 5.0) -> dict[str, Any]:
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    except FileNotFoundError as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc), "elapsed_ms": round((perf_counter() - started) * 1000.0, 3)}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nTimed out after {timeout_seconds}s",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }


def _parse_nvidia_smi(stdout: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        gpu: dict[str, Any] = {"name": parts[0]}
        if len(parts) > 1:
            try:
                gpu["memory_total_mb"] = int(float(parts[1]))
            except ValueError:
                gpu["memory_total_mb"] = parts[1]
        if len(parts) > 2 and parts[2]:
            gpu["driver_version"] = parts[2]
        gpus.append(gpu)
    return gpus


def detect_hardware() -> dict[str, Any]:
    nvidia = _run_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = _parse_nvidia_smi(str(nvidia.get("stdout") or "")) if nvidia["returncode"] == 0 else []
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "python": sys.executable,
        "gpus": gpus,
        "probes": {
            "nvidia-smi": {
                "ok": nvidia["returncode"] == 0,
                "returncode": nvidia["returncode"],
                "stderr": str(nvidia.get("stderr") or "")[:1000],
            }
        },
    }


def _model_args(model: str) -> list[str]:
    normalized = model.lower().replace("_", "-")
    if "26" in normalized:
        return ["--model", "26b"]
    if "12" in normalized or "9" in normalized:
        return ["--model", "12b"]
    return []


def _build_matrix_command(
    *,
    model: str,
    output_dir: str | Path,
    run_id: str | None,
    dry_run: bool,
    allow_missing: bool = True,
) -> list[str]:
    command = [sys.executable, "scripts/run_gemma_matrix.py"]
    if dry_run:
        command.append("--dry-run")
    if allow_missing:
        command.append("--allow-missing")
    command.extend(["--result", str(output_dir)])
    if run_id:
        command.extend(["--run-id", run_id])
    command.extend(_model_args(model))
    return command


def launch_hardware_benchmark(
    *,
    hardware: str = "auto",
    model: str = "gemma-4-12b",
    workflow: str = "agentic",
    output_dir: str | Path = "runs",
    run_id: str | None = None,
    dry_run: bool = False,
    optimize_for: str = "balanced",
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    hardware_snapshot = detect_hardware() if hardware == "auto" else {"label": hardware, "gpus": []}
    command = _build_matrix_command(model=model, output_dir=output_dir, run_id=run_id, dry_run=dry_run)
    benchmark: dict[str, Any] = {"command": command, "cwd": str(_repo_root())}
    status = "planned"
    if not dry_run:
        completed = subprocess.run(
            command,
            cwd=str(_repo_root()),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
        benchmark.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        status = "completed" if completed.returncode == 0 else "failed"
    return {
        "status": status,
        "hardware": hardware_snapshot,
        "benchmark": benchmark,
        "route_hint": {
            "workflow": workflow,
            "optimize_for": optimize_for,
            "next_step": "Run aireceipes route --comparison <comparison.json> after the benchmark completes.",
        },
    }


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> int:
    result = launch_hardware_benchmark(dry_run=True)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
