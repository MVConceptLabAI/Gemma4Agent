from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Iterable
from urllib import request
import json
import os
import re
import subprocess

from .catalog import Catalog, DEFAULT_RECIPES_DIR, Recipe
from .runner import RunResult, run_recipe


DEFAULT_GEMMA_BAKEOFF_RECIPES = [
    "inference.gemma4-regular-bakeoff",
    "inference.gemma4-mtp-bakeoff",
    "inference.diffusiongemma-bakeoff",
]

REPORT_METRICS = [
    "success_rate",
    "case_success_rate",
    "accuracy",
    "workflow_success_rate",
    "quality_loss",
    "loss_proxy",
    "ttft_ms_avg",
    "latency_ms_avg",
    "tokens_per_sec",
    "end_to_end_tokens_per_sec",
    "effective_tokens_per_sec",
    "completion_tokens_total",
    "output_tokens_avg",
]

METRIC_EXPLANATIONS = {
    "success_rate": "Successful API calls divided by total prompt attempts.",
    "case_success_rate": "Prompt attempts where every expected_contains / expected_regex / expected_not_contains check passed.",
    "accuracy": "Average scoring result over the prompt checks. A partially correct prompt can score between 0 and 1.",
    "workflow_success_rate": "Case success rate restricted to prompts tagged with vertical = agentic.",
    "quality_loss": "1 - accuracy, used when the endpoint does not expose a real loss value.",
    "loss_proxy": "Alias of quality_loss for models that do not return loss/eval_loss.",
    "ttft_ms_avg": "Average streaming time to first token, in milliseconds.",
    "latency_ms_avg": "Average end-to-end request latency, in milliseconds.",
    "tokens_per_sec": "Completion tokens per second over decode time only: completion_tokens / max(latency - TTFT, 0).",
    "end_to_end_tokens_per_sec": "Completion tokens per second over the full request latency.",
    "effective_tokens_per_sec": "Quality-adjusted throughput: end_to_end_tokens_per_sec * accuracy when checks exist, else * case/success rate.",
    "completion_tokens_total": "Total completion token count reported by the server, or estimated if usage is missing.",
    "output_tokens_avg": "Average completion tokens per successful prompt attempt.",
}

LOWER_IS_BETTER = {
    "quality_loss",
    "loss_proxy",
    "loss_avg",
    "eval_loss_avg",
    "ttft_ms_avg",
    "latency_ms_avg",
}


@dataclass(frozen=True)
class RecipeRunReport:
    recipe_id: str
    recipe_name: str
    run_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]
    lifecycle: dict[str, Any]


@dataclass(frozen=True)
class BakeoffSuiteResult:
    suite_dir: Path
    comparison_json_path: Path
    comparison_html_path: Path
    recipe_runs: list[RecipeRunReport]
    comparison: dict[str, Any]


def _safe_run_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "-" for character in value)


def _runtime_value(runtime: dict[str, Any], key: str) -> str | None:
    env_name = runtime.get(f"{key}_env")
    if env_name:
        value = os.environ.get(str(env_name))
        if value:
            return value
    direct = runtime.get(key)
    if direct not in (None, ""):
        return str(direct)
    return None


def _runtime_float(runtime: dict[str, Any], key: str, default: float) -> float:
    value = runtime.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _wait_for_ready(
    url: str,
    timeout_seconds: float,
    poll_seconds: float,
    process: subprocess.Popen[str] | None = None,
    headers: dict[str, str] | None = None,
    json_ready_key: str | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    attempts = 0
    last_error: str | None = None
    while perf_counter() - started <= timeout_seconds:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Process exited with {process.returncode} before readiness URL {url} became available: {last_error}"
            )
        attempts += 1
        try:
            http_request = request.Request(url, headers=headers or {})
            with request.urlopen(http_request, timeout=min(5.0, max(1.0, poll_seconds))) as response:
                payload_value: Any = None
                if json_ready_key:
                    body = response.read().decode("utf-8", errors="replace")
                    payload_value = json.loads(body)
                    for part in json_ready_key.split("."):
                        if isinstance(payload_value, dict):
                            payload_value = payload_value.get(part)
                        else:
                            payload_value = None
                            break
                    if not payload_value:
                        last_error = f"JSON readiness key {json_ready_key!r} is not truthy"
                        sleep(poll_seconds)
                        continue
                readiness = {
                    "ok": True,
                    "url": url,
                    "attempts": attempts,
                    "status": getattr(response, "status", None),
                    "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
                }
                if json_ready_key:
                    readiness["json_ready_key"] = json_ready_key
                    readiness["json_ready_value"] = payload_value
                return readiness
        except Exception as exc:  # readiness failures are expected while the model loads
            last_error = str(exc)
            sleep(poll_seconds)
    raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for readiness URL {url}: {last_error}")


def _run_shell_command(
    command: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            shell=True,
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
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nTimed out after {timeout_seconds}s",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "timeout_seconds": timeout_seconds,
        }


def _wait_for_shutdown(url: str | None, timeout_seconds: float, poll_seconds: float) -> dict[str, Any]:
    if not url:
        return {"ok": True, "status": "skipped_no_ready_url"}
    if timeout_seconds <= 0:
        return {"ok": True, "url": url, "status": "skipped_no_wait"}

    started = perf_counter()
    attempts = 0
    last_status: int | None = None
    while perf_counter() - started <= timeout_seconds:
        attempts += 1
        try:
            with request.urlopen(url, timeout=min(5.0, max(1.0, poll_seconds))) as response:
                last_status = getattr(response, "status", None)
        except Exception as exc:  # the desired state is that the endpoint is no longer reachable
            return {
                "ok": True,
                "url": url,
                "attempts": attempts,
                "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
                "last_error": str(exc),
            }
        sleep(poll_seconds)
    return {
        "ok": False,
        "url": url,
        "attempts": attempts,
        "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        "last_status": last_status,
        "error": f"Endpoint still responded after {timeout_seconds}s; model may still be resident.",
    }


def load_test_wrapper_command(command_file: str | Path | None) -> str | None:
    """Read a .cmd wrapper file and return the non-comment command to run before each test.

    The file is intentionally .cmd-friendly so users can edit/run it on Windows, but the runner
    treats it as a command template: blank lines, REM/@REM, ::, #, and @echo lines are ignored;
    remaining lines are joined with && and executed before every recipe/test. Put `bash -lc '...'`
    in the file when the command needs Bash features such as `$result`.
    """
    if command_file in (None, ""):
        return None
    path = Path(command_file)
    if not path.exists():
        return None
    command_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if line.startswith("#") or line.startswith("::"):
            continue
        if lower == "rem" or lower.startswith("rem ") or lower.startswith("@rem"):
            continue
        if lower.startswith("@echo") or lower.startswith("echo off"):
            continue
        command_lines.append(line)
    return " && ".join(command_lines) if command_lines else None


def _recipe_run_dir(suite_dir: Path, index: int, recipe_id: str) -> Path:
    return suite_dir / f"{index:02d}-{_safe_run_name(recipe_id)}"


def _run_test_wrapper(
    recipe: Recipe,
    *,
    suite_dir: Path,
    run_dir: Path,
    index: int,
    wrapper_command: str | None,
) -> dict[str, Any]:
    if not wrapper_command:
        return {"status": "skipped_no_wrapper_command"}

    log_path = suite_dir / f"{_safe_run_name(recipe.id)}-wrapper.log"
    env = os.environ.copy()
    env.update(
        {
            "result": str(suite_dir),
            "RESULT_DIR": str(suite_dir),
            "result_root": str(suite_dir.parent),
            "RESULT_ROOT": str(suite_dir.parent),
            "test_result": str(run_dir),
            "TEST_RESULT_DIR": str(run_dir),
            "recipe_id": recipe.id,
            "RECIPE_ID": recipe.id,
            "recipe_name": recipe.name,
            "RECIPE_NAME": recipe.name,
            "recipe_index": f"{index:02d}",
            "RECIPE_INDEX": f"{index:02d}",
            "base_url": str(recipe.runtime.get("base_url", "")),
            "BASE_URL": str(recipe.runtime.get("base_url", "")),
            "model": str(recipe.runtime.get("model", "")),
            "MODEL": str(recipe.runtime.get("model", "")),
        }
    )
    timeout_seconds = _runtime_float(recipe.runtime, "wrapper_timeout_seconds", 600.0)
    command_result = _run_shell_command(wrapper_command, cwd=suite_dir, env=env, timeout_seconds=timeout_seconds)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== wrapper before {recipe.id} ({index:02d}) ===\n")
        log_file.write(f"command: {wrapper_command}\n")
        log_file.write(f"result: {suite_dir}\n")
        log_file.write(f"test_result: {run_dir}\n")
        log_file.write(f"returncode: {command_result['returncode']}\n")
        if command_result.get("stdout"):
            log_file.write("--- stdout ---\n")
            log_file.write(str(command_result["stdout"]))
            if not str(command_result["stdout"]).endswith("\n"):
                log_file.write("\n")
        if command_result.get("stderr"):
            log_file.write("--- stderr ---\n")
            log_file.write(str(command_result["stderr"]))
            if not str(command_result["stderr"]).endswith("\n"):
                log_file.write("\n")
    info = {
        "status": "ok" if command_result["returncode"] == 0 else "failed",
        "command": wrapper_command,
        "log_path": str(log_path),
        "result": str(suite_dir),
        "test_result": str(run_dir),
        "command_result": command_result,
    }
    if command_result["returncode"] != 0:
        raise RuntimeError(f"Wrapper command failed for {recipe.id} with {command_result['returncode']}; see {log_path}")
    return info


def _lifecycle_key(recipe: Recipe) -> str | None:
    runtime = recipe.runtime
    start_command = _runtime_value(runtime, "start_command")
    if not start_command:
        return None
    explicit_key = runtime.get("lifecycle_key") or runtime.get("model_memory_key")
    if explicit_key not in (None, ""):
        return f"explicit:{explicit_key}"
    payload = {
        "start_command": start_command,
        "stop_command": _runtime_value(runtime, "stop_command"),
        "ready_url": _runtime_value(runtime, "ready_url"),
        "base_url": _runtime_value(runtime, "base_url"),
        "model": _runtime_value(runtime, "model"),
    }
    return json.dumps(payload, sort_keys=True)


def _should_keep_lifecycle(recipe: Recipe, next_recipe: Recipe | None, process: subprocess.Popen[str] | None, *, reuse_lifecycle: bool) -> bool:
    if not reuse_lifecycle or next_recipe is None or process is None or process.poll() is not None:
        return False
    key = _lifecycle_key(recipe)
    return key is not None and key == _lifecycle_key(next_recipe)


def _start_lifecycle(recipe: Recipe, suite_dir: Path) -> tuple[subprocess.Popen[str] | None, dict[str, Any]]:
    runtime = recipe.runtime
    target_gpu = str(runtime.get("target_gpu", "")) or None
    start_command = _runtime_value(runtime, "start_command")
    ready_url = _runtime_value(runtime, "ready_url")
    ready_timeout_seconds = _runtime_float(runtime, "ready_timeout_seconds", 900.0)
    ready_poll_seconds = _runtime_float(runtime, "ready_poll_seconds", 1.0)
    startup_grace_seconds = _runtime_float(runtime, "startup_grace_seconds", 0.2)

    lifecycle: dict[str, Any] = {
        "target_gpu": target_gpu,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "start_command_configured": bool(start_command),
        "ready_url": ready_url,
    }
    if not start_command:
        lifecycle["status"] = "skipped_no_start_command"
        return None, lifecycle

    log_path = suite_dir / f"{_safe_run_name(recipe.id)}-lifecycle.log"
    log_file = log_path.open("ab")
    process_env = os.environ.copy()
    process_env.update(
        {
            "result": str(suite_dir),
            "RESULT_DIR": str(suite_dir),
            "recipe_id": recipe.id,
            "RECIPE_ID": recipe.id,
            "recipe_name": recipe.name,
            "RECIPE_NAME": recipe.name,
        }
    )
    process = subprocess.Popen(
        start_command,
        cwd=str(suite_dir),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        shell=True,
        text=False,
        env=process_env,
    )
    setattr(process, "_aireceipes_log_file", log_file)
    lifecycle.update(
        {
            "status": "started",
            "pid": process.pid,
            "command": start_command,
            "log_path": str(log_path),
        }
    )

    try:
        returncode = process.wait(timeout=startup_grace_seconds)
        lifecycle["process_exited_during_startup"] = True
        lifecycle["startup_returncode"] = returncode
        if returncode != 0:
            if not getattr(log_file, "closed", True):
                log_file.close()
            raise RuntimeError(f"Start command for {recipe.id} exited with {returncode}; see {log_path}")
    except subprocess.TimeoutExpired:
        lifecycle["process_exited_during_startup"] = False

    try:
        if ready_url:
            ready_headers: dict[str, str] = {}
            ready_api_key = _runtime_value(runtime, "ready_api_key") or _runtime_value(runtime, "api_key")
            ready_json_key = _runtime_value(runtime, "ready_json_key")
            if ready_api_key:
                ready_headers["Authorization"] = f"Bearer {ready_api_key}"
            lifecycle["readiness"] = _wait_for_ready(
                ready_url,
                ready_timeout_seconds,
                ready_poll_seconds,
                process,
                headers=ready_headers,
                json_ready_key=ready_json_key,
            )
    except Exception as exc:
        lifecycle["startup_error"] = str(exc)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_file = getattr(process, "_aireceipes_log_file", None)
        if log_file is not None and not getattr(log_file, "closed", True):
            log_file.close()
        raise
    return process, lifecycle


def _stop_lifecycle(recipe: Recipe, process: subprocess.Popen[str] | None, lifecycle: dict[str, Any]) -> dict[str, Any]:
    runtime = recipe.runtime
    stop_command = _runtime_value(runtime, "stop_command")
    ready_url = _runtime_value(runtime, "ready_url")
    unload_wait_seconds = _runtime_float(runtime, "unload_wait_seconds", 5.0)
    unload_poll_seconds = _runtime_float(runtime, "unload_poll_seconds", 1.0)
    stop_info: dict[str, Any] = {"stopped_at": datetime.now(timezone.utc).isoformat()}
    process_env = os.environ.copy()
    result_dir = lifecycle.get("result") or lifecycle.get("suite_dir")
    if isinstance(result_dir, str) and result_dir:
        process_env["result"] = result_dir
        process_env["RESULT_DIR"] = result_dir
    process_env.update({"recipe_id": recipe.id, "RECIPE_ID": recipe.id, "recipe_name": recipe.name, "RECIPE_NAME": recipe.name})

    try:
        if stop_command:
            command_result = _run_shell_command(stop_command, env=process_env, timeout_seconds=unload_wait_seconds)
            stop_info["stop_command"] = command_result
            if command_result["returncode"] != 0:
                stop_info["status"] = "stop_command_failed"
                lifecycle["stop"] = stop_info
                return lifecycle

        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=unload_wait_seconds)
                stop_info["status"] = "terminated"
                stop_info["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=unload_wait_seconds)
                stop_info["status"] = "killed"
                stop_info["returncode"] = process.returncode
        elif process is not None:
            stop_info["status"] = "already_exited"
            stop_info["returncode"] = process.returncode
        else:
            stop_info["status"] = "no_process"

        stop_info["shutdown_check"] = _wait_for_shutdown(ready_url, unload_wait_seconds, unload_poll_seconds)
    finally:
        log_file = getattr(process, "_aireceipes_log_file", None) if process is not None else None
        if log_file is not None and not getattr(log_file, "closed", True):
            log_file.close()

    lifecycle["stop"] = stop_info
    lifecycle["finished_at"] = datetime.now(timezone.utc).isoformat()
    return lifecycle


def _write_metrics_with_lifecycle(result: RunResult, lifecycle: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(result.metrics)
    metrics["lifecycle"] = lifecycle
    result.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _numeric_metric(metrics: dict[str, Any], metric_name: str) -> float | None:
    value = metrics.get(metric_name)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _best_by_metric(models: list[dict[str, Any]], metric_names: Iterable[str]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for metric_name in metric_names:
        candidates = [model for model in models if _numeric_metric(model["kpis"], metric_name) is not None]
        if not candidates:
            continue
        reverse = metric_name not in LOWER_IS_BETTER
        chosen = sorted(candidates, key=lambda model: float(model["kpis"][metric_name]), reverse=reverse)[0]
        best[metric_name] = {
            "recipe_id": chosen["recipe_id"],
            "value": chosen["kpis"][metric_name],
            "direction": "min" if metric_name in LOWER_IS_BETTER else "max",
        }
    return best


def _mean_float(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _model_file_paths(command: str | None) -> list[str]:
    if not command:
        return []
    paths: list[str] = []
    pattern = re.compile(r"(?:\"([^\"]+?\.gguf)\"|'([^']+?\.gguf)'|(\S+?\.gguf))", re.IGNORECASE)
    for match in pattern.finditer(command):
        path = next((group for group in match.groups() if group), None)
        if path and path not in paths:
            paths.append(path)
    return paths


def _file_name(path: str | None) -> str | None:
    if not path:
        return None
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _response_models(metrics: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for result in metrics.get("results", []):
        if not isinstance(result, dict):
            continue
        model = result.get("model")
        if isinstance(model, str) and model and model not in models:
            models.append(model)
    return models


def _model_details(recipe_run: RecipeRunReport) -> dict[str, Any]:
    metrics = recipe_run.metrics
    lifecycle = recipe_run.lifecycle
    command = lifecycle.get("command") if isinstance(lifecycle, dict) else None
    model_files = _model_file_paths(command if isinstance(command, str) else None)
    response_models = _response_models(metrics)
    primary_model_file = model_files[0] if model_files else None
    draft_model_file = model_files[1] if len(model_files) > 1 else None
    primary_version = _file_name(primary_model_file) or (response_models[0] if response_models else None)
    draft_version = _file_name(draft_model_file)
    version_parts = [part for part in (primary_version, f"draft={draft_version}" if draft_version else None) if part]
    model_version = " + ".join(version_parts) if version_parts else None
    display_name = f"{recipe_run.recipe_name} ({model_version})" if model_version else recipe_run.recipe_name
    return {
        "display_name": display_name,
        "model_version": model_version,
        "primary_model_file": primary_model_file,
        "primary_model_filename": _file_name(primary_model_file),
        "draft_model_file": draft_model_file,
        "draft_model_filename": draft_version,
        "model_files": model_files,
        "response_models": response_models,
    }


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _check_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        for detail in result.get("check_details", []) if isinstance(result.get("check_details"), list) else []:
            if not isinstance(detail, dict):
                continue
            key = (str(detail.get("type", "check")), str(detail.get("value", "")))
            item = checks.setdefault(
                key,
                {"type": key[0], "value": key[1], "passed": 0, "total": 0, "pass_rate": 0.0},
            )
            item["total"] += 1
            if detail.get("passed"):
                item["passed"] += 1
    for item in checks.values():
        item["pass_rate"] = item["passed"] / item["total"] if item["total"] else 0.0
    return list(checks.values())


def _prompt_evaluations(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: dict[str, int] = {}
    for result in metrics.get("results", []):
        if not isinstance(result, dict):
            continue
        key = str(result.get("case_id", result.get("prompt_index", "prompt")))
        grouped.setdefault(key, []).append(result)
        order.setdefault(key, _as_int(result.get("prompt_index")))

    evaluations: list[dict[str, Any]] = []
    for case_id, results in sorted(grouped.items(), key=lambda item: (order[item[0]], item[0])):
        attempts = len(results)
        success_count = sum(1 for result in results if result.get("ok"))
        case_success_count = sum(1 for result in results if result.get("case_success"))
        total_latency_ms = sum(_as_float(result.get("latency_ms")) or 0.0 for result in results)
        total_completion_tokens = sum(_as_int(result.get("completion_tokens")) for result in results if result.get("ok"))
        scores = [score for result in results if (score := _as_float(result.get("score"))) is not None]
        ttft_values = [ttft for result in results if (ttft := _as_float(result.get("ttft_ms"))) is not None]
        checks_total = sum(_as_int(result.get("checks_total")) for result in results)
        checks_passed = sum(_as_int(result.get("checks_passed")) for result in results)
        accuracy = _mean_float(scores) if scores else (success_count / attempts if attempts else 0.0)
        quality_factor = accuracy if checks_total else (success_count / attempts if attempts else 0.0)
        end_to_end_tokens_per_sec = (
            total_completion_tokens * 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0
        )
        decode_latency_ms = 0.0
        for result in results:
            if not result.get("ok"):
                continue
            latency = _as_float(result.get("latency_ms")) or 0.0
            ttft = _as_float(result.get("ttft_ms"))
            decode_latency_ms += max(latency - ttft, 0.0) if ttft is not None else latency
        tokens_per_sec = total_completion_tokens * 1000.0 / decode_latency_ms if decode_latency_ms > 0 else 0.0
        sample_output = next((str(result.get("output", "")) for result in results if result.get("ok")), "")
        first_result = results[0]
        evaluations.append(
            {
                "case_id": case_id,
                "vertical": first_result.get("vertical"),
                "prompt": first_result.get("prompt"),
                "attempts": attempts,
                "success_count": success_count,
                "case_success_rate": case_success_count / attempts if attempts else 0.0,
                "accuracy": accuracy,
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "checks": _check_summaries(results),
                "latency_ms_avg": total_latency_ms / attempts if attempts else 0.0,
                "ttft_ms_avg": _mean_float(ttft_values),
                "completion_tokens_total": total_completion_tokens,
                "end_to_end_tokens_per_sec": end_to_end_tokens_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "quality_factor": quality_factor,
                "effective_tokens_per_sec": end_to_end_tokens_per_sec * quality_factor,
                "effective_tokens_per_sec_formula": {
                    "formula": "(completion_tokens_total * 1000 / latency_ms_total) * quality_factor",
                    "completion_tokens_total": total_completion_tokens,
                    "latency_ms_total": round(total_latency_ms, 3),
                    "quality_factor": quality_factor,
                },
                "sample_output": sample_output,
            }
        )
    return evaluations


def build_comparison_payload(recipe_runs: list[RecipeRunReport], *, run_id: str) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for index, recipe_run in enumerate(recipe_runs, start=1):
        metrics = recipe_run.metrics
        kpis = metrics.get("kpis", {}) if isinstance(metrics.get("kpis"), dict) else {}
        details = _model_details(recipe_run)
        classification = metrics.get("classification", {}) if isinstance(metrics.get("classification"), dict) else {}
        primary_model_filename = details["primary_model_filename"] or classification.get("primary_model_filename")
        draft_model_filename = details["draft_model_filename"] or classification.get("draft_model_filename")
        models.append(
            {
                "order": index,
                "recipe_id": recipe_run.recipe_id,
                "name": details["display_name"],
                "base_name": recipe_run.recipe_name,
                "model_version": details["model_version"],
                "primary_model_file": details["primary_model_file"],
                "primary_model_filename": primary_model_filename,
                "draft_model_file": details["draft_model_file"],
                "draft_model_filename": draft_model_filename,
                "model_files": details["model_files"],
                "response_models": details["response_models"],
                "classification": classification,
                "variant": classification.get("variant"),
                "mode": classification.get("mode"),
                "reasoning": classification.get("reasoning"),
                "context_size": classification.get("context_size"),
                "mtp_draft_n": classification.get("mtp_draft_n"),
                "backend": classification.get("backend"),
                "model_family": classification.get("model_family"),
                "target_gpu": recipe_run.lifecycle.get("target_gpu"),
                "metrics_path": str(recipe_run.metrics_path),
                "kpis": {metric: kpis.get(metric) for metric in REPORT_METRICS},
                "verticals": metrics.get("verticals", {}),
                "prompt_evaluations": _prompt_evaluations(metrics),
                "success_count": metrics.get("success_count"),
                "error_count": metrics.get("error_count"),
                "skipped_count": metrics.get("skipped_count"),
                "skip_reason": metrics.get("skip_reason") or recipe_run.lifecycle.get("skip_reason"),
                "skip_detail": metrics.get("skip_detail") or recipe_run.lifecycle.get("skip_detail"),
                "prompt_count": metrics.get("prompt_count"),
                "lifecycle": recipe_run.lifecycle,
            }
        )

    return {
        "title": "Gemma Variant Bakeoff",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(models),
        "metrics": REPORT_METRICS,
        "metric_explanations": {metric: METRIC_EXPLANATIONS.get(metric, "") for metric in REPORT_METRICS},
        "prompt_evaluation_note": "Per-prompt effective_tokens_per_sec is computed as (completion tokens / full request latency) multiplied by the prompt accuracy/quality factor. This penalizes fast outputs that fail benchmark checks.",
        "models": models,
        "best_by_metric": _best_by_metric(models, REPORT_METRICS),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _truncate_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\r\n", "\n")
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _format_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "—"
    if value in (None, ""):
        return "—"
    return str(value)


def _preflight_sections(comparison: dict[str, Any]) -> str:
    checks = comparison.get("preflight_checks")
    if not isinstance(checks, list) or not checks:
        return ""
    rows: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        install_command = check.get("install_command")
        if isinstance(install_command, list):
            install_display = " ".join(str(part) for part in install_command)
        else:
            install_display = _format_list(install_command)
        detail_bits = []
        message = check.get("message")
        if message:
            detail_bits.append(f"<div>{escape(str(message))}</div>")
        if check.get("python"):
            detail_bits.append(f"<div>Python: <code>{escape(str(check['python']))}</code></div>")
        if install_display != "—":
            detail_bits.append(f"<div>Install command: <code>{escape(install_display)}</code></div>")
        rows.append(
            "<tr>"
            f"<td><strong>{escape(str(check.get('name') or 'preflight'))}</strong></td>"
            f"<td><code>{escape(str(check.get('backend') or '—'))}</code></td>"
            f"<td><span class=\"badge warn\">{escape(str(check.get('status') or 'unknown'))}</span></td>"
            f"<td>{escape(_format_list(check.get('checked_modules')))}</td>"
            f"<td>{escape(_format_list(check.get('missing_modules')))}</td>"
            f"<td>{escape(str(check.get('install_requested', '—')))}</td>"
            f"<td>{''.join(detail_bits) or '—'}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<section class=\"panel\">"
        "<h2>Backend preflight checks</h2>"
        "<table><thead><tr><th>Check</th><th>Backend</th><th>Status</th><th>Checked modules</th>"
        "<th>Missing modules</th><th>Install requested</th><th>Details</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )


def build_comparison_html(comparison: dict[str, Any]) -> str:
    metrics = comparison["metrics"]
    models = comparison["models"]
    header_cells = "".join(f"<th class=\"num metric\">{escape(metric)}</th>" for metric in metrics)
    identity_header_cells = "".join(
        [
            "<th class=\"order-col\">#</th>",
            "<th class=\"recipe-col\">Recipe</th>",
            "<th>Model family</th>",
            "<th>Mode</th>",
            "<th class=\"num\">Ctx</th>",
            "<th>Think</th>",
            "<th class=\"num\">MTP n</th>",
            "<th>Backend</th>",
            "<th class=\"file-col\">Primary GGUF</th>",
            "<th class=\"file-col\">Draft GGUF</th>",
            "<th class=\"file-col\">Response model</th>",
            "<th>GPU</th>",
            "<th class=\"num\">Success</th>",
        ]
    )
    rows = []
    for model in models:
        kpis = model.get("kpis", {}) if isinstance(model.get("kpis"), dict) else {}
        metric_cells = "".join(
            f"<td class=\"num\" data-metric=\"{escape(metric)}\">{escape(_format_value(kpis.get(metric)))}</td>"
            for metric in metrics
        )
        response_models = ", ".join(str(item) for item in model.get("response_models", [])) or "—"
        context_size = model.get("context_size")
        try:
            context_display = f"{int(context_size) // 1024}k" if context_size is not None else "—"
        except (TypeError, ValueError):
            context_display = str(context_size or "—")
        recipe_name = str(model.get("base_name") or model.get("name") or model["recipe_id"])
        replacement_badge = ""
        if model.get("replacement_source"):
            replacement_badge = "<br><span class=\"badge warn\">fallback row</span>"
        lifecycle = model.get("lifecycle") if isinstance(model.get("lifecycle"), dict) else {}
        if lifecycle.get("status") == "skipped_preflight":
            skip_reason = model.get("skip_reason") or lifecycle.get("skip_reason") or "skipped_preflight"
            skip_detail = model.get("skip_detail") or lifecycle.get("skip_detail")
            replacement_badge += f"<br><span class=\"badge warn\">skipped preflight: {escape(str(skip_reason))}</span>"
            if skip_detail:
                replacement_badge += f"<br><small>{escape(str(skip_detail))}</small>"
        rows.append(
            "<tr>"
            f"<td class=\"num order-col\">{model['order']}</td>"
            f"<td class=\"recipe-col\"><strong>{escape(recipe_name)}</strong><br>"
            f"<code>{escape(model['recipe_id'])}</code>{replacement_badge}</td>"
            f"<td><code>{escape(str(model.get('model_family') or '—'))}</code></td>"
            f"<td><span class=\"pill\">{escape(str(model.get('mode') or '—'))}</span></td>"
            f"<td class=\"num\">{escape(context_display)}</td>"
            f"<td>{escape(str(model.get('reasoning') or '—'))}</td>"
            f"<td class=\"num\">{escape(str(model.get('mtp_draft_n') or '—'))}</td>"
            f"<td><code>{escape(str(model.get('backend') or '—'))}</code></td>"
            f"<td class=\"file-col\"><code>{escape(str(model.get('primary_model_filename') or '—'))}</code></td>"
            f"<td class=\"file-col\"><code>{escape(str(model.get('draft_model_filename') or '—'))}</code></td>"
            f"<td class=\"file-col\"><code>{escape(response_models)}</code></td>"
            f"<td>{escape(str(model.get('target_gpu') or '—'))}</td>"
            f"<td class=\"num\">{escape(str(model.get('success_count')))} / {escape(str(model.get('prompt_count')))}</td>"
            f"{metric_cells}"
            "</tr>"
        )

    best_items = []
    for metric, best in comparison.get("best_by_metric", {}).items():
        best_items.append(
            f"<li><strong>{escape(metric)}</strong>: {escape(best['recipe_id'])} "
            f"({escape(_format_value(best['value']))}, {escape(best['direction'])})</li>"
        )

    metric_items = []
    explanations = comparison.get("metric_explanations", {})
    for metric in metrics:
        metric_items.append(f"<li><code>{escape(metric)}</code> — {escape(str(explanations.get(metric, '')))}</li>")

    vertical_sections = []
    for model in models:
        vertical_rows = []
        verticals = model.get("verticals", {}) if isinstance(model.get("verticals"), dict) else {}
        for vertical, values in sorted(verticals.items()):
            vertical_rows.append(
                "<tr>"
                f"<td>{escape(str(vertical))}</td>"
                f"<td>{escape(_format_value(values.get('accuracy')))}</td>"
                f"<td>{escape(_format_value(values.get('case_success_rate')))}</td>"
                f"<td>{escape(_format_value(values.get('effective_tokens_per_sec')))}</td>"
                f"<td>{escape(_format_value(values.get('end_to_end_tokens_per_sec')))}</td>"
                f"<td>{escape(_format_value(values.get('tokens_per_sec')))}</td>"
                f"<td>{escape(_format_value(values.get('latency_ms_avg')))}</td>"
                "</tr>"
            )
        vertical_sections.append(
            f"<section><h3>{escape(str(model.get('name') or model['recipe_id']))}</h3>"
            "<table><thead><tr><th>Vertical</th><th>Accuracy</th><th>Case success</th>"
            "<th>Effective tok/s</th><th>E2E tok/s</th><th>Decode tok/s</th><th>Latency ms</th></tr></thead>"
            f"<tbody>{''.join(vertical_rows)}</tbody></table></section>"
        )

    prompt_rows = []
    for model in models:
        for prompt_eval in model.get("prompt_evaluations", []):
            check_bits = []
            for check in prompt_eval.get("checks", []):
                check_bits.append(
                    f"<li>{escape(str(check.get('type')))} <code>{escape(_truncate_text(check.get('value'), 80))}</code>: "
                    f"{escape(str(check.get('passed')))} / {escape(str(check.get('total')))}</li>"
                )
            prompt_rows.append(
                "<tr>"
                f"<td><strong>{escape(str(model.get('name') or model['recipe_id']))}</strong><br>"
                f"<span>{escape(model['recipe_id'])}</span></td>"
                f"<td><strong>{escape(str(prompt_eval.get('case_id')))}</strong><br>"
                f"<span>{escape(str(prompt_eval.get('vertical') or ''))}</span><br>"
                f"<div class=\"prompt\">{escape(_truncate_text(prompt_eval.get('prompt'), 260))}</div></td>"
                f"<td>{escape(_format_value(prompt_eval.get('accuracy')))}<br>"
                f"<span>{escape(str(prompt_eval.get('checks_passed')))} / {escape(str(prompt_eval.get('checks_total')))} checks</span></td>"
                f"<td>{escape(_format_value(prompt_eval.get('case_success_rate')))}</td>"
                f"<td>{escape(_format_value(prompt_eval.get('latency_ms_avg')))}</td>"
                f"<td>{escape(_format_value(prompt_eval.get('ttft_ms_avg')))}</td>"
                f"<td>{escape(_format_value(prompt_eval.get('end_to_end_tokens_per_sec')))}</td>"
                f"<td>{escape(_format_value(prompt_eval.get('quality_factor')))}</td>"
                f"<td><strong>{escape(_format_value(prompt_eval.get('effective_tokens_per_sec')))}</strong><br>"
                f"<span>{escape(str(prompt_eval.get('completion_tokens_total')))} tokens total</span></td>"
                f"<td><ul class=\"compact\">{''.join(check_bits)}</ul>"
                f"<details><summary>Sample output</summary><pre>{escape(_truncate_text(prompt_eval.get('sample_output'), 500))}</pre></details></td>"
                "</tr>"
            )

    comparison_count = comparison.get("model_count", len(models))

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(comparison['title'])}</title>
  <style>
    :root {{ color-scheme: dark; --bg: #0f172a; --panel: #111827; --text: #e5e7eb; --muted: #9ca3af; --accent: #38bdf8; --line: #334155; --good: #22c55e; --warn: #f59e0b; }}
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 32px; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .meta, span {{ color: var(--muted); }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px; margin: 20px 0; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1100px; }}
    table.comparison {{ width: max-content; min-width: 2600px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; vertical-align: top; }}
    .comparison th, .comparison td {{ font-size: 13px; }}
    th {{ color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    td.num, th.num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .order-col {{ width: 42px; }}
    .recipe-col {{ min-width: 260px; max-width: 360px; }}
    .file-col {{ min-width: 180px; max-width: 280px; overflow-wrap: anywhere; word-break: break-word; }}
    .metric {{ white-space: nowrap; }}
    .pill, .badge {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; background: #020617; color: var(--text); font-size: 12px; }}
    .badge.warn {{ color: var(--warn); border-color: var(--warn); margin-top: 6px; }}
    ul {{ columns: 2; }}
    ul.compact {{ columns: 1; margin: 6px 0 0; padding-left: 18px; }}
    code {{ color: var(--accent); }}
    pre {{ white-space: pre-wrap; background: #020617; border: 1px solid var(--line); border-radius: 10px; padding: 12px; max-width: 520px; }}
    .prompt {{ margin-top: 8px; max-width: 460px; }}
    .metric-note {{ line-height: 1.5; }}
  </style>
</head>
<body>
<main>
  <h1>{escape(comparison['title'])}</h1>
  <p class=\"meta\">Run ID: <code>{escape(comparison['run_id'])}</code> · Generated: {escape(comparison['generated_at'])}</p>
  <section class=\"panel\">
    <h2>{escape(str(comparison_count))}-recipe comparison with split model metadata</h2>
    <table class=\"comparison\">
      <thead><tr>{identity_header_cells}{header_cells}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
  {_preflight_sections(comparison)}
  <section class=\"panel metric-note\">
    <h2>Metric definitions</h2>
    <p>{escape(str(comparison.get('prompt_evaluation_note') or ''))}</p>
    <ul>{''.join(metric_items)}</ul>
  </section>
  <section class=\"panel\">
    <h2>Best by metric</h2>
    <ul>{''.join(best_items)}</ul>
  </section>
  <section class=\"panel\">
    <h2>Prompt-level evaluation and effective token/s</h2>
    <table>
      <thead><tr><th>Model</th><th>Prompt / vertical</th><th>Accuracy</th><th>Case success</th><th>Latency ms</th><th>TTFT ms</th><th>E2E tok/s</th><th>Quality factor</th><th>Effective tok/s</th><th>Checks / sample</th></tr></thead>
      <tbody>{''.join(prompt_rows)}</tbody>
    </table>
  </section>
  <section class=\"panel\">
    <h2>Vertical breakdown</h2>
    {''.join(vertical_sections)}
  </section>
</main>
</body>
</html>
"""


def run_bakeoff_suite(
    recipe_ids: list[str] | tuple[str, ...] | None = None,
    *,
    recipes_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    lifecycle: bool = True,
    test_wrapper_command: str | None = None,
    reuse_lifecycle: bool = True,
) -> BakeoffSuiteResult:
    catalog = Catalog(recipes_dir or DEFAULT_RECIPES_DIR)
    selected_recipe_ids = list(recipe_ids or DEFAULT_GEMMA_BAKEOFF_RECIPES)
    actual_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_output_dir = Path(output_dir) if output_dir is not None else Path("runs")
    suite_dir = base_output_dir / f"{actual_run_id}-gemma-variant-bakeoff"
    suite_dir.mkdir(parents=True, exist_ok=True)

    recipe_runs: list[RecipeRunReport] = []
    active_process: subprocess.Popen[str] | None = None
    active_key: str | None = None
    active_recipe: Recipe | None = None
    active_lifecycle: dict[str, Any] | None = None

    for index, recipe_id in enumerate(selected_recipe_ids, start=1):
        recipe = catalog.get(recipe_id)
        next_recipe = catalog.get(selected_recipe_ids[index]) if index < len(selected_recipe_ids) else None
        process: subprocess.Popen[str] | None = None
        lifecycle_info: dict[str, Any] = {
            "target_gpu": recipe.runtime.get("target_gpu"),
            "status": "disabled",
            "suite_dir": str(suite_dir),
            "result": str(suite_dir),
            "lifecycle_reuse_enabled": reuse_lifecycle,
        }
        try:
            run_dir = _recipe_run_dir(suite_dir, index, recipe.id)
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
            result = run_recipe(
                recipe_id,
                recipes_dir=catalog.recipes_dir,
                output_dir=suite_dir,
                run_id=f"{index:02d}",
            )
        finally:
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

    comparison = build_comparison_payload(recipe_runs, run_id=actual_run_id)
    comparison_json_path = suite_dir / "comparison.json"
    comparison_html_path = suite_dir / "comparison.html"
    comparison_json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    comparison_html_path.write_text(build_comparison_html(comparison), encoding="utf-8")
    return BakeoffSuiteResult(
        suite_dir=suite_dir,
        comparison_json_path=comparison_json_path,
        comparison_html_path=comparison_html_path,
        recipe_runs=recipe_runs,
        comparison=comparison,
    )
