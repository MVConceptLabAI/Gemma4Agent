from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import os
import re
from urllib import error, request

from .catalog import Catalog, DEFAULT_RECIPES_DIR


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]


def _safe_run_name(recipe_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", recipe_id)


def _echo_completion(prompt: str) -> str:
    return f"Echo: {prompt}"


def _runtime_value(runtime: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    env_name = runtime.get(f"{key}_env")
    if env_name:
        env_value = os.environ.get(str(env_name))
        if env_value:
            return env_value

    direct_value = runtime.get(key)
    if direct_value not in (None, ""):
        return str(direct_value)

    if required:
        raise ValueError(f"runtime.{key} or runtime.{key}_env is required")
    return None


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _build_messages(prompt: str, runtime: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_prompt = runtime.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": prompt})
    return messages


def _chat_payload(model: str, messages: list[dict[str, str]], parameters: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    for key in (
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "repeat_penalty",
        "reasoning",
        "reasoning_budget",
        "chat_template_kwargs",
        "logprobs",
        "top_logprobs",
    ):
        if key in parameters:
            payload[key] = parameters[key]

    stream_options = parameters.get("stream_options")
    if isinstance(stream_options, dict):
        payload["stream_options"] = stream_options
    elif stream and _as_bool(parameters.get("stream_include_usage"), default=False):
        payload["stream_options"] = {"include_usage": True}
    return payload


def _request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_error_with_body(exc: error.HTTPError) -> RuntimeError:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    message = str(exc)
    if body:
        message = f"{message}; response body: {body[:4000]}"
    return RuntimeError(message)


def _stream_chat_completion(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout_seconds: float,
    fallback_model: str,
) -> dict[str, Any]:
    started = perf_counter()
    http_request = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    response_model = fallback_model
    first_token_at_ms: float | None = None
    stream_chunk_count = 0
    try:
        response_context = request.urlopen(http_request, timeout=timeout_seconds)
    except error.HTTPError as exc:
        raise _http_error_with_body(exc) from exc
    with response_context as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if isinstance(chunk.get("model"), str):
                response_model = chunk["model"]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            first_choice = choices[0]
            finish_reason = first_choice.get("finish_reason") or finish_reason
            delta = first_choice.get("delta") or {}
            content = delta.get("content") if isinstance(delta, dict) else None
            reasoning_content = delta.get("reasoning_content") if isinstance(delta, dict) else None
            if isinstance(reasoning_content, str) and reasoning_content:
                if first_token_at_ms is None:
                    first_token_at_ms = (perf_counter() - started) * 1000.0
                stream_chunk_count += 1
                reasoning_chunks.append(reasoning_content)
            if content is None:
                content = first_choice.get("text")
            if not isinstance(content, str) or content == "":
                continue

            if first_token_at_ms is None:
                first_token_at_ms = (perf_counter() - started) * 1000.0
            stream_chunk_count += 1
            chunks.append(content)

    result: dict[str, Any] = {
        "output": "".join(chunks),
        "model": response_model,
        "stream": True,
        "stream_chunks": stream_chunk_count,
        "generation_ms": round((perf_counter() - started) * 1000.0, 3),
    }
    if first_token_at_ms is not None:
        result["ttft_ms"] = round(first_token_at_ms, 3)
    if reasoning_chunks:
        result["reasoning_content"] = "".join(reasoning_chunks)
        result["reasoning_chars"] = len(result["reasoning_content"])
    if usage is not None:
        result["usage"] = usage
    if finish_reason is not None:
        result["finish_reason"] = finish_reason
    return result


def _openai_compatible_chat(prompt: str, runtime: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    base_url = _runtime_value(runtime, "base_url")
    model = _runtime_value(runtime, "model")
    api_key = _runtime_value(runtime, "api_key", required=False)
    assert base_url is not None  # for type checkers; _runtime_value enforces required=True
    assert model is not None

    stream = _as_bool(parameters.get("stream", runtime.get("stream")), default=False)
    messages = _build_messages(prompt, runtime)
    payload = _chat_payload(model, messages, parameters, stream=stream)
    headers = _request_headers(api_key)
    timeout_seconds = float(runtime.get("timeout_seconds", parameters.get("timeout_seconds", 60)))
    url = _chat_completions_url(base_url)
    if stream:
        return _stream_chat_completion(url, payload, headers, timeout_seconds=timeout_seconds, fallback_model=model)

    http_request = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        response_context = request.urlopen(http_request, timeout=timeout_seconds)
    except error.HTTPError as exc:
        raise _http_error_with_body(exc) from exc
    with response_context as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    choices = response_payload.get("choices", [])
    if not choices:
        raise ValueError("OpenAI-compatible response did not contain choices")
    first_choice = choices[0]
    message = first_choice.get("message") or {}
    output = message.get("content") if isinstance(message, dict) and "content" in message else first_choice.get("text")
    if not isinstance(output, str):
        raise ValueError("OpenAI-compatible response did not contain text content")

    result: dict[str, Any] = {
        "output": output,
        "model": response_payload.get("model", model),
        "stream": False,
    }
    if "usage" in response_payload:
        result["usage"] = response_payload["usage"]
    reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None
    if isinstance(reasoning_content, str) and reasoning_content:
        result["reasoning_content"] = reasoning_content
        result["reasoning_chars"] = len(reasoning_content)
    if "timings" in response_payload:
        result["timings"] = response_payload["timings"]
    for metric_key in ("loss", "eval_loss"):
        if metric_key in response_payload:
            result[metric_key] = response_payload[metric_key]
    if "finish_reason" in first_choice:
        result["finish_reason"] = first_choice["finish_reason"]
    return result


def _execute_prompt(adapter: str, prompt: str, runtime: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    if adapter == "echo":
        return {"output": _echo_completion(prompt)}
    if adapter == "openai_compatible_chat":
        return _openai_compatible_chat(prompt, runtime, parameters)
    raise ValueError(
        f"Unsupported runtime adapter '{adapter}'. Supported adapters: echo, openai_compatible_chat."
    )


def _expected_values(case: dict[str, Any], key: str) -> list[str]:
    value = case.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _score_output(output: str, case: dict[str, Any]) -> dict[str, Any]:
    case_sensitive = _as_bool(case.get("case_sensitive"), default=False)
    strip_output = _as_bool(case.get("strip_output"), default=True)
    text = output.strip() if strip_output else output
    comparable_text = text if case_sensitive else text.lower()
    details: list[dict[str, Any]] = []

    for expected in _expected_values(case, "expected_exact"):
        expected_text = expected.strip() if strip_output else expected
        comparable_expected = expected_text if case_sensitive else expected_text.lower()
        details.append({"type": "exact", "value": expected, "passed": comparable_text == comparable_expected})

    for expected in _expected_values(case, "expected_contains"):
        comparable_expected = expected if case_sensitive else expected.lower()
        details.append({"type": "contains", "value": expected, "passed": comparable_expected in comparable_text})

    for unexpected in _expected_values(case, "expected_not_contains"):
        comparable_unexpected = unexpected if case_sensitive else unexpected.lower()
        details.append(
            {"type": "not_contains", "value": unexpected, "passed": comparable_unexpected not in comparable_text}
        )

    regex_flags = re.DOTALL if case_sensitive else re.DOTALL | re.IGNORECASE
    for pattern in _expected_values(case, "expected_regex"):
        try:
            detail: dict[str, Any] = {
                "type": "regex",
                "value": pattern,
                "passed": re.search(pattern, text, regex_flags) is not None,
            }
        except re.error as exc:
            detail = {"type": "regex", "value": pattern, "passed": False, "error": str(exc)}
        details.append(detail)

    checks_total = len(details)
    checks_passed = sum(1 for detail in details if detail["passed"])
    return {
        "checks_total": checks_total,
        "checks_passed": checks_passed,
        "score": checks_passed / checks_total if checks_total else 1.0,
        "case_success": checks_passed == checks_total,
        "check_details": details,
    }


def _failure_score(case: dict[str, Any]) -> dict[str, Any]:
    score = _score_output("", case)
    if score["checks_total"]:
        score["checks_passed"] = 0
        score["score"] = 0.0
    score["case_success"] = False
    return score


def _load_cases(recipe: Any) -> list[dict[str, Any]]:
    raw_cases = recipe.dataset.get("cases")
    if raw_cases is not None:
        if not isinstance(raw_cases, list) or not all(isinstance(case, dict) for case in raw_cases):
            raise ValueError(f"Recipe '{recipe.id}' dataset.cases must be a list of tables")
        cases: list[dict[str, Any]] = []
        for index, raw_case in enumerate(raw_cases):
            prompt = raw_case.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"Recipe '{recipe.id}' dataset.cases[{index}].prompt must be a string")
            case = dict(raw_case)
            case["id"] = str(case.get("id", index))
            case["vertical"] = str(case.get("vertical", recipe.classification.get("vertical", "unspecified")))
            cases.append(case)
        return cases

    prompts = recipe.dataset.get("prompts", [])
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"Recipe '{recipe.id}' dataset.prompts must be a list of strings")
    default_vertical = str(recipe.classification.get("vertical", "unspecified"))
    return [{"id": str(index), "vertical": default_vertical, "prompt": prompt} for index, prompt in enumerate(prompts)]


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _first_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _completion_token_info(prompt_result: dict[str, Any]) -> dict[str, Any]:
    output = str(prompt_result.get("output", ""))
    usage = prompt_result.get("usage")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    source = "estimated_regex"

    if isinstance(usage, dict):
        prompt_tokens = _first_int(usage, ("prompt_tokens", "input_tokens"))
        completion_tokens = _first_int(usage, ("completion_tokens", "output_tokens"))
        if completion_tokens is not None:
            source = "usage"

    timings = prompt_result.get("timings")
    if completion_tokens is None and isinstance(timings, dict):
        completion_tokens = _first_int(timings, ("predicted_n", "tokens_predicted"))
        if completion_tokens is not None:
            source = "timings"

    if completion_tokens is None:
        completion_tokens = _estimate_token_count(output)

    token_info: dict[str, Any] = {
        "completion_tokens": completion_tokens,
        "completion_tokens_estimated": source == "estimated_regex",
        "token_count_source": source,
    }
    if prompt_tokens is not None:
        token_info["prompt_tokens"] = prompt_tokens
    return token_info


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _summary_for_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success = sum(1 for result in results if result["ok"])
    total_latency = sum(float(result["latency_ms"]) for result in results)
    total_chars = sum(len(str(result.get("output", ""))) for result in results if result["ok"])
    total_completion_tokens = sum(int(result.get("completion_tokens", 0)) for result in results if result["ok"])
    scored_results = [result for result in results if int(result.get("checks_total", 0)) > 0]
    accuracy_values = [float(result.get("score", 0.0)) for result in scored_results]
    accuracy = _mean(accuracy_values)
    case_success = sum(1 for result in results if result.get("case_success", False))
    ttft_values = [float(result["ttft_ms"]) for result in results if result["ok"] and result.get("ttft_ms") is not None]

    generation_latency = 0.0
    for result in results:
        if not result["ok"]:
            continue
        latency = float(result["latency_ms"])
        ttft = result.get("ttft_ms")
        generation_latency += max(latency - float(ttft), 0.0) if ttft is not None else latency

    success_rate = success / total if total else 0.0
    case_success_rate = case_success / total if total else 0.0
    quality_score = accuracy if accuracy is not None else success_rate
    quality_factor = accuracy if accuracy is not None else case_success_rate
    end_to_end_tokens_per_sec = (total_completion_tokens * 1000.0 / total_latency) if total_latency > 0 else 0.0
    tokens_per_sec = (total_completion_tokens * 1000.0 / generation_latency) if generation_latency > 0 else 0.0
    loss_values = [float(result["loss"]) for result in results if result.get("loss") is not None]
    eval_loss_values = [float(result["eval_loss"]) for result in results if result.get("eval_loss") is not None]

    return {
        "success_rate": success_rate,
        "case_success_rate": case_success_rate,
        "latency_ms_avg": total_latency / total if total else 0.0,
        "ttft_ms_avg": _mean(ttft_values),
        "output_chars_avg": total_chars / success if success else 0.0,
        "completion_tokens_total": total_completion_tokens,
        "output_tokens_avg": total_completion_tokens / success if success else 0.0,
        "tokens_per_sec": tokens_per_sec,
        "end_to_end_tokens_per_sec": end_to_end_tokens_per_sec,
        "effective_tokens_per_sec": end_to_end_tokens_per_sec * quality_factor,
        "accuracy": quality_score,
        "quality_loss": 1.0 - quality_score,
        "loss_proxy": 1.0 - quality_score,
        "loss_avg": _mean(loss_values),
        "eval_loss_avg": _mean(eval_loss_values),
        "throughput_prompts_per_sec": (total * 1000.0 / total_latency) if total_latency > 0 else 0.0,
    }


def _compute_kpis(results: list[dict[str, Any]], requested_metrics: list[str] | None = None) -> dict[str, Any]:
    all_kpis = _summary_for_results(results)
    agentic_results = [result for result in results if result.get("vertical") == "agentic"]
    all_kpis["workflow_success_rate"] = (
        sum(1 for result in agentic_results if result.get("case_success", False)) / len(agentic_results)
        if agentic_results
        else None
    )
    if not requested_metrics:
        return all_kpis
    return {metric: all_kpis[metric] for metric in requested_metrics if metric in all_kpis}


def _compute_vertical_breakdown(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    verticals = sorted({str(result.get("vertical", "unspecified")) for result in results})
    return {
        vertical: _summary_for_results(
            [result for result in results if str(result.get("vertical", "unspecified")) == vertical]
        )
        for vertical in verticals
    }


def run_recipe(
    recipe_id: str,
    *,
    recipes_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> RunResult:
    catalog = Catalog(recipes_dir or DEFAULT_RECIPES_DIR)
    recipe = catalog.get(recipe_id)
    adapter = str(recipe.runtime.get("adapter", ""))
    cases = _load_cases(recipe)

    repetitions = int(recipe.parameters.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("parameters.repetitions must be >= 1")

    started_at = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for prompt_index, case in enumerate(cases):
            prompt = str(case["prompt"])
            case_metadata = {
                "case_id": str(case.get("id", prompt_index)),
                "vertical": str(case.get("vertical", "unspecified")),
            }
            before = perf_counter()
            try:
                prompt_result = _execute_prompt(adapter, prompt, recipe.runtime, recipe.parameters)
                latency_ms = (perf_counter() - before) * 1000.0
                score_result = _score_output(str(prompt_result.get("output", "")), case)
                token_info = _completion_token_info(prompt_result)
                results.append(
                    {
                        "ok": True,
                        "repetition": repetition,
                        "prompt_index": prompt_index,
                        **case_metadata,
                        "prompt": prompt,
                        "latency_ms": round(latency_ms, 3),
                        **prompt_result,
                        **token_info,
                        **score_result,
                    }
                )
            except Exception as exc:  # keep per-prompt failures in the run artifact
                latency_ms = (perf_counter() - before) * 1000.0
                results.append(
                    {
                        "ok": False,
                        "repetition": repetition,
                        "prompt_index": prompt_index,
                        **case_metadata,
                        "prompt": prompt,
                        "error": str(exc),
                        "latency_ms": round(latency_ms, 3),
                        **_failure_score(case),
                    }
                )

    requested_metrics = recipe.kpis.get("metrics")
    metrics = {
        "recipe_id": recipe.id,
        "category": recipe.category,
        "classification": recipe.classification,
        "adapter": adapter,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "prompt_count": len(results),
        "success_count": sum(1 for result in results if result["ok"]),
        "error_count": sum(1 for result in results if not result["ok"]),
        "kpis": _compute_kpis(results, requested_metrics),
        "verticals": _compute_vertical_breakdown(results),
        "results": results,
    }

    base_output_dir = Path(output_dir) if output_dir is not None else Path("runs")
    actual_run_id = run_id or started_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_output_dir / f"{actual_run_id}-{_safe_run_name(recipe.id)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return RunResult(run_dir=run_dir, metrics_path=metrics_path, metrics=metrics)
