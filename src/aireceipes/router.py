from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

LOWER_IS_BETTER = {"latency_ms_avg", "ttft_ms_avg", "quality_loss", "loss_proxy", "cost_per_1m_output_tokens_usd"}
DEFAULT_COST_PER_1M_OUTPUT_TOKENS_USD = {
    "local": 0.0,
    "local-llama": 0.0,
    "llama.cpp": 0.0,
    "diffusiongemma": 0.0,
    "unsloth-studio-or-compatible": 0.0,
}


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _load_comparison(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"comparison {path} must contain a JSON object")
    return payload


def _workflow_metrics(model: dict[str, Any], workflow: str) -> dict[str, Any]:
    verticals = model.get("verticals")
    if isinstance(verticals, dict):
        workflow_values = verticals.get(workflow)
        if isinstance(workflow_values, dict):
            return workflow_values
    kpis = model.get("kpis")
    return kpis if isinstance(kpis, dict) else {}


def _cost(model: dict[str, Any]) -> float:
    routing = model.get("routing")
    if isinstance(routing, dict):
        value = _as_float(routing.get("cost_per_1m_output_tokens_usd"))
        if value is not None:
            return value
    backend = str(model.get("backend") or "local").lower()
    return DEFAULT_COST_PER_1M_OUTPUT_TOKENS_USD.get(backend, 0.0)


def _candidate_from_model(model: dict[str, Any], *, workflow: str, source: str) -> dict[str, Any]:
    metrics = _workflow_metrics(model, workflow)
    kpis = model.get("kpis") if isinstance(model.get("kpis"), dict) else {}
    quality = (
        _as_float(metrics.get("accuracy"))
        or _as_float(metrics.get("case_success_rate"))
        or _as_float(kpis.get("accuracy"))
        or _as_float(kpis.get("case_success_rate"))
        or 0.0
    )
    effective_tps = _as_float(metrics.get("effective_tokens_per_sec")) or _as_float(kpis.get("effective_tokens_per_sec")) or 0.0
    latency_ms = _as_float(metrics.get("latency_ms_avg")) or _as_float(kpis.get("latency_ms_avg"))
    prompt_count = int(_as_float(model.get("prompt_count")) or 0)
    success_count = int(_as_float(model.get("success_count")) or 0)
    cost = _cost(model)
    speed_score = effective_tps
    cost_score = 1.0 / (1.0 + cost)
    balanced_score = (quality * 0.50) + (speed_score / max(speed_score, 1.0) * 0.30) + (cost_score * 0.20)
    return {
        "recipe_id": str(model.get("recipe_id") or model.get("name") or "unknown"),
        "name": model.get("name"),
        "backend": model.get("backend"),
        "model_family": model.get("model_family"),
        "mode": model.get("mode"),
        "target_gpu": model.get("target_gpu"),
        "workflow": workflow,
        "quality": round(quality, 6),
        "effective_tokens_per_sec": round(effective_tps, 6),
        "latency_ms_avg": round(latency_ms, 6) if latency_ms is not None else None,
        "cost_per_1m_output_tokens_usd": round(cost, 6),
        "prompt_count": prompt_count,
        "success_count": success_count,
        "source": source,
        "scores": {
            "fastest": round(speed_score, 6),
            "cheapest": round(-cost, 6),
            "balanced": round(balanced_score, 6),
        },
    }


def _sort_key(candidate: dict[str, Any], strategy: str) -> tuple[float, float, float, float]:
    quality = float(candidate["quality"])
    speed = float(candidate["effective_tokens_per_sec"])
    cost = float(candidate["cost_per_1m_output_tokens_usd"])
    latency = candidate.get("latency_ms_avg")
    latency_value = float(latency) if latency is not None else 1_000_000_000.0
    if strategy == "fastest":
        return (speed, quality, -cost, -latency_value)
    if strategy == "cheapest":
        return (-cost, quality, speed, -latency_value)
    if strategy == "quality":
        return (quality, speed, -cost, -latency_value)
    if strategy != "balanced":
        raise ValueError("strategy must be one of: balanced, fastest, cheapest, quality")
    speed_component = speed / max(speed, 1.0)
    cost_component = 1.0 / (1.0 + cost)
    score = (quality * 0.50) + (speed_component * 0.30) + (cost_component * 0.20)
    return (score, quality, speed, -cost)


def build_route_plan(
    comparisons: Iterable[str | Path],
    *,
    workflow: str = "agentic",
    strategy: str = "balanced",
    min_accuracy: float = 0.0,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in comparisons:
        source = str(Path(path))
        sources.append(source)
        comparison = _load_comparison(path)
        models = comparison.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            candidate = _candidate_from_model(model, workflow=workflow, source=source)
            if candidate["quality"] < min_accuracy:
                rejected.append({**candidate, "reject_reason": "below_min_accuracy"})
            elif candidate["prompt_count"] and candidate["success_count"] <= 0:
                rejected.append({**candidate, "reject_reason": "no_successful_prompts"})
            else:
                candidates.append(candidate)

    ordered = sorted(candidates, key=lambda item: _sort_key(item, strategy), reverse=True)
    rejected_ordered = sorted(rejected, key=lambda item: (item.get("reject_reason", ""), item["recipe_id"]))
    selected = ordered[0] if ordered else None
    return {
        "workflow": workflow,
        "strategy": strategy,
        "min_accuracy": min_accuracy,
        "sources": sources,
        "selected": selected,
        "candidates": ordered,
        "rejected": rejected_ordered,
        "route_policy": [
            {
                "match": {"vertical": workflow},
                "route_to": {
                    "recipe_id": selected["recipe_id"],
                    "backend": selected.get("backend"),
                    "model_family": selected.get("model_family"),
                    "mode": selected.get("mode"),
                },
                "fallback": [
                    {
                        "recipe_id": candidate["recipe_id"],
                        "backend": candidate.get("backend"),
                        "model_family": candidate.get("model_family"),
                        "mode": candidate.get("mode"),
                    }
                    for candidate in ordered[1:]
                ],
            }
        ]
        if selected
        else [],
    }


def write_route_plan(
    comparisons: Iterable[str | Path],
    output_path: str | Path,
    *,
    workflow: str = "agentic",
    strategy: str = "balanced",
    min_accuracy: float = 0.0,
) -> Path:
    path = Path(output_path)
    plan = build_route_plan(comparisons, workflow=workflow, strategy=strategy, min_accuracy=min_accuracy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return path
