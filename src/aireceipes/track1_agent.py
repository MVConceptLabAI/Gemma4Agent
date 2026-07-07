from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_INPUT_PATH = Path("/input/tasks.json")
DEFAULT_OUTPUT_PATH = Path("/output/results.json")

CATEGORY_TOKEN_LIMITS = {
    "sentiment_classification": 64,
    "named_entity_recognition": 160,
    "mathematical_reasoning": 128,
    "factual_knowledge": 256,
    "text_summarisation": 256,
    "logical_deductive_reasoning": 256,
    "code_debugging": 512,
    "code_generation": 768,
}


@dataclass(frozen=True)
class FireworksConfig:
    api_key: str
    base_url: str
    allowed_models: tuple[str, ...]
    model: str


@dataclass(frozen=True)
class LocalAttempt:
    answer: str | None
    confidence: float
    category: str
    reason: str


RemoteAnswerer = Callable[[str, str, FireworksConfig, int], str]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(encoded)
        tmp_name = handle.name
    Path(tmp_name).replace(path)


def load_tasks(input_path: str | Path) -> list[dict[str, str]]:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tasks.json must contain a JSON array")
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"task at index {index} must be an object")
        task_id = item.get("task_id")
        prompt = item.get("prompt")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task at index {index} has invalid task_id")
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {task_id} has invalid prompt")
        seen.add(task_id)
        tasks.append({"task_id": task_id, "prompt": prompt})
    return tasks


def classify_task(prompt: str) -> str:
    text = prompt.lower()
    if re.search(r"\b(sentiment|positive|negative|neutral)\b", text):
        return "sentiment_classification"
    if re.search(r"\b(named entities|entity|entities|ner|person|organization|location)\b", text):
        return "named_entity_recognition"
    if re.search(r"\b(summarize|summarise|summary|tl;dr)\b", text):
        return "text_summarisation"
    if re.search(r"\b(debug|bug|fix the code|traceback|exception|syntaxerror)\b", text):
        return "code_debugging"
    if re.search(r"\b(write|generate|implement|create)\b.*\b(code|function|class|program|script)\b", text):
        return "code_generation"
    if re.search(r"\b(logic|deduce|deduction|therefore|syllogism|true or false|knights?|knaves?)\b", text):
        return "logical_deductive_reasoning"
    if _extract_math_expression(prompt) is not None or re.search(r"\b(calculate|compute|solve|what is)\b.*\d", text):
        return "mathematical_reasoning"
    return "factual_knowledge"


def _safe_eval_math(expr: str) -> float | int | None:
    allowed_binops: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    allowed_unary: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: lambda a: a,
        ast.USub: lambda a: -a,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return allowed_binops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return allowed_unary[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported math syntax: {ast.dump(node)}")

    try:
        parsed = ast.parse(expr, mode="eval")
        result = visit(parsed)
    except Exception:
        return None
    if not (abs(result) < 10**18):
        return None
    if float(result).is_integer():
        return int(result)
    return result


def _extract_math_expression(prompt: str) -> str | None:
    normalized = (
        prompt.replace("×", "*")
        .replace("x", "*")
        .replace("÷", "/")
        .replace("plus", "+")
        .replace("minus", "-")
        .replace("times", "*")
    )
    candidates = re.findall(r"[-+*/().\d\s]{3,}", normalized)
    useful: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip(" .,:;?!\n\t")
        if len(re.findall(r"\d", candidate)) >= 2 and re.search(r"[+*/-]", candidate):
            useful.append(candidate)
    return max(useful, key=len) if useful else None


def solve_math(prompt: str) -> LocalAttempt:
    expr = _extract_math_expression(prompt)
    if not expr:
        return LocalAttempt(None, 0.0, "mathematical_reasoning", "no_arithmetic_expression")
    result = _safe_eval_math(expr)
    if result is None:
        return LocalAttempt(None, 0.0, "mathematical_reasoning", "unsafe_or_unsupported_expression")
    answer = str(result)
    if isinstance(result, float):
        answer = (f"{result:.10f}").rstrip("0").rstrip(".")
    return LocalAttempt(answer, 0.96, "mathematical_reasoning", "safe_arithmetic_eval")


def solve_sentiment(prompt: str) -> LocalAttempt:
    text = prompt.lower()
    sample = text.split(":", 1)[-1]
    positive_words = {
        "amazing",
        "awesome",
        "best",
        "excellent",
        "fantastic",
        "fast",
        "good",
        "great",
        "happy",
        "love",
        "loved",
        "perfect",
        "reliable",
        "wonderful",
    }
    negative_words = {
        "awful",
        "bad",
        "broken",
        "buggy",
        "hate",
        "hated",
        "poor",
        "slow",
        "terrible",
        "unhappy",
        "unreliable",
        "worst",
    }
    words = re.findall(r"[a-z']+", sample)
    positive = sum(1 for word in words if word in positive_words)
    negative = sum(1 for word in words if word in negative_words)
    if positive > negative:
        return LocalAttempt("positive", 0.92, "sentiment_classification", "lexical_positive")
    if negative > positive:
        return LocalAttempt("negative", 0.92, "sentiment_classification", "lexical_negative")
    if re.search(r"\b(okay|fine|average|neutral|ordinary|acceptable)\b", sample):
        return LocalAttempt("neutral", 0.82, "sentiment_classification", "lexical_neutral")
    return LocalAttempt(None, 0.0, "sentiment_classification", "ambiguous_sentiment")


def solve_ner(prompt: str) -> LocalAttempt:
    text = prompt.split(":", 1)[-1].strip()
    entities: list[str] = []
    for match in re.finditer(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b", text):
        value = match.group(0).strip()
        if value.lower() not in {"extract", "named", "entities", "answer"} and value not in entities:
            entities.append(value)
    if not entities:
        return LocalAttempt(None, 0.0, "named_entity_recognition", "no_regex_entities")
    return LocalAttempt(", ".join(entities), 0.74, "named_entity_recognition", "capitalized_entity_regex")


def solve_summarisation(prompt: str) -> LocalAttempt:
    text = prompt.split(":", 1)[-1].strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(text) < 220 or not sentences:
        return LocalAttempt(None, 0.0, "text_summarisation", "too_short_or_ambiguous")
    summary = sentences[0]
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "..."
    return LocalAttempt(summary, 0.68, "text_summarisation", "first_sentence_summary")


def try_local_solver(prompt: str, category: str | None = None) -> LocalAttempt:
    category = category or classify_task(prompt)
    if category == "mathematical_reasoning":
        return solve_math(prompt)
    if category == "sentiment_classification":
        return solve_sentiment(prompt)
    if category == "named_entity_recognition":
        return solve_ner(prompt)
    if category == "text_summarisation":
        return solve_summarisation(prompt)
    return LocalAttempt(None, 0.0, category, "requires_model_reasoning")


def normalize_answer(answer: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(answer)).strip()
    cleaned = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", cleaned).strip()
    return cleaned or "I don't know."


def parse_allowed_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(model.strip() for model in raw.split(",") if model.strip())


def select_model(category: str, allowed_models: Sequence[str]) -> str:
    if not allowed_models:
        raise ValueError("ALLOWED_MODELS is required for Fireworks fallback")
    lowered = [(model, model.lower()) for model in allowed_models]
    gemma = [model for model, lower in lowered if "gemma" in lower]
    if gemma:
        return gemma[0]
    return allowed_models[0]


def load_fireworks_config(category: str) -> FireworksConfig:
    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    base_url = os.environ.get("FIREWORKS_BASE_URL", "").strip().rstrip("/")
    allowed_models = parse_allowed_models(os.environ.get("ALLOWED_MODELS"))
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY is required for Fireworks fallback")
    if not base_url:
        raise ValueError("FIREWORKS_BASE_URL is required for Fireworks fallback")
    model = select_model(category, allowed_models)
    return FireworksConfig(api_key=api_key, base_url=base_url, allowed_models=allowed_models, model=model)


def compact_prompt(prompt: str, category: str) -> str:
    instruction_by_category = {
        "factual_knowledge": "Answer the question accurately in English. Keep the answer concise.",
        "mathematical_reasoning": "Solve the problem. Return only the final answer unless a short unit is required.",
        "sentiment_classification": "Classify sentiment as exactly one of: positive, negative, neutral.",
        "text_summarisation": "Summarize the text in English, preserving the requested length or style.",
        "named_entity_recognition": "Extract named entities. Return a concise comma-separated list unless another format is requested.",
        "code_debugging": "Fix or explain the code bug concisely. Return the requested code or answer only.",
        "logical_deductive_reasoning": "Reason carefully and return the concise final answer in English.",
        "code_generation": "Generate correct code for the request. Return code only unless explanation is explicitly requested.",
    }
    return f"{instruction_by_category.get(category, instruction_by_category['factual_knowledge'])}\n\nTask:\n{prompt.strip()}"


def call_fireworks_chat(prompt: str, category: str, config: FireworksConfig, max_tokens: int) -> str:
    url = f"{config.base_url}/chat/completions"
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "You are a benchmark agent. Answer in English only. Be concise and do not reveal chain-of-thought."},
            {"role": "user", "content": compact_prompt(prompt, category)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=28) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Fireworks HTTP {exc.code}: {detail}") from exc
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Fireworks response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("Fireworks choice was not an object")
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    raise RuntimeError("Fireworks response did not include text content")


def answer_task(prompt: str, remote_answerer: RemoteAnswerer = call_fireworks_chat) -> tuple[str, dict[str, Any]]:
    category = classify_task(prompt)
    local = try_local_solver(prompt, category)
    audit: dict[str, Any] = {
        "category": category,
        "local_confidence": local.confidence,
        "local_reason": local.reason,
    }
    if local.answer is not None and local.confidence >= 0.80:
        audit.update({"route": "local", "remote_used": False, "model": None})
        return normalize_answer(local.answer), audit

    try:
        config = load_fireworks_config(category)
        max_tokens = CATEGORY_TOKEN_LIMITS.get(category, 256)
        answer = remote_answerer(prompt, category, config, max_tokens)
        audit.update({"route": "fireworks", "remote_used": True, "model": config.model, "max_tokens": max_tokens})
        return normalize_answer(answer), audit
    except Exception as exc:
        # The official harness supplies Fireworks variables. This fallback keeps local
        # smoke tests valid and preserves one answer per task if the env is absent.
        audit.update({"route": "fallback", "remote_used": False, "model": None, "error": str(exc)})
        if local.answer:
            return normalize_answer(local.answer), audit
        return "I don't know.", audit


def run_competition_agent(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    remote_answerer: RemoteAnswerer = call_fireworks_chat,
) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)
    tasks = load_tasks(input_path)
    results: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    for task in tasks:
        answer, audit = answer_task(task["prompt"], remote_answerer=remote_answerer)
        results.append({"task_id": task["task_id"], "answer": answer})
        audit_rows.append({"task_id": task["task_id"], **audit})

    input_ids = [task["task_id"] for task in tasks]
    output_ids = [result["task_id"] for result in results]
    if input_ids != output_ids:
        raise RuntimeError("internal error: output task_id order does not match input")
    if any(not result["answer"] for result in results):
        raise RuntimeError("internal error: empty answer")

    _atomic_write_json(output_path, results)
    _atomic_write_json(output_path.parent / "route_audit.json", audit_rows)
    print(json.dumps({"answered": len(results), "output": str(output_path)}, sort_keys=True), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track 1 competition agent: /input/tasks.json -> /output/results.json")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input tasks JSON path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output results JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_competition_agent(input_path=args.input, output_path=args.output)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
