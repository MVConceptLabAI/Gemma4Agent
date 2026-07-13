#!/usr/bin/env python3
"""Use Gemma 4 as an offline promotion judge for a complete results JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx


def parse_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("audit model returned no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("audit result is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("GEMMA_AUDIT_MODEL", "google/gemma-4-31b-it"))
    args = parser.parse_args()

    key = "".join(os.environ.get("OPENROUTER_API_KEY", "").split())
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    prompt = (
        "You are an independent Gemma 4 promotion judge reviewing a complete batch of video captions. "
        "The formal caption is the factual anchor available for each task. Audit the JSON batch for: "
        "(1) exact or semantically repeated jokes/templates across tasks; (2) shared cliches; (3) style mismatch; "
        "(4) concrete claims, subjects, actions, places, counts, camera techniques, thoughts, intent, speech, "
        "failure, absence, or stillness introduced by a styled caption but unsupported by its formal anchor; "
        "(5) awkward grammar, truncated clauses, internal pipeline wording, or weak non-jokes. Do not penalize "
        "ordinary factual overlap such as 'the video shows'. Be strict but distinguish a clearly figurative "
        "comparison from a literal hallucination. Return ONLY strict JSON with this schema: "
        '{"pass":boolean,"accuracy_risk":number,"style_quality":number,"diversity":number,'
        '"shared_patterns":[{"phrase":string,"tasks":[string],"severity":"low|medium|high"}],'
        '"issues":[{"task_id":string,"style":string,"severity":"low|medium|high",'
        '"type":"hallucination_risk|repetition|cliche|style|grammar|weak_joke",'
        '"phrase":string,"reason":string,"suggested_rewrite":string}],"summary":string}. '
        "Scores are 0-1. Set pass=false for any high-severity issue or repeated stock fallback.\n\nRESULTS:\n"
        + json.dumps(rows, ensure_ascii=False)
    )
    url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") + "/chat/completions"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": 0.0,
        "max_tokens": 2600,
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(url, headers={"Authorization": f"Bearer {key}"}, json=body)
        response.raise_for_status()
        raw = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            audit = parse_object(raw)
        except (json.JSONDecodeError, ValueError):
            body["messages"].append({
                "role": "user",
                "content": "Return the same audit again as one strict double-quoted JSON object only.",
            })
            response = client.post(url, headers={"Authorization": f"Bearer {key}"}, json=body)
            response.raise_for_status()
            audit = parse_object(response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
