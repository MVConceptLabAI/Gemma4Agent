"""Blind paired multimodal evaluation for caption candidates.

The assignment cache is concealed from the judge. Caption A and Caption B are
scored together against six real review frames, then remapped to their source
arms only after the response has passed strict validation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

import httpx


Arm = Literal["baseline", "candidate"]
Assignment = tuple[Arm, Arm]
AssignmentMap = dict[tuple[str, str], Assignment]

REQUIRED_STYLES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)
CACHE_SCHEMA = "blind-assignment-cache/v1"
REPORT_SCHEMA = "paired-multimodal-eval/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

STYLE_DEFINITIONS = {
    "formal": "Professional, objective, factual tone",
    "sarcastic": "Dry, ironic, lightly mocking",
    "humorous_tech": "Funny, with technology or programming references",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon",
}

OFFICIAL_PAIRED_RUBRIC = (
    "You are the LLM-Judge of a video-captioning contest. You see six frames "
    "sampled in chronological order from ONE video clip, one requested style, "
    "and two anonymous captions. Score each caption independently on exactly "
    "two dimensions:\n"
    "1. accuracy (0-1): how faithfully the caption reflects the video content. "
    "Penalize unsupported subjects, actions, objects, settings, counts, colors, "
    "text, and chronology; reward correct, specific coverage.\n"
    "2. style (0-1): how well the caption matches the requested tone.\n"
    "Return STRICT JSON only, with exactly this shape: "
    '{"a":{"accuracy":0.0,"style":0.0,"wrong_claims":[]},'
    '"b":{"accuracy":0.0,"style":0.0,"wrong_claims":[]}}. '
    "wrong_claims must be a JSON array of short strings. Do not infer which "
    "caption came from which system."
)

FrameLoader = Callable[[str, int], list[str]]
JudgeFunction = Callable[
    [Any, list[str], str, str, str, str],
    Awaitable[dict[str, object]],
]


def blind_assignment(
    seed: str,
    task_id: str,
    style: str,
    judge_model: str,
) -> Assignment:
    """Return a stable single-pair fallback assignment derived from SHA-256."""
    material = "|".join((seed, task_id, style, judge_model)).encode("utf-8")
    bit = hashlib.sha256(material).digest()[0] & 1
    return (
        ("baseline", "candidate")
        if bit == 0
        else ("candidate", "baseline")
    )


def build_blind_assignments(
    seed: str,
    judge_model: str,
    pairs: Sequence[tuple[str, str]],
) -> AssignmentMap:
    """Build balanced assignments for a complete four-style corpus run."""
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("complete blind assignment requires at least one pair")
    if len(set(pair_list)) != len(pair_list):
        raise ValueError("blind assignment pairs must be unique")

    expected_styles = set(REQUIRED_STYLES)
    styles_by_clip: dict[str, set[str]] = {}
    for task_id, style in pair_list:
        if style not in expected_styles:
            raise ValueError(f"unknown requested style: {style}")
        styles_by_clip.setdefault(task_id, set()).add(style)
    for task_id, styles in styles_by_clip.items():
        if styles != expected_styles:
            missing = sorted(expected_styles - styles)
            raise ValueError(f"incomplete styles for {task_id}: {missing}")

    clips = sorted(
        styles_by_clip,
        key=lambda task_id: hashlib.sha256(
            f"{seed}|{judge_model}|{task_id}".encode("utf-8")
        ).digest(),
    )
    assignments: AssignmentMap = {}
    for clip_index, task_id in enumerate(clips):
        pattern = (0, 1, 0, 1) if clip_index % 2 == 0 else (1, 0, 1, 0)
        for style_index, style in enumerate(REQUIRED_STYLES):
            assignments[(task_id, style)] = (
                ("baseline", "candidate")
                if pattern[style_index] == 0
                else ("candidate", "baseline")
            )
    return assignments


def judge_caption_prompt(caption_a: str, caption_b: str) -> str:
    """Encode anonymous captions in unambiguous judge-facing containers."""
    return json.dumps(
        {"Caption A": caption_a, "Caption B": caption_b},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_assignment_cache(
    path: Path,
    seed: str,
    judge_model: str,
    assignments: Mapping[tuple[str, str], Assignment],
) -> None:
    """Persist the concealed remapping separately from judge-facing prompts."""
    style_rank = {style: index for index, style in enumerate(REQUIRED_STYLES)}
    records = []
    for (task_id, style), order in sorted(
        assignments.items(),
        key=lambda item: (item[0][0], style_rank.get(item[0][1], len(style_rank))),
    ):
        if order not in {
            ("baseline", "candidate"),
            ("candidate", "baseline"),
        }:
            raise ValueError(f"invalid blind assignment for {task_id}/{style}")
        records.append(
            {
                "caption_a_source": order[0],
                "caption_b_source": order[1],
                "style": style,
                "task_id": task_id,
            }
        )
    payload = {
        "assignments": records,
        "judge_model": judge_model,
        "schema": CACHE_SCHEMA,
        "seed": seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_assignment_cache(
    path: Path,
    *,
    expected_seed: str,
    expected_judge_model: str,
) -> AssignmentMap:
    """Load a concealed mapping only when its provenance matches expectations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported blind assignment cache schema")
    for field, expected in (
        ("seed", expected_seed),
        ("judge_model", expected_judge_model),
    ):
        if field not in payload:
            raise ValueError(f"blind assignment cache is missing {field}")
        if payload[field] != expected:
            raise ValueError(f"blind assignment cache {field} mismatch")
    records = payload.get("assignments")
    if not isinstance(records, list):
        raise ValueError("blind assignment cache must contain an assignments list")

    assignments: AssignmentMap = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("blind assignment cache record must be an object")
        task_id = record.get("task_id")
        style = record.get("style")
        order = (record.get("caption_a_source"), record.get("caption_b_source"))
        if not isinstance(task_id, str) or style not in REQUIRED_STYLES:
            raise ValueError("invalid blind assignment cache key")
        if order not in {
            ("baseline", "candidate"),
            ("candidate", "baseline"),
        }:
            raise ValueError(f"invalid blind assignment cache order for {task_id}/{style}")
        key = (task_id, style)
        if key in assignments:
            raise ValueError(f"duplicate blind assignment cache key: {task_id}/{style}")
        assignments[key] = order  # type: ignore[assignment]
    return assignments


def _strict_unit_score(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field} must be finite and within 0-1")
    return score


def validate_paired_verdict(payload: object) -> dict[str, object]:
    """Validate and normalize the exact anonymous two-arm response schema."""
    if not isinstance(payload, dict) or set(payload) != {"a", "b"}:
        raise ValueError("paired verdict must contain exactly a and b")

    normalized: dict[str, object] = {}
    for label in ("a", "b"):
        item = payload[label]
        if not isinstance(item, dict) or set(item) != {
            "accuracy",
            "style",
            "wrong_claims",
        }:
            raise ValueError(
                f"{label} must contain exactly accuracy, style, and wrong_claims"
            )
        wrong_claims = item["wrong_claims"]
        if not isinstance(wrong_claims, list) or not all(
            isinstance(claim, str) for claim in wrong_claims
        ):
            raise ValueError(f"{label}.wrong_claims must be a list of strings")
        normalized[label] = {
            "accuracy": _strict_unit_score(
                item["accuracy"], f"{label}.accuracy"
            ),
            "style": _strict_unit_score(item["style"], f"{label}.style"),
            "wrong_claims": list(wrong_claims),
        }
    return normalized


def parse_paired_verdict(text: str) -> dict[str, object]:
    """Parse one strict JSON object, tolerating only a surrounding code fence."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("unterminated JSON code fence")
        candidate = "\n".join(lines[1:-1]).strip()
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(candidate, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("judge response is not one JSON object") from exc
    return validate_paired_verdict(payload)


def paired_request_payload(
    style: str,
    caption_a: str,
    caption_b: str,
    frames: Sequence[str],
    judge_model: str,
) -> dict[str, object]:
    """Build a blind OpenAI-compatible multimodal request with six frames."""
    if style not in STYLE_DEFINITIONS:
        raise ValueError(f"unknown requested style: {style}")
    if len(frames) != 6:
        raise ValueError(f"paired judge requires exactly six frames, got {len(frames)}")
    anonymous_captions = judge_caption_prompt(caption_a, caption_b)
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                f"Requested style: {style} = {STYLE_DEFINITIONS[style]}\n"
                f"Anonymous captions: {anonymous_captions}\n"
                "Frames follow in chronological order:"
            ),
        }
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
        }
        for frame in frames
    )
    return {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": OFFICIAL_PAIRED_RUBRIC},
            {"role": "user", "content": content},
        ],
        "max_tokens": 1200,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }


async def judge_pair(
    client: Any,
    frames: list[str],
    style: str,
    caption_a: str,
    caption_b: str,
    judge_model: str,
    *,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> dict[str, object]:
    """Score one anonymous pair, retrying transport and format failures."""
    if retries < 1:
        raise ValueError("retries must be at least one")
    payload = paired_request_payload(
        style, caption_a, caption_b, frames, judge_model
    )
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = "unknown judge failure"
    for attempt in range(retries):
        try:
            response = await client.post(
                os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_URL),
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("judge message content must be text")
            return parse_paired_verdict(text)
        except Exception as exc:  # noqa: BLE001 - retry all unreliable provider failures
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries and retry_delay:
                await asyncio.sleep(retry_delay)
    return {"skipped": f"judge failed after {retries} attempts: {last_error}"}


def _default_frame_loader(video_url: str, n: int = 6) -> list[str]:
    # Lazy import keeps offline assignment/parser tests independent of .env and ffmpeg.
    from eval.frame_judge import frames_b64

    return frames_b64(video_url, n=n)


def _normalize_results(
    results: Sequence[Mapping[str, object]],
    label: str,
) -> dict[tuple[str, str], str]:
    captions_by_pair: dict[tuple[str, str], str] = {}
    seen_tasks: set[str] = set()
    for row in results:
        task_id = row.get("task_id")
        captions = row.get("captions")
        if not isinstance(task_id, str) or not isinstance(captions, Mapping):
            raise ValueError(f"invalid {label} result row")
        if task_id in seen_tasks:
            raise ValueError(f"duplicate {label} task_id: {task_id}")
        seen_tasks.add(task_id)
        for style, caption in captions.items():
            if style not in REQUIRED_STYLES or not isinstance(caption, str):
                raise ValueError(f"invalid {label} caption for {task_id}/{style}")
            captions_by_pair[(task_id, style)] = caption
    if not captions_by_pair:
        raise ValueError(f"{label} results contain no captions")
    return captions_by_pair


def _normalize_tasks(
    tasks: Sequence[Mapping[str, object]] | Mapping[str, str],
) -> dict[str, str]:
    if isinstance(tasks, Mapping):
        normalized = dict(tasks)
        if not all(
            isinstance(task_id, str) and isinstance(video_url, str)
            for task_id, video_url in normalized.items()
        ):
            raise ValueError("task mapping must contain string URLs")
        return normalized

    normalized: dict[str, str] = {}
    for task in tasks:
        task_id = task.get("task_id")
        video_url = task.get("video_url")
        if not isinstance(task_id, str) or not isinstance(video_url, str):
            raise ValueError("invalid task row")
        if task_id in normalized:
            raise ValueError(f"duplicate task_id: {task_id}")
        normalized[task_id] = video_url
    return normalized


def _declared_task_styles(
    tasks: Sequence[Mapping[str, object]] | Mapping[str, str],
) -> dict[str, set[str]]:
    if isinstance(tasks, Mapping):
        return {}
    declared: dict[str, set[str]] = {}
    for task in tasks:
        task_id = task.get("task_id")
        styles = task.get("styles")
        if styles is None:
            continue
        if not isinstance(task_id, str) or not isinstance(styles, list):
            raise ValueError("task styles must be a JSON array")
        if not styles or not all(
            isinstance(style, str) and style in REQUIRED_STYLES for style in styles
        ):
            raise ValueError(f"invalid requested styles for {task_id}")
        style_set = set(styles)
        if len(style_set) != len(styles):
            raise ValueError(f"duplicate requested style for {task_id}")
        declared[task_id] = style_set
    return declared


def _ordered_pairs(pairs: set[tuple[str, str]]) -> list[tuple[str, str]]:
    style_rank = {style: index for index, style in enumerate(REQUIRED_STYLES)}
    return sorted(pairs, key=lambda pair: (pair[0], style_rank[pair[1]]))


def _resolve_assignments(
    pairs: list[tuple[str, str]],
    seed: str,
    judge_model: str,
    assignment_cache_path: Path | None,
) -> AssignmentMap:
    pair_set = set(pairs)
    if assignment_cache_path is not None and assignment_cache_path.exists():
        assignments = read_assignment_cache(
            assignment_cache_path,
            expected_seed=seed,
            expected_judge_model=judge_model,
        )
        if set(assignments) != pair_set:
            raise ValueError("blind assignment cache pair set mismatch")
        return assignments

    try:
        assignments = build_blind_assignments(seed, judge_model, pairs)
    except ValueError as exc:
        if not str(exc).startswith("incomplete styles for"):
            raise
        assignments = {
            pair: blind_assignment(seed, pair[0], pair[1], judge_model)
            for pair in pairs
        }
    if assignment_cache_path is not None:
        write_assignment_cache(
            assignment_cache_path,
            seed,
            judge_model,
            assignments,
        )
    return assignments


def _round_score(value: float) -> float:
    return round(value, 12)


def _arm_summary(rows: Sequence[Mapping[str, object]], arm: Arm) -> dict[str, float]:
    accuracy = _round_score(
        sum(float(row["arms"][arm]["accuracy"]) for row in rows) / len(rows)  # type: ignore[index]
    )
    style = _round_score(
        sum(float(row["arms"][arm]["style"]) for row in rows) / len(rows)  # type: ignore[index]
    )
    return {
        "accuracy": accuracy,
        "style": style,
        "final": _round_score((accuracy + style) / 2.0),
    }


def _build_summary(
    valid_rows: list[dict[str, object]],
    total_pairs: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if not valid_rows:
        empty_arm = {"accuracy": None, "style": None, "final": None}
        summary = {
            "total_pairs": total_pairs,
            "valid_pairs": 0,
            "skipped_pairs": total_pairs,
            "baseline": dict(empty_arm),
            "candidate": dict(empty_arm),
            "delta": dict(empty_arm),
            "clip_minima": {
                "baseline_final": None,
                "candidate_final": None,
                "delta_final": None,
            },
        }
        return {}, summary

    baseline = _arm_summary(valid_rows, "baseline")
    candidate = _arm_summary(valid_rows, "candidate")
    delta = {
        axis: _round_score(candidate[axis] - baseline[axis])
        for axis in ("accuracy", "style", "final")
    }

    rows_by_clip: dict[str, list[dict[str, object]]] = {}
    for row in valid_rows:
        rows_by_clip.setdefault(str(row["task_id"]), []).append(row)
    clip_summaries: dict[str, object] = {}
    for task_id, rows in sorted(rows_by_clip.items()):
        clip_baseline = _arm_summary(rows, "baseline")
        clip_candidate = _arm_summary(rows, "candidate")
        clip_summaries[task_id] = {
            "valid_pairs": len(rows),
            "baseline": clip_baseline,
            "candidate": clip_candidate,
            "delta": {
                axis: _round_score(clip_candidate[axis] - clip_baseline[axis])
                for axis in ("accuracy", "style", "final")
            },
        }

    summary = {
        "total_pairs": total_pairs,
        "valid_pairs": len(valid_rows),
        "skipped_pairs": total_pairs - len(valid_rows),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "clip_minima": {
            "baseline_final": min(
                item["baseline"]["final"] for item in clip_summaries.values()  # type: ignore[index,union-attr]
            ),
            "candidate_final": min(
                item["candidate"]["final"] for item in clip_summaries.values()  # type: ignore[index,union-attr]
            ),
            "delta_final": min(
                item["delta"]["final"] for item in clip_summaries.values()  # type: ignore[index,union-attr]
            ),
        },
    }
    return clip_summaries, summary


async def evaluate_paired_results(
    baseline_results: Sequence[Mapping[str, object]],
    candidate_results: Sequence[Mapping[str, object]],
    tasks: Sequence[Mapping[str, object]] | Mapping[str, str],
    client: Any,
    judge_model: str,
    seed: str,
    assignment_cache_path: Path | None,
    *,
    frame_loader: FrameLoader = _default_frame_loader,
    judge_fn: JudgeFunction = judge_pair,
) -> dict[str, object]:
    """Run blind paired judgments and aggregate only valid finite verdicts."""
    baseline = _normalize_results(baseline_results, "baseline")
    candidate = _normalize_results(candidate_results, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate task/style pairs must match exactly")
    task_urls = _normalize_tasks(tasks)
    for task_id, requested_styles in _declared_task_styles(tasks).items():
        actual_styles = {
            style for result_task_id, style in baseline if result_task_id == task_id
        }
        if actual_styles != requested_styles:
            raise ValueError(
                f"result styles for {task_id} do not match tasks.json: "
                f"expected {sorted(requested_styles)}, got {sorted(actual_styles)}"
            )
    pairs = _ordered_pairs(set(baseline))
    missing_tasks = sorted({task_id for task_id, _ in pairs} - set(task_urls))
    if missing_tasks:
        raise ValueError(f"tasks are missing video URLs: {missing_tasks}")
    assignments = _resolve_assignments(
        pairs, seed, judge_model, assignment_cache_path
    )

    valid_rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    pairs_by_task: dict[str, list[tuple[str, str]]] = {}
    for pair in pairs:
        pairs_by_task.setdefault(pair[0], []).append(pair)

    for task_id, task_pairs in pairs_by_task.items():
        try:
            frames = frame_loader(task_urls[task_id], 6)
            if len(frames) != 6:
                raise ValueError(f"expected six frames, got {len(frames)}")
        except Exception as exc:  # noqa: BLE001 - one bad video must not abort the panel
            reason = f"frame extraction failed: {type(exc).__name__}: {exc}"
            skipped.extend(
                {"task_id": task_id, "style": style, "reason": reason}
                for _, style in task_pairs
            )
            continue

        for _, style in task_pairs:
            order = assignments[(task_id, style)]
            captions = {
                "baseline": baseline[(task_id, style)],
                "candidate": candidate[(task_id, style)],
            }
            caption_a = captions[order[0]]
            caption_b = captions[order[1]]
            try:
                raw_verdict = await judge_fn(
                    client,
                    frames,
                    style,
                    caption_a,
                    caption_b,
                    judge_model,
                )
                if "skipped" in raw_verdict:
                    raise ValueError(str(raw_verdict["skipped"]))
                verdict = validate_paired_verdict(raw_verdict)
            except Exception as exc:  # noqa: BLE001 - invalid/non-finite votes are excluded
                skipped.append(
                    {
                        "task_id": task_id,
                        "style": style,
                        "reason": f"judge skipped: {type(exc).__name__}: {exc}",
                    }
                )
                continue

            arms = {
                order[0]: verdict["a"],
                order[1]: verdict["b"],
            }
            baseline_score = arms["baseline"]
            candidate_score = arms["candidate"]
            deltas = {
                axis: _round_score(
                    float(candidate_score[axis]) - float(baseline_score[axis])  # type: ignore[index]
                )
                for axis in ("accuracy", "style")
            }
            deltas["final"] = _round_score(
                (deltas["accuracy"] + deltas["style"]) / 2.0
            )
            valid_rows.append(
                {
                    "task_id": task_id,
                    "style": style,
                    "presentation": {
                        "caption_a_source": order[0],
                        "caption_b_source": order[1],
                        "caption_a": caption_a,
                        "caption_b": caption_b,
                    },
                    "judge_raw": verdict,
                    "arms": arms,
                    "deltas": deltas,
                }
            )

    clip_summaries, summary = _build_summary(valid_rows, len(pairs))
    return {
        "schema": REPORT_SCHEMA,
        "judge_model": judge_model,
        "seed": seed,
        "pairs": valid_rows,
        "skipped": skipped,
        "clip_summaries": clip_summaries,
        "summary": summary,
    }


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


async def _run_cli(args: argparse.Namespace) -> dict[str, object]:
    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    tasks = _load_json(args.tasks)
    if not isinstance(baseline, list) or not isinstance(candidate, list):
        raise ValueError("baseline and candidate result files must be JSON arrays")
    if not isinstance(tasks, list):
        raise ValueError("tasks file must be a JSON array")
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        return await evaluate_paired_results(
            baseline,
            candidate,
            tasks,
            client,
            args.judge_model,
            args.seed,
            args.assignment_cache,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind paired multimodal caption evaluator"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--seed", default="v38-paired-gate")
    parser.add_argument("--assignment-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        # frame_judge performs the repository's existing key-only .env load.
        from eval import frame_judge as _frame_judge  # noqa: F401

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required")
    report = asyncio.run(_run_cli(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = report["summary"]
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
