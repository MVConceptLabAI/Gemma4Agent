"""Dev-only multimodal factual post-audit for causal caption A/B tests.

This tool is intentionally outside the production application and Docker path.
It samples real video frames through ``eval/frame_judge.py``, asks one
multimodal OpenRouter model to minimally correct all four captions together,
and records per-task audit status in a sidecar JSON file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx
from dotenv import load_dotenv


STYLE_KEYS = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)
DEFAULT_MODEL = "openai/gpt-5.5"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FRAME_COUNT = 8
MAX_ATTEMPTS = 3

AUDIT_PROMPT = """You are a minimal edit factual auditor for video captions.
You see eight real frames from one video and four captions describing that same
video in four requested styles.

Apply only minimal edits. Remove or replace only unsupported concrete claims,
including unsupported directions, counts, colors, OCR/text, identities,
intentions, or backstory. A replacement may only make a claim less specific and
visually supported. Preserve supported details, the existing length band, each
requested style, and jokes as framing. Add no new concrete fact. Do not rewrite
for elegance, add detail, shorten substantially, or flatten the four voices.

Return exact four-key JSON with these keys and no others: formal, sarcastic,
humorous_tech, humorous_non_tech. Every value must be a non-empty string. Return
the JSON object only, without markdown or commentary.
"""


def parse_audit_response(text: str) -> dict[str, str]:
    """Parse and validate the auditor's strict four-caption JSON response."""

    if not isinstance(text, str):
        raise ValueError("audit response must be text")
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("audit response is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("audit response must be a JSON object")
    if set(payload) != set(STYLE_KEYS) or len(payload) != len(STYLE_KEYS):
        raise ValueError("audit response must contain exactly the four style keys")

    corrected: dict[str, str] = {}
    for key in STYLE_KEYS:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"audit caption {key!r} must be a non-empty string")
        corrected[key] = value.strip()
    return corrected


def load_real_frames(video_source: str) -> list[str]:
    """Load exactly eight real frames using frame_judge's local/URL support."""

    try:
        import frame_judge
    except ModuleNotFoundError:
        from eval import frame_judge

    return frame_judge.frames_b64(video_source, n=FRAME_COUNT)


def _request_content(frames: Sequence[str], captions: Mapping[str, str]) -> list[dict[str, Any]]:
    caption_json = json.dumps(
        {key: captions[key] for key in STYLE_KEYS},
        ensure_ascii=False,
        indent=2,
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Audit these four captions together against the eight frames. "
                "Keep every supported phrase unchanged whenever possible.\n\n"
                f"Captions:\n{caption_json}\n\nFrames follow in chronological order."
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
    return content


def _safe_error(exc: BaseException, api_key: str) -> str:
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return f"{type(exc).__name__}: {message}"[:1000]


def _response_content(response: Any) -> str:
    status_code = int(getattr(response, "status_code", 0))
    if not 200 <= status_code < 300:
        raise ValueError(f"OpenRouter returned HTTP {status_code}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("OpenRouter response is missing message content") from exc
    if not isinstance(content, str):
        raise ValueError("OpenRouter message content must be text")
    return content


async def audit_caption_set(
    client: Any,
    frames: Sequence[str],
    captions: Mapping[str, str],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_OPENROUTER_URL,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Audit one four-caption set, preserving originals after three failures."""

    originals = parse_audit_response(json.dumps(dict(captions), ensure_ascii=False))
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"expected {FRAME_COUNT} frames, got {len(frames)}")

    request_json = {
        "model": model,
        "messages": [
            {"role": "system", "content": AUDIT_PROMPT},
            {"role": "user", "content": _request_content(frames, originals)},
        ],
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    last_error = "unknown audit failure"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_json,
            )
            corrected = parse_audit_response(_response_content(response))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            last_error = _safe_error(exc, api_key)
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(1)
            continue

        return corrected, {
            "status": "unchanged" if corrected == originals else "corrected",
            "attempts": attempt,
        }

    return dict(originals), {
        "status": "failed",
        "attempts": MAX_ATTEMPTS,
        "error": last_error,
    }


async def audit_results(
    results: Sequence[Mapping[str, Any]],
    tasks: Mapping[str, str],
    *,
    client: Any,
    api_key: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_OPENROUTER_URL,
    frame_loader: Callable[[str], list[str]] = load_real_frames,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit every results row and return corrected results plus sidecar data."""

    corrected_results: list[dict[str, Any]] = []
    task_metadata: list[dict[str, Any]] = []

    for source_row in results:
        row = dict(source_row)
        task_id = str(row.get("task_id", ""))
        source_captions = row.get("captions", {})
        originals = dict(source_captions) if isinstance(source_captions, Mapping) else {}
        row["captions"] = dict(originals)

        try:
            video_source = tasks[task_id]
            frames = frame_loader(video_source)
            corrected, metadata = await audit_caption_set(
                client,
                frames,
                originals,
                api_key=api_key,
                model=model,
                endpoint=endpoint,
            )
        except Exception as exc:  # one bad clip must not destroy the A/B artifact
            corrected = dict(originals)
            metadata = {
                "status": "failed",
                "attempts": 0,
                "error": _safe_error(exc, api_key),
            }

        row["captions"] = corrected
        corrected_results.append(row)
        task_metadata.append({"task_id": task_id, **metadata})

    return corrected_results, {
        "model": model,
        "frame_count": FRAME_COUNT,
        "tasks": task_metadata,
    }


def _endpoint_from_environment() -> str:
    configured = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    if not configured:
        return DEFAULT_OPENROUTER_URL
    if configured.rstrip("/").endswith("/chat/completions"):
        return configured.rstrip("/")
    return configured.rstrip("/") + "/chat/completions"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def _run_cli(args: argparse.Namespace) -> None:
    load_dotenv(override=False)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")

    results_path = Path(args.results)
    tasks_path = Path(args.tasks)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    task_rows = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = {str(row["task_id"]): str(row["video_url"]) for row in task_rows}

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        corrected, sidecar = await audit_results(
            results,
            tasks,
            client=client,
            api_key=api_key,
            model=args.model,
            endpoint=_endpoint_from_environment(),
        )

    _write_json(Path(args.output), corrected)
    _write_json(Path(args.metadata), sidecar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="input results.json")
    parser.add_argument("--tasks", required=True, help="tasks.json with video_url values")
    parser.add_argument("--output", required=True, help="corrected results.json")
    parser.add_argument("--metadata", required=True, help="audit sidecar JSON")
    parser.add_argument(
        "--model",
        default=os.environ.get("POST_AUDIT_MODEL", DEFAULT_MODEL),
        help=f"OpenRouter multimodal model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    asyncio.run(_run_cli(parser.parse_args()))


if __name__ == "__main__":
    main()
