"""
AMD Track 2 - Video Captioning Agent entry point.

Contract (from Participant Guide):
    IN : /input/tasks.json   -> [{task_id, video_url, styles:[...]}]
    OUT: /output/results.json -> [{task_id, captions:{style: text, ...}}]

Constraints:
    - Container must be READY within 60 s (no heavy startup)
    - Total runtime < 10 min
    - Response < 30 s / request
    - Output MUST be valid JSON, all requested styles present (missing -> 0)
    - English only
    - Runs on linux/amd64
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from app.models import (
    REQUIRED_STYLES,
    fallback_caption,
    normalize_captions,
    parse_tasks,
    validate_results,
)
from app.pipeline import caption_one_video
from app.ensemble import caption_ensemble

CAPTION_ENGINE = os.environ.get("CAPTION_ENGINE", "pipeline")

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))

# Hard budget per task (leaves margin under the 30 s/request harness limit).
PER_TASK_TIMEOUT_S = float(os.environ.get("PER_TASK_TIMEOUT_S", "25"))
# Max concurrent videos processed in parallel. Keep low to avoid rate limits.
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "3"))
# The judging harness kills the WHOLE run at 10 minutes; a partially-degraded
# score beats a TIMEOUT (observed: stalled provider retries ran the clock out
# and the submission was marked unscored). Tasks that would start or run past
# this budget emit styled fallbacks instead.
GLOBAL_BUDGET_S = float(os.environ.get("GLOBAL_BUDGET_S", "540"))
_RUN_T0 = time.monotonic()


def _remaining_budget() -> float:
    return GLOBAL_BUDGET_S - (time.monotonic() - _RUN_T0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("track2")


def _empty_caption_set(styles: list[str]) -> dict[str, str]:
    """A caption for each requested style, even on failure."""
    return {s: fallback_caption(s) for s in styles}


async def _run_one(sem: asyncio.Semaphore, task: dict[str, Any]) -> dict[str, Any]:
    task_id = task.get("task_id", "?")
    styles = task.get("styles") or list(REQUIRED_STYLES)
    video_url = task.get("video_url", "")

    async with sem:
        t0 = time.perf_counter()
        # Never run past the global budget: shrink this task's timeout to what
        # is left, and skip straight to fallbacks when the budget is spent.
        task_timeout = min(PER_TASK_TIMEOUT_S, _remaining_budget())
        if task_timeout < 10:
            log.warning("[%s] global budget spent - emitting fallback captions", task_id)
            return {"task_id": task_id, "captions": normalize_captions(_empty_caption_set(styles), styles)}
        try:
            if CAPTION_ENGINE == "ensemble":
                try:
                    captions = await asyncio.wait_for(
                        caption_ensemble(video_url=video_url, styles=styles),
                        timeout=task_timeout,
                    )
                except Exception as e:  # noqa: BLE001
                    # Ensemble needs paid frontier APIs; on any failure (e.g. 402
                    # out-of-credit) degrade to the single-model pipeline, which
                    # itself falls back to Groq — never emit generic captions.
                    log.warning("[%s] ensemble failed (%s); falling back to pipeline", task_id, e)
                    captions = await asyncio.wait_for(
                        caption_one_video(video_url=video_url, styles=styles),
                        timeout=min(PER_TASK_TIMEOUT_S, _remaining_budget()),
                    )
            elif CAPTION_ENGINE == "qwen_direct":
                # Opt-in experimental path. Lazy import keeps the default and
                # legacy v36 paths unchanged unless explicitly selected.
                from app.qwen_direct import caption_qwen_direct

                captions = await asyncio.wait_for(
                    caption_qwen_direct(video_url=video_url, styles=styles),
                    timeout=task_timeout,
                )
            else:
                captions = await asyncio.wait_for(
                    caption_one_video(video_url=video_url, styles=styles),
                    timeout=task_timeout,
                )
        except asyncio.TimeoutError:
            log.warning("[%s] TIMEOUT after %.1fs - emitting fallback captions", task_id, task_timeout)
            captions = _empty_caption_set(styles)
        except Exception as e:  # noqa: BLE001 - never let one clip take the run down
            log.exception("[%s] pipeline failed: %s", task_id, e)
            captions = _empty_caption_set(styles)

        # Missing or empty captions score 0, so normalize before writing.
        captions = normalize_captions(captions, styles)

        dt = time.perf_counter() - t0
        log.info("[%s] done in %.1fs", task_id, dt)
        return {"task_id": task_id, "captions": captions}


async def _amain() -> int:
    if not INPUT_PATH.exists():
        log.error("Input file not found: %s", INPUT_PATH)
        # Write an empty valid JSON so the harness sees SOMETHING.
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text("[]", encoding="utf-8")
        return 1

    try:
        raw_tasks = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        tasks_in = parse_tasks(raw_tasks)
    except Exception as e:  # noqa: BLE001
        log.exception("Invalid input file %s: %s", INPUT_PATH, e)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text("[]", encoding="utf-8")
        return 1

    log.info("Loaded %d task(s) from %s", len(tasks_in), INPUT_PATH)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results = await asyncio.gather(*(_run_one(sem, t) for t in tasks_in))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_results(results)
    OUTPUT_PATH.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %d result(s) -> %s", len(results), OUTPUT_PATH)
    return 0


def main() -> None:
    try:
        rc = asyncio.run(_amain())
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
