"""Gemma Hybrid V20: V19 speed with bounded V18 evidence recovery."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Callable

from app.demo_pipeline import (
    _creative_anchor_count,
    _has_lexical_corruption,
    _has_process_leak,
    caption_demo,
)
from app.gemma_fast import (
    _contradicts_formal,
    _fallbacks,
    _too_similar,
    caption_gemma_fast,
)
from app.models import REQUIRED_STYLES, FALLBACK_CAPTIONS, normalize_captions

FAST_TIMEOUT_S = float(os.environ.get("HYBRID_FAST_TIMEOUT_S", "185"))
RECOVERY_TIMEOUT_S = float(os.environ.get("HYBRID_RECOVERY_TIMEOUT_S", "220"))
BATCH_RECOVERY_TIMEOUT_S = float(os.environ.get("HYBRID_BATCH_RECOVERY_TIMEOUT_S", "200"))
BATCH_RECOVERY_MAX = int(os.environ.get("HYBRID_BATCH_RECOVERY_MAX", "2"))
# A V18 recovery fans out into many model calls.  It is a safety net, not a
# second default pipeline: more than one recovery per batch can starve the
# remaining videos and force static captions at the end of the run.
RECOVERY_MAX_PER_RUN = int(os.environ.get("HYBRID_RECOVERY_MAX_PER_RUN", "1"))
_recovery_claims = 0

log = logging.getLogger("track2.gemma_hybrid")

_RAW_STRUCTURE = re.compile(r"(?:\{\s*['\"]?fact['\"]?\s*:|\[\s*\{\s*['\"]|<\/?.+?>)", re.I)
_GENERIC = re.compile(
    r"\b(?:visible subjects?|visible action receives|visible people or objects|"
    r"foreground elements|background context in the queue|history books nearly missed|"
    r"small everyday moment (?:steps forward|carries on)|truly a (?:pinnacle|masterclass|monumental))\b",
    re.I,
)
_UNSUPPORTED_VALUE = re.compile(
    r"\b(?:expensive|costly|pricey|luxury|luxurious|spending (?:a )?fortune|worth (?:a )?fortune)\b",
    re.I,
)
_CONTENT_STOPWORDS = {
    "about", "after", "again", "along", "also", "around", "because", "before",
    "being", "between", "from", "into", "just", "like", "near", "only", "over",
    "that", "their", "there", "these", "they", "this", "through", "under", "very",
    "while", "with", "would", "video", "scene", "shows", "showing",
}


def _claim_recovery_slot() -> bool:
    """Reserve the batch's scarce V18 path without awaiting between check/set."""
    global _recovery_claims
    if _recovery_claims >= max(0, RECOVERY_MAX_PER_RUN):
        return False
    _recovery_claims += 1
    return True


def _reset_recovery_slots() -> None:
    """Test hook; a contest invocation uses one fresh Python process."""
    global _recovery_claims
    _recovery_claims = 0


def _requires_deep_recovery(reasons: list[str]) -> bool:
    """Reserve expensive visual re-analysis for broken, not merely imperfect, output."""
    return any(
        reason.endswith(("missing-or-too-short", "generic-fallback", "pipeline-leak", "lexical-corruption"))
        for reason in reasons
    )


def _weak_formal_grounding(formal: str, style: str, value: str) -> bool:
    """Use the verified fast formal caption to reject category-only jokes."""
    if style not in {"sarcastic", "humorous_tech", "humorous_non_tech"}:
        return False
    return _creative_anchor_count(formal, value) < 3


def _creative_recovery_fallbacks(formal: str) -> dict[str, str]:
    """Ground a bounded recovery fallback in the exact accepted formal caption."""
    anchor = formal.strip().rstrip(".!?") or "Visible subjects move through the scene"
    words = anchor.split()
    if len(words) > 24:
        anchor = " ".join(words[:24]).rstrip(",;:")
    return {
        "sarcastic": (
            f"{anchor}, receiving the level of ceremony normally reserved for a state occasion."
        ),
        "humorous_tech": (
            f"{anchor}, moving like a network routing service keeping every visible path in order."
        ),
        "humorous_non_tech": (
            f"{anchor}, like guests finding their way through one crowded doorway without losing their place."
        ),
    }


def _repair_fast_after_failed_recovery(
    captions: dict[str, str], styles: list[str], reasons: list[str]
) -> dict[str, str]:
    """Never return a risky fast joke merely because the richer path timed out."""
    repaired = dict(captions)
    fallbacks = _creative_recovery_fallbacks(repaired.get("formal", ""))
    risky_styles = {
        reason.split(":", 1)[0]
        for reason in reasons
        if ":" in reason
    }
    for style in risky_styles:
        if style in fallbacks and style in styles:
            repaired[style] = fallbacks[style]
    return normalize_captions(repaired, styles)


def caption_risks(captions: dict[str, str], styles: list[str]) -> list[str]:
    """Return high-confidence reasons that justify the expensive V18 recovery."""
    reasons: list[str] = []
    for style in styles:
        value = captions.get(style, "")
        if not isinstance(value, str) or len(value.split()) < 7:
            reasons.append(f"{style}:missing-or-too-short")
            continue
        if value.strip() == FALLBACK_CAPTIONS.get(style, "").strip() or _GENERIC.search(value):
            reasons.append(f"{style}:generic-fallback")
        if _RAW_STRUCTURE.search(value) or _has_process_leak(value):
            reasons.append(f"{style}:pipeline-leak")
        if _has_lexical_corruption(value):
            reasons.append(f"{style}:lexical-corruption")

    formal = captions.get("formal", "")
    for style in styles:
        if style != "formal" and formal and captions.get(style):
            if _contradicts_formal(formal, captions[style]):
                reasons.append(f"{style}:unsupported-scene-noun")
            if _weak_formal_grounding(formal, style, captions[style]):
                reasons.append(f"{style}:weak-formal-grounding")
            if (
                _UNSUPPORTED_VALUE.search(captions[style])
                and not re.search(r"\b(?:race|racing|supercar|luxury)\b", formal, re.I)
            ):
                reasons.append(f"{style}:unsupported-value-claim")
    ordered = [style for style in REQUIRED_STYLES if style in styles]
    for index, style in enumerate(ordered):
        if any(_too_similar(captions.get(style, ""), captions.get(prior, "")) for prior in ordered[:index]):
            reasons.append(f"{style}:near-duplicate-voice")
    return sorted(set(reasons))


async def _recover(video_url: str, styles: list[str], timeout: float, reason: str) -> dict[str, str]:
    log.warning("V20 activating V18 evidence recovery (%s; %.0fs budget)", reason, timeout)
    recovered = await asyncio.wait_for(
        caption_demo(video_url=video_url, styles=styles),
        timeout=timeout,
    )
    # V18 already writes, rewrites, verifies, and quality-checks each creative
    # caption against its evidence record.  Running V19's fast-path repair here
    # used to replace that richer copy with a generic category fallback (for
    # example, the traffic CPU-scheduler template).  Keep the evidence-first
    # result and use only the contract/style normalizer at this boundary.
    return normalize_captions(recovered, styles)


async def caption_gemma_hybrid(video_url: str, styles: list[str]) -> dict[str, str]:
    """Try one-call V19, then recover only a failed or unsafe task with V18."""
    try:
        fast = await asyncio.wait_for(
            caption_gemma_fast(video_url=video_url, styles=styles),
            timeout=FAST_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        # Keep this broad by design: provider HTTP errors, malformed model JSON,
        # download/FFmpeg failures and timeouts all need the same grounded route.
        if not _claim_recovery_slot():
            log.warning("V20 fast path failed but the batch recovery quota is spent (%s)", type(exc).__name__)
            return normalize_captions(_fallbacks(""), styles)
        return normalize_captions(
            await _recover(video_url, styles, RECOVERY_TIMEOUT_S, type(exc).__name__),
            styles,
        )
    fast = normalize_captions(fast, styles)
    risks = caption_risks(fast, styles)
    if not risks:
        log.info("V20 accepted the Gemma Fast result")
        return fast
    # Most risks are a style or calibration issue in an otherwise usable
    # single-call result.  Repair them against its visible formal anchors
    # locally instead of turning every quality signal into a 24-frame V18 run.
    if not _requires_deep_recovery(risks):
        log.info("V20 repaired fast style risks locally: %s", ", ".join(risks))
        return _repair_fast_after_failed_recovery(fast, styles, risks)
    if not _claim_recovery_slot():
        log.warning("V20 deep recovery quota spent; repairing fast result locally: %s", ", ".join(risks))
        return _repair_fast_after_failed_recovery(fast, styles, risks)
    try:
        return normalize_captions(
            await _recover(video_url, styles, RECOVERY_TIMEOUT_S, ", ".join(risks)),
            styles,
        )
    except Exception as exc:  # noqa: BLE001
        # Keep the formal caption, but do not leak a risky creative draft just
        # because the richer V18 pass reached its bounded timeout.
        log.warning("V20 evidence recovery failed (%s); grounding risky fast creative captions", exc)
        return _repair_fast_after_failed_recovery(fast, styles, risks)


def _meaningful_ngrams(value: str, size: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    output: set[str] = set()
    for index in range(max(0, len(words) - size + 1)):
        gram = words[index : index + size]
        if sum(word not in _CONTENT_STOPWORDS and len(word) >= 4 for word in gram) >= 3:
            output.add(" ".join(gram))
    return output


def repeated_task_ids(results: list[dict[str, Any]]) -> list[str]:
    """Find later tasks that reuse an exact, content-heavy five-word joke span."""
    owners: dict[tuple[str, str], str] = {}
    repeated: list[str] = []
    for result in results:
        task_id = str(result.get("task_id", ""))
        captions = result.get("captions", {})
        for style in ("sarcastic", "humorous_tech", "humorous_non_tech"):
            for gram in _meaningful_ngrams(str(captions.get(style, ""))):
                key = (style, gram)
                if key in owners and owners[key] != task_id:
                    if task_id and task_id not in repeated:
                        repeated.append(task_id)
                else:
                    owners[key] = task_id
    return repeated


async def recover_batch_repetitions(
    tasks: list[dict[str, Any]],
    results: list[dict[str, Any]],
    remaining_budget: Callable[[], float],
) -> list[dict[str, Any]]:
    """Re-run at most two repeated tasks with V18 while the global clock allows."""
    targets = repeated_task_ids(results)[: max(0, BATCH_RECOVERY_MAX)]
    if not targets:
        return results
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    results_by_id = {str(result["task_id"]): result for result in results}
    for task_id in targets:
        remaining = remaining_budget()
        timeout = min(BATCH_RECOVERY_TIMEOUT_S, max(0.0, remaining - 15.0))
        if timeout < 45.0:
            log.warning("V20 found repeated wording in %s but global budget is too low", task_id)
            continue
        task = tasks_by_id.get(task_id)
        if not task:
            continue
        try:
            captions = await _recover(
                str(task["video_url"]),
                list(task.get("styles") or REQUIRED_STYLES),
                timeout,
                "batch repetition",
            )
            results_by_id[task_id]["captions"] = normalize_captions(
                captions,
                list(task.get("styles") or REQUIRED_STYLES),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("V20 repetition recovery failed for %s: %s", task_id, exc)
    return [results_by_id[str(result["task_id"])] for result in results]
