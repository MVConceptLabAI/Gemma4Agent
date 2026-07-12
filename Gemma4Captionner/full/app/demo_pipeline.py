"""Gemma 4 caption flow shared with the browser demo.

The public demo samples frames locally, asks Gemma for two grounded facts per
frame, then asks the same model to write, improve the two humour captions and
verify the final result.  This server-side equivalent keeps that exact
evidence-first sequence for the hackathon Docker contract.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from app import pipeline as P

MODEL = os.environ.get("DEMO_GEMMA_MODEL", "google/gemma-4-31b-it")
API_KEY = "".join(os.environ.get("OPENROUTER_API_KEY", "").split())
API_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
FRAME_COUNT = int(os.environ.get("NUM_FRAMES", "24"))
FRAME_EDGE = int(os.environ.get("FRAME_MAX_EDGE", "640"))
OBSERVATION_CONCURRENCY = int(os.environ.get("DEMO_OBSERVATION_CONCURRENCY", "3"))

_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _limiter() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(OBSERVATION_CONCURRENCY)
        _semaphore_loop = loop
    return _semaphore


def _json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Gemma did not return caption JSON")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Gemma returned a non-object JSON response")
    return value


def _facts(text: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    try:
        value = json.loads(text[start : end + 1]) if start >= 0 and end > start else []
    except json.JSONDecodeError:
        value = []
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:96] for item in value if str(item).strip()][:2]


async def _ask(
    client: httpx.AsyncClient, content: Any, max_tokens: int, temperature: float = 0.25
) -> str:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for the Gemma 4 demo flow")
    response = await client.post(
        f"{API_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    response.raise_for_status()
    answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Gemma returned no text content")
    return answer


def _image_content(frame: Path) -> list[dict[str, Any]]:
    data = base64.b64encode(frame.read_bytes()).decode("ascii")
    prompt = (
        "You are a precise visual analyst. This is ONE frame from a video. Return ONLY a JSON "
        "array of at most TWO very short, clearly visible facts: subjects, actions, objects, "
        "setting, motion or lighting. Do not infer identity, intent, speech, brands, unreadable "
        "text, exact counts, chronology, or off-screen events."
    )
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
    ]


def _evidence(observations: list[list[str]]) -> str:
    return "\n\n".join(
        f"FRAME {index + 1} OF {len(observations)}:\n- " + "\n- ".join(facts)
        for index, facts in enumerate(observations)
        if facts
    )


def _captions(value: dict[str, Any], styles: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for style in styles:
        caption = value.get(style)
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Gemma missed the {style} caption")
        output[style] = caption.strip()
    return output


async def _write(client: httpx.AsyncClient, evidence: str, styles: list[str]) -> dict[str, str]:
    prompt = (
        "You are a careful video-captioning writer. Consecutive single-frame observations describe "
        "ONE video. Cross-check them. The clip may be a montage: when scenes differ, describe their "
        "sequence instead of pretending they happen in one physical scene. Keep only facts supported "
        "by the records; omit conflicts and do not invent identity, intent, speech, brands, unreadable "
        "text, exact counts, or off-screen events. Return ONLY valid JSON with exactly these string "
        "keys: formal, sarcastic, humorous_tech, humorous_non_tech. formal is objective and factual "
        "in one or two sentences. sarcastic is one dry, lightly mocking sentence with no technology "
        "jargon. humorous_tech is one genuinely funny sentence: use exactly one natural technology "
        "comparison and a concrete playful payoff, never a loose jargon list, 'too many tabs', "
        "'glitchy cache', or 'random' subject. humorous_non_tech is one warm everyday joke with a "
        "concrete playful payoff; never invent a school, event, profession, intent, or unseen setting, "
        "and never use 'mixed bag', 'scrapbook', 'watching paint dry', 'watching grass grow', "
        "'nothing happens', or 'the only thing of interest'.\n\n"
        + evidence
    )
    return _captions(_json_object(await _ask(client, [{"type": "text", "text": prompt}], 800)), styles)


async def _rewrite_humour(
    client: httpx.AsyncClient, evidence: str, candidate: str, style: str
) -> str:
    tech = style == "humorous_tech"
    prompt = (
        "You are editing ONE " + style + " video caption. Rewrite the candidate using only the "
        "verified evidence below. It must begin with a visible setup and land a small playful payoff "
        "tied to a visible contrast, action, or transition. "
        + ("Use exactly ONE natural technology comparison; do not use 'too many tabs', 'glitchy cache', or 'random'. " if tech else "Use a fresh everyday comparison and never use technical vocabulary, 'mixed bag', 'scrapbook', 'watching paint dry', 'watching grass grow', 'nothing happens', or 'the only thing of interest'. ")
        + "Never invent identity, speech, brands, intent, backstory, audience, profession, or unseen events. "
        "Keep it to one sentence. Return ONLY JSON: {\"caption\":\"...\"}.\n\nVERIFIED EVIDENCE:\n"
        + evidence + "\n\nCURRENT CANDIDATE:\n" + candidate
    )
    result = _json_object(await _ask(client, [{"type": "text", "text": prompt}], 300))
    caption = result.get("caption")
    return caption.strip() if isinstance(caption, str) and caption.strip() else candidate


async def _verify(client: httpx.AsyncClient, evidence: str, captions: dict[str, str], styles: list[str]) -> dict[str, str]:
    prompt = (
        "You are the final grounded-caption verifier. Compare the candidate captions with the verified "
        "visual evidence for ONE video. Return ONLY valid JSON with exactly these string keys: formal, "
        "sarcastic, humorous_tech, humorous_non_tech. If every literal claim is supported, copy that "
        "candidate VERBATIM. Otherwise change ONLY the smallest risky clause; preserve supported visual "
        "nouns, actions, sequence details, style, humour and length. Never summarize or simplify a "
        "detailed caption. Remove claims about frames, sampling, prompts, models or analysis. A tech or "
        "everyday comparison may remain only as a figurative joke anchored to a visible fact.\n\n"
        "VERIFIED EVIDENCE:\n" + evidence + "\n\nCANDIDATE CAPTIONS:\n" + json.dumps(captions)
    )
    reviewed = _captions(
        _json_object(await _ask(client, [{"type": "text", "text": prompt}], 800)), styles
    )
    # Match the demo's detail-preservation guard: a verifier may be correct but
    # over-cautiously replace a rich grounded sentence with a generic summary.
    for style in styles:
        source, checked = captions[style], reviewed[style]
        if (
            len(source.split()) >= 20
            and len(checked.split()) < len(source.split()) * 0.7
            and not any(term in source.lower() for term in ("frame", "sampling", "prompt", "model", "analysis"))
        ):
            reviewed[style] = source
    return reviewed


async def caption_demo(video_url: str, styles: list[str]) -> dict[str, str]:
    """Run the same Gemma 4 evidence/write/rewrite/verify flow as docs/demo.html."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        video = await P._download(video_url, workdir / "clip.mp4")
        frames = P._extract_keyframes(video, workdir, FRAME_COUNT, FRAME_EDGE)
        async with httpx.AsyncClient(timeout=httpx.Timeout(150.0)) as client:
            async def observe(frame: Path) -> list[str]:
                for attempt in range(2):
                    try:
                        async with _limiter():
                            facts = _facts(await _ask(client, _image_content(frame), 180))
                        if facts:
                            return facts
                    except (httpx.HTTPError, ValueError):
                        if attempt:
                            raise
                        await asyncio.sleep(0.5)
                return []

            observations = await asyncio.gather(*(observe(frame) for frame in frames))
            record = _evidence(observations)
            if not record:
                raise RuntimeError("Gemma returned no grounded frame observations")
            captions = await _write(client, record, styles)
            rewrites = await asyncio.gather(
                _rewrite_humour(client, record, captions["humorous_tech"], "humorous_tech"),
                _rewrite_humour(client, record, captions["humorous_non_tech"], "humorous_non_tech"),
                return_exceptions=True,
            )
            for style, rewrite in zip(("humorous_tech", "humorous_non_tech"), rewrites):
                if isinstance(rewrite, str) and rewrite:
                    captions[style] = rewrite
            return await _verify(client, record, captions, styles)
