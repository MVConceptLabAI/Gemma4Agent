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
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

from app import pipeline as P

MODEL = os.environ.get("DEMO_GEMMA_MODEL", "google/gemma-4-31b-it")
VIDEO_MODEL = os.environ.get("DEMO_VIDEO_MODEL", "google/gemma-4-26b-a4b-it")
API_KEY = "".join(os.environ.get("OPENROUTER_API_KEY", "").split())
API_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
FRAME_COUNT = int(os.environ.get("NUM_FRAMES", "24"))
FRAME_EDGE = int(os.environ.get("FRAME_MAX_EDGE", "640"))
OBSERVATION_CONCURRENCY = int(os.environ.get("DEMO_OBSERVATION_CONCURRENCY", "3"))
VIDEO_CONTEXT_MAX_SECONDS = int(os.environ.get("VIDEO_CONTEXT_MAX_SECONDS", "60"))
VIDEO_CONTEXT_MAX_BYTES = int(os.environ.get("VIDEO_CONTEXT_MAX_BYTES", "12000000"))

log = logging.getLogger("track2.gemma_video")

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
    client: httpx.AsyncClient,
    content: Any,
    max_tokens: int,
    temperature: float = 0.25,
    model: str = MODEL,
) -> str:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for the Gemma 4 demo flow")
    for attempt in range(2):
        response = await client.post(
            f"{API_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(answer, str) and answer.strip():
            return answer
        if attempt == 0:
            log.warning("Gemma model %s returned empty content; retrying once", model)
            await asyncio.sleep(1)
    raise ValueError("Gemma returned no text content after retry")


def _video_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return max(0.0, float(result.stdout.strip()))


def _compress_video_segments(
    video: Path, workdir: Path
) -> list[tuple[int, float, str]]:
    """Create one or two bounded MP4 segments covering up to two minutes."""
    try:
        duration = _video_duration(video)
        starts = [0.0]
        if duration > VIDEO_CONTEXT_MAX_SECONDS:
            # Hackathon clips are capped at two minutes. For a slightly longer
            # development clip, anchor the second part to the end rather than
            # silently discarding the final seconds.
            starts.append(max(0.0, duration - VIDEO_CONTEXT_MAX_SECONDS))
        segments: list[tuple[int, float, str]] = []
        for index, start in enumerate(starts, start=1):
            output = workdir / f"gemma4_video_context_{index}.mp4"
            remaining = max(1.0, duration - start) if duration else float(VIDEO_CONTEXT_MAX_SECONDS)
            segment_seconds = min(float(VIDEO_CONTEXT_MAX_SECONDS), remaining)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{segment_seconds:.3f}",
                    "-vf", "fps=1,scale=512:-2", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "34", "-c:a", "aac",
                    "-b:a", "32k", "-movflags", "+faststart", str(output),
                ],
                check=True,
                timeout=120,
            )
            raw = output.read_bytes()
            if not raw or len(raw) > VIDEO_CONTEXT_MAX_BYTES:
                log.warning(
                    "direct video segment %d unavailable or too large (%d bytes)",
                    index,
                    len(raw),
                )
                continue
            segments.append((index, start, base64.b64encode(raw).decode("ascii")))
        log.info("prepared %d direct video segment(s) for %.1fs clip", len(segments), duration)
        return segments
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log.warning("direct video compression failed: %s", exc)
        return []


def _video_facts(text: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    try:
        value = json.loads(text[start : end + 1]) if start >= 0 and end > start else []
    except json.JSONDecodeError:
        value = []
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:180] for item in value if str(item).strip()][:8]


async def _direct_video_evidence(
    client: httpx.AsyncClient,
    video_b64: str,
    frame_record: str,
    segment_index: int,
    segment_start: float,
    segment_count: int,
) -> list[str]:
    """Use Gemma 4 native video only after the frame evidence is established."""
    if not video_b64 or not VIDEO_MODEL:
        return []
    prompt = (
        "The verified frame observations below span one video. Use them as the factual anchor, "
        f"then inspect MP4 part {segment_index} of {segment_count}, beginning around "
        f"{segment_start:.0f} seconds, to identify only additional temporal actions, transitions, "
        "or clearly audible information that is directly confirmed and consistent with those "
        "observations. Never infer speech, sound, identity, intent, location, or an unseen event. "
        "If audio is absent, unclear, or unsupported, do not mention it. Return ONLY a JSON array "
        "of at most eight short facts. Return [] when the video adds no reliable information.\n\n"
        "VERIFIED FRAME EVIDENCE:\n" + frame_record
    )
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        },
    ]
    try:
        raw = await _ask(
            client,
            content,
            500,
            temperature=0.0,
            model=VIDEO_MODEL,
        )
        facts = _video_facts(raw)
        if facts:
            log.info(
                "direct Gemma 4 video part %d added %d fact(s): %s",
                segment_index,
                len(facts),
                facts,
            )
        else:
            log.warning("direct Gemma 4 video observer returned no usable facts")
        return facts
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("direct Gemma 4 video observer failed; retaining frame evidence: %s", exc)
        return []


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
        "verified evidence below. It must begin with one precise verified subject-action setup and land a "
        "distinct, clearly figurative comic consequence tied to that setup. Never force an unrelated second "
        "visible detail into the joke. The payoff must add a comic turn, not merely restate the comparison. "
        + ("Use exactly ONE natural technology comparison that explains a visible fact; do not use 'too many tabs', 'glitchy cache', 'random', a jargon list, 'X feels like Y, except', or 'the only thing'. " if tech else "Use one fresh familiar everyday situation, no technology vocabulary, and never use 'mixed bag', 'scrapbook', 'watching paint dry', 'watching grass grow', 'nothing happens', 'X is like Y, except', or 'the only thing'. ")
        + "Never claim that something is absent, still, unchanged, loading, or malfunctioning unless the evidence explicitly proves it. "
        + "Preserve exact subject-action-object-location relationships; never relocate an action onto another visible object or surface. "
        + "Never invent identity, speech, brands, intent, backstory, audience, profession, or unseen events. "
        "Keep it to one sentence of 24-42 words. Return ONLY JSON: {\"caption\":\"...\"}.\n\nVERIFIED EVIDENCE:\n"
        + evidence + "\n\nCURRENT CANDIDATE:\n" + candidate
    )
    result = _json_object(await _ask(client, [{"type": "text", "text": prompt}], 300))
    caption = result.get("caption")
    return caption.strip() if isinstance(caption, str) and caption.strip() else candidate


async def _rewrite_sarcastic(
    client: httpx.AsyncClient, evidence: str, candidate: str
) -> str:
    prompt = (
        "You are editing ONE sarcastic video caption. Rewrite it using only the verified evidence below. "
        "Produce dry, intelligent, lightly mocking irony, not loud comedy. Start with one precise verified "
        "subject, action, contrast, or montage transition, then use understated praise or restrained "
        "overstatement, and finish with a short sting tied to that same evidence. The sting must expose a "
        "verified mismatch, abrupt juxtaposition, or disproportion; generic praise alone is not sarcasm. "
        "For a montage, target the contrast between its verified scenes instead of merely listing them. For "
        "a simple scene, contrast the ordinary visible action with the grand importance assigned to it. The "
        "sentence must still work as an accurate caption when the irony is removed. Never use technology jargon. Never use the "
        "stale openings 'thrilling footage', 'thrilling tour', 'exhilarating experience', 'we are treated to', "
        "'apparently', or 'because apparently'. Do not mention frames, sampling, prompts, models, analysis, "
        "or processing. Do not invent absence, stillness, failure, identity, intent, speech, brands, an "
        "audience, backstory, or unseen events. Preserve exact sequence and subject-action-object-location "
        "relationships. Use one sentence of 22-38 words. Return ONLY JSON: {\"caption\":\"...\"}."
        "\n\nVERIFIED EVIDENCE:\n" + evidence + "\n\nCURRENT CANDIDATE:\n" + candidate
    )
    result = _json_object(await _ask(client, [{"type": "text", "text": prompt}], 320))
    caption = result.get("caption")
    return caption.strip() if isinstance(caption, str) and caption.strip() else candidate


async def _polish_humour(
    client: httpx.AsyncClient, evidence: str, captions: dict[str, str]
) -> dict[str, str]:
    prompt = (
        "You are the final comedy editor for TWO grounded video captions. Return ONLY valid JSON with "
        "exactly these string keys: humorous_tech and humorous_non_tech. Judge each candidate independently. "
        "Keep it verbatim only if it has (1) one precise verified subject-action setup, (2) one clear comparison "
        "appropriate to its style, and (3) a distinct, clearly figurative comic consequence tied to that setup. "
        "Otherwise rewrite it. Never force an unrelated second visible detail into a joke. The tech caption must use "
        "exactly one natural technology comparison. The non-tech caption must contain no technology vocabulary "
        "and use one familiar everyday situation. Never use 'X feels/is like Y, except', 'the only thing', a "
        "loose jargon list, or a generic description with no comic turn. Never invent literal absence, stillness, "
        "failure, identity, intent, speech, brands, backstory, viewer circumstances, or unseen events. Preserve "
        "exact subject-action-object-location relationships; never relocate an action onto another visible object "
        "or surface. Facts listed separately must remain parallel facts; never say that one approaches, passes, "
        "touches, or affects another unless the evidence explicitly states that relationship. Preserve montage order when "
        "relevant. Each caption must be one sentence of 24-42 words.\n\nVERIFIED EVIDENCE:\n"
        + evidence
        + "\n\nCANDIDATES:\n"
        + json.dumps({
            "humorous_tech": captions["humorous_tech"],
            "humorous_non_tech": captions["humorous_non_tech"],
        })
    )
    value = _json_object(await _ask(client, [{"type": "text", "text": prompt}], 600))
    return {
        style: value[style].strip() if isinstance(value.get(style), str) and value[style].strip()
        else captions[style]
        for style in ("humorous_tech", "humorous_non_tech")
    }


async def _verify(client: httpx.AsyncClient, evidence: str, captions: dict[str, str], styles: list[str]) -> dict[str, str]:
    prompt = (
        "You are the final grounded-caption verifier. Compare the candidate captions with the verified "
        "visual evidence for ONE video. Return ONLY valid JSON with exactly these string keys: formal, "
        "sarcastic, humorous_tech, humorous_non_tech. If every literal claim is supported, copy that "
        "candidate VERBATIM. Otherwise change ONLY the smallest risky clause; preserve supported visual "
        "nouns, actions, sequence details, style, humour and length. Never summarize or simplify a "
        "detailed caption. Remove claims about frames, sampling, prompts, models or analysis. A tech or "
        "everyday comparison may remain only as a figurative joke anchored to a visible fact. Preserve exact "
        "subject-action-object-location relationships. Facts listed separately are not interacting: remove any "
        "claim that one approaches, passes, touches, or affects another unless the evidence explicitly states it. "
        "When uncertain, remove only the unsupported relationship and keep the joke.\n\n"
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


def _edit_distance(left: str, right: str) -> int:
    a, b = left.lower(), right.lower()
    row = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, 1):
        diagonal = row[0]
        row[0] = index_a
        for index_b, char_b in enumerate(b, 1):
            previous = row[index_b]
            row[index_b] = min(
                row[index_b] + 1,
                row[index_b - 1] + 1,
                diagonal + (char_a != char_b),
            )
            diagonal = previous
    return row[-1]


def _has_lexical_corruption(value: str, evidence: str = "") -> bool:
    # Uppercase and mixed-case tokens can be legitimate OCR (for example the
    # literal ERROR shown on a display). Only reject such a token when it is
    # absent from the verified evidence; generated letter runs remain caught.
    for match in re.finditer(r"\b(?:[a-z]+[A-Z][A-Za-z]*|[A-Z]{4,}|totalest)\b", value):
        token = match.group(0)
        if token.lower() == "totalest":
            return True
        if not re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", evidence, re.IGNORECASE):
            return True
    if re.search(r"\b([A-Za-z]{3,})\b(?:\s+\1\b)+", value, re.IGNORECASE):
        return True
    words = [word.lower() for word in re.findall(r"[A-Za-z]+", value)]
    for left, right in zip(words, words[1:]):
        if min(len(left), len(right)) >= 5 and left != right and _edit_distance(left, right) <= 2:
            return True
    bigrams: dict[str, int] = {}
    for index, (left, right) in enumerate(zip(words, words[1:])):
        if len(left) < 4 or len(right) < 4:
            continue
        key = f"{left} {right}"
        if key in bigrams and index - bigrams[key] <= 20:
            return True
        bigrams[key] = index
    return False


def _has_unsupported_absolute(evidence: str, value: str) -> bool:
    if re.search(
        r"\b(?:permanent(?:ly)?|never moves?|refuses? to move|does not move|no movement|nothing changes?)\b",
        value,
        re.IGNORECASE,
    ):
        return True
    if not re.search(r"\b(?:stationary|motionless|frozen|freezes?|unchanged)\b", value, re.IGNORECASE):
        return False
    return not re.search(
        r"\b(?:stationary|motionless|remains? still|does not move|no movement|unchanged)\b",
        evidence,
        re.IGNORECASE,
    )


def _has_caption_quality_risk(evidence: str, value: str) -> bool:
    return _has_lexical_corruption(value, evidence) or _has_unsupported_absolute(evidence, value)


def _safe_caption(style: str) -> str:
    return {
        "formal": "The video presents a sequence of visible subjects, actions, and settings.",
        "sarcastic": "The visible sequence gives its ordinary events a remarkably serious presentation, ensuring that no modest moment escapes the full documentary treatment.",
        "humorous_tech": "The visible sequence unfolds like a carefully queued system update, giving each grounded scene its own turn before the final result.",
        "humorous_non_tech": "The visible sequence arrives like a family photo album passed around the table, giving each grounded moment its own little entrance.",
    }[style]


async def _repair_caption_quality(
    client: httpx.AsyncClient,
    evidence: str,
    captions: dict[str, str],
    styles: list[str],
) -> dict[str, str]:
    affected = [style for style in styles if _has_caption_quality_risk(evidence, captions[style])]
    if not affected:
        return captions
    log.warning("detected caption wording or grounding risk in styles: %s", ", ".join(affected))
    prompt = (
        "You are a strict copy editor repairing quality defects in grounded video captions. Return ONLY "
        "valid JSON with exactly these string keys: formal, sarcastic, humorous_tech, humorous_non_tech. "
        "Preserve each caption's supported facts, sequence, style, comparison, and intended joke. Change only "
        "malformed words, accidental letter runs, duplicated fragments, and broken grammar. Also remove or "
        "replace absolute claims such as stationary, frozen, permanent, never moving, or unchanged when the "
        "verified evidence does not explicitly support them. If the evidence contains motion, the repaired joke "
        "must use the visible motion or pace rather than a freeze or stillness metaphor. Do not add, remove, or "
        "reinterpret factual claims. Do not add commentary.\n\nVERIFIED EVIDENCE:\n" + evidence
        + "\n\nCAPTIONS TO CHECK:\n" + json.dumps(captions)
    )
    try:
        captions = _captions(
            _json_object(await _ask(client, [{"type": "text", "text": prompt}], 800)), styles
        )
    except (httpx.HTTPError, ValueError, KeyError):
        log.warning("combined lexical repair unavailable; retrying affected styles")
    affected = [style for style in styles if _has_caption_quality_risk(evidence, captions[style])]

    async def repair_one(style: str) -> str:
        prompt = (
            f"Repair ONLY the quality defect in this {style} video caption. Preserve every supported fact, "
            "the style, and the joke. Remove malformed text and any claim of stillness, freezing, permanence, "
            "no movement, or no change that is unsupported by the evidence. When motion is verified, anchor the "
            "repaired joke in that motion or its pace. Do not add facts. Return ONLY JSON: {\"caption\":\"...\"}."
            "\n\nEVIDENCE:\n" + evidence + "\n\nCAPTION:\n" + captions[style]
        )
        result = _json_object(await _ask(client, [{"type": "text", "text": prompt}], 320))
        caption = result.get("caption")
        return caption.strip() if isinstance(caption, str) and caption.strip() else captions[style]

    if affected:
        retries = await asyncio.gather(*(repair_one(style) for style in affected), return_exceptions=True)
        for style, repaired in zip(affected, retries):
            if isinstance(repaired, str):
                captions[style] = repaired
    for style in styles:
        if _has_caption_quality_risk(evidence, captions[style]):
            log.error("caption quality risk persisted for %s; using safe complete fallback", style)
            captions[style] = _safe_caption(style)
    return _captions(captions, styles)


async def caption_demo(video_url: str, styles: list[str]) -> dict[str, str]:
    """Run the same Gemma 4 evidence/write/rewrite/verify flow as docs/demo.html."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        video = await P._download(video_url, workdir / "clip.mp4")
        frames = P._extract_keyframes(video, workdir, FRAME_COUNT, FRAME_EDGE)
        async with httpx.AsyncClient(timeout=httpx.Timeout(150.0)) as client:
            video_segments_task = asyncio.create_task(
                asyncio.to_thread(_compress_video_segments, video, workdir)
            )

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
            video_segments = await video_segments_task
            segment_results = await asyncio.gather(
                *(
                    _direct_video_evidence(
                        client,
                        video_b64,
                        record,
                        segment_index,
                        segment_start,
                        len(video_segments),
                    )
                    for segment_index, segment_start, video_b64 in video_segments
                )
            ) if video_segments else []
            video_facts: list[str] = []
            for facts in segment_results:
                for fact in facts:
                    if fact not in video_facts:
                        video_facts.append(fact)
            grounded_record = record
            if video_facts:
                grounded_record += (
                    "\n\nDIRECT GEMMA 4 VIDEO EVIDENCE (use only when consistent with the frames):\n- "
                    + "\n- ".join(video_facts)
                )
            captions = await _write(client, grounded_record, styles)
            rewrites = await asyncio.gather(
                _rewrite_sarcastic(client, grounded_record, captions["sarcastic"]),
                _rewrite_humour(client, grounded_record, captions["humorous_tech"], "humorous_tech"),
                _rewrite_humour(client, grounded_record, captions["humorous_non_tech"], "humorous_non_tech"),
                return_exceptions=True,
            )
            for style, rewrite in zip(("sarcastic", "humorous_tech", "humorous_non_tech"), rewrites):
                if isinstance(rewrite, str) and rewrite:
                    captions[style] = rewrite
            try:
                captions.update(await _polish_humour(client, grounded_record, captions))
            except (httpx.HTTPError, ValueError, KeyError):
                log.warning("V18 humour quality pass unavailable; keeping grounded rewrites")
            captions = await _verify(client, grounded_record, captions, styles)
            return await _repair_caption_quality(client, grounded_record, captions, styles)
