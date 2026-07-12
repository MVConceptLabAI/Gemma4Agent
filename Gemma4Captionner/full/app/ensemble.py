"""Ensemble captioning engine for the container.

Multiple frontier vision models each list every visible detail from the sampled
frames, then one writer cross-references all lists and writes the four styled
captions. Reuses the pipeline's download + frame extraction. Selected with
CAPTION_ENGINE=ensemble.

Measured (vision audit, 3 demo clips): 209 correct details / 12 captions vs
~78-122 for any single model, 0 safety issues. Cross-model agreement recovers
detail no single model gets right (e.g. a small street sign).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from app import pipeline as P

log = logging.getLogger("track2.ensemble")


def _strict_env_bool(name: str, default: bool = False) -> bool:
    """Parse an explicit environment boolean without enabling on typos."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"", "0", "false", "off", "no"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/true/on/yes or 0/false/off/no"
    )


OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OBSERVERS = [m.strip() for m in os.environ.get(
    "ENSEMBLE_OBSERVERS",
    "openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.5",
).split(",") if m.strip()]
WRITER = os.environ.get("ENSEMBLE_WRITER", "anthropic/claude-opus-4.5")
# Leaderboard hedge: an unknown judge may reward concise captions on style-match.
# ENSEMBLE_CONCISE=1 keeps the verified-detail advantage but caps each caption to
# 2-3 dense sentences instead of a long paragraph.
CONCISE = os.environ.get("ENSEMBLE_CONCISE", "0") != "0"

# Tone-only few-shot exemplars on deliberately UNRELATED content. Competitor-
# measured (+0.07 style, +0.11 acc for them); our A/B on 4 hard official clips:
# FINAL 0.944 -> 0.967. The writer hears the voice, no content can leak.
EXEMPLARS = os.environ.get("STYLE_EXEMPLARS", "0") != "0"
# 4th observer that watches the REAL VIDEO (compressed), not sampled frames -
# catches temporal actions and changes frames miss. The Track 2 leader feeds
# Gemini actual video; Gemini via OpenRouter accepts data:video/mp4 input.
VIDEO_OBSERVER = os.environ.get("VIDEO_OBSERVER", "")  # e.g. google/gemini-3.1-pro-preview
# Some writers (Gemini) default to terse captions; long rich ones score better
# with the official judge (measured 0.89 long vs 0.84 concise). Optional hint.
WRITER_LENGTH_HINT = os.environ.get("WRITER_LENGTH_HINT", "")
# Severe 2-axis panel diagnosis: accuracy=0.79 is the weak axis (style=0.92),
# the writer invents specifics (fake collars, buses, manicures). This flips
# grounding from "rich+vivid" to "only what was observed", and drops writer
# temperature to curb invention.
STRICT_GROUNDING = os.environ.get("STRICT_GROUNDING", "0") != "0"
WRITER_TEMP = float(os.environ.get("WRITER_TEMP", "0.5"))
# Causal ablation: optional creative-style discipline. Off by default so the
# existing writer system remains byte-identical unless explicitly enabled.
CREATIVE_DISCIPLINE = _strict_env_bool("CREATIVE_DISCIPLINE")
# W4 causal ablation: keep the v38 observation/factual spine intact, but let
# four independent writer calls focus on one requested style each. Off by
# default so the v38 common-writer path remains unchanged unless selected.
W4_STYLE_SPLIT = _strict_env_bool("W4_STYLE_SPLIT")
_WRITER_TOTAL_MAX_TOKENS = 3000
_STYLE_WRITER_STYLES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)
_GROUNDING_RULE = (
    "\n\nSTRICT GROUNDING + MAX COVERAGE (the judge rewards rich CORRECT detail): every "
    "concrete noun, colour, count, vehicle/animal/object TYPE, action, and piece of text "
    "you write MUST appear in at least one observation list above. Do NOT invent "
    "motivations, greetings, clothing, jewelry, breeds, vehicle types, or signage. "
    "At the same time, COVER AS MANY well-supported observations as possible - subjects, "
    "actions, setting, background, lighting, motion: a caption that omits observed key "
    "elements loses as many points as one that invents them. Replace every invented "
    "specific with a REAL one from the lists, never by deleting richness. Before "
    "returning, re-read each caption: remove unsupported specifics AND add any important "
    "observed element still missing."
)
_EXEMPLAR_BLOCK = (
    "\n\nTONE EXAMPLES - these describe DIFFERENT videos; copy the VOICE, never the content:\n"
    'formal: "A commuter train crosses an elevated bridge at dusk, its lit windows reflected '
    'in the river below as traffic passes along the embankment road."\n'
    'sarcastic: "Ah yes, a dog has caught a frisbee mid-air - truly the pinnacle of athletic '
    'achievement, and judging by that tail, nobody has ever been prouder of anything."\n'
    'humorous_tech: "This golden retriever executes a flawless mid-air catch - a zero-downtime '
    'deployment of pure enthusiasm, with tail-wag telemetry reporting all systems nominal."\n'
    "humorous_non_tech: \"A golden retriever catches the frisbee like it's auditioning for its "
    'own sports documentary, then victory-laps the yard as if the neighbors paid admission."\n'
    "Write as if you personally watched the clip."
)

_CONCISE_RULE = (
    " LENGTH: write each caption as 2-3 dense sentences (about 40-60 words) that "
    "pack the strongest verified details - vivid and specific, not a long paragraph."
)

_CREATIVE_DISCIPLINE_RULE = (
    "\n\nCREATIVE DISCIPLINE: Creative humor is framing only; preserve literal scene "
    "claims. For every creative caption, never assign an unseen profession, intent, "
    "backstory, future action, or off-screen event. For humorous_tech, tech terms must "
    "be explicit similes or metaphors, and every caption must include at least one "
    "natural, visible-scene-tied marker chosen from API, latency, cache, runtime, server, "
    "pipeline, or scheduler; vary the marker across clips and never force a cliche. Other "
    "precise terms such as throughput may accompany that marker when relevant. Never turn "
    "a person into a developer, and never turn typing into commits or code. Use at most 2 "
    "metaphor or punchline devices per caption. Never "
    'open with "Behold" or "Ah yes". Avoid repeating API, endpoint, deployment, or '
    "zero-latency patterns. Apply these discipline rules only to sarcastic, "
    "humorous_tech, and humorous_non_tech. This discipline adds no length limit: "
    "preserve every already-active length instruction exactly, and leave formal governed "
    "solely by the pre-existing formal rules."
)

OBSERVE_SYSTEM = (
    "You are a meticulous visual analyst. You see frames sampled in order from ONE short "
    "video clip. Each frame carries a small overlay banner (frame number, timestamp, "
    "duration): use it to ground WHEN things happen, but NEVER describe the banner "
    "itself as scene content. Return a JSON array of SHORT strings, one per concrete detail you can "
    "verify across the frames. Be EXHAUSTIVE: every subject and appearance (hairstyle, "
    "clothing layers+colors, jewelry, nails, animal coat/markings, chest/paw color), every "
    "object, actions and motion (what changes, direction), setting, background structures "
    "and count, tree color, terrain, lighting, whether it is a time-lapse. Only include what "
    "is clearly visible. NEVER state race/ethnicity/skin color or eye color. Do NOT quote a "
    "sign unless the letters are unambiguous (say 'an unreadable sign'). Attribute a color "
    "only to the object it truly belongs to. Express positions in VIEWER terms only "
    "('on the left of the frame'), NEVER as the subject's own left/right. "
    "Return ONLY the JSON array."
)

WRITE_SYSTEM = (
    "You are a world-class video-captioning writer. Several expert vision models each listed "
    "what they saw in ONE clip; you receive all lists. Cross-reference them: a detail reported "
    "by 2+ models is high-confidence - use those freely. A detail from only ONE model is "
    "UNRELIABLE: include it ONLY if it is generic and safe; DROP any single-model SPECIFIC "
    "claim (an exact color, a brand/logo, a count, sign/text, or a left/right or foreground/"
    "background placement) unless another model agrees. When two models conflict, omit the "
    "point. A wrong detail costs far more than a missing one - when in doubt, leave it out. "
    "NEVER add anything no model reported. Write four captions of the SAME "
    "scene, one per style, richly detailed and vivid; do not state race/skin/eye color, do not "
    "quote an unreadable sign, attribute colors correctly. Only assert lighting effects (light streaks, glows, headlight/taillight trails) when unmistakably visible - never as a time-lapse cliche. Positions: use viewer terms "
    "('on the left of the frame'), NEVER the subject's own left/right ('to her left'). Styles: formal = professional, "
    "objective, factual, no jokes/exclamations/1st-2nd person; sarcastic = dry ironic wit "
    "with ZERO technology words (no model, server, cache, commit, runtime, API, deploy, "
    "pipeline, code, bug, latency); humorous_tech = clever tech metaphors (API, latency, "
    "cache, pipeline, runtime, server) tied to visible things; humorous_non_tech = warm "
    "everyday humor with ZERO technology words. Return STRICT JSON only: "
    '{"formal":"...","sarcastic":"...","humorous_tech":"...","humorous_non_tech":"..."}'
)


def _writer_system() -> str:
    """Build the writer prompt, preserving the legacy bytes when ablations are off."""
    return (
        WRITE_SYSTEM
        + (_CONCISE_RULE if CONCISE else "")
        + ((" " + WRITER_LENGTH_HINT) if WRITER_LENGTH_HINT else "")
        + (_GROUNDING_RULE if STRICT_GROUNDING else "")
        + (_EXEMPLAR_BLOCK if EXEMPLARS else "")
        + (_CREATIVE_DISCIPLINE_RULE if CREATIVE_DISCIPLINE else "")
    )


def _style_writer_system(style: str, base_system: str) -> str:
    """Append only the per-style output contract to the exact v38 prompt."""
    if style not in _STYLE_WRITER_STYLES:
        raise ValueError(f"unsupported writer style: {style}")
    return (
        base_system
        + "\n\nSTYLE-SPECIFIC OUTPUT OVERRIDE: For this call, produce only the "
        + f'"{style}" caption while applying every factual and style rule above. '
        + "Return STRICT JSON only: {\"caption\":\"...\"}"
    )


async def _call(client: httpx.AsyncClient, model: str, system: str, content: Any,
                max_tokens: int, temperature: float = 0.5) -> str:
    r = await client.post(
        f"{OR_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": temperature, "max_tokens": max_tokens},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _frames_content(frames: list[Path]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": "Frames in order:"}]
    for fp in frames:
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def _parse_list(text: str) -> list[str]:
    s = text[text.find("["): text.rfind("]") + 1]
    try:
        return [str(x).strip() for x in json.loads(s) if str(x).strip()]
    except Exception:  # noqa: BLE001
        return [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]


def _parse_obj(text: str) -> dict:
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def _compress_for_video_observer(video: Path, workdir: Path) -> str | None:
    """Re-encode to a small MP4 and return base64, or None if too large/fails."""
    import subprocess
    out = workdir / "obs_video.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
             "-vf", "scale=640:-2", "-r", "4", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "32", "-an", str(out)],
            check=True, timeout=90,
        )
        raw = out.read_bytes()
        if len(raw) > 14_000_000:  # keep the request well under provider caps
            return None
        return base64.b64encode(raw).decode("ascii")
    except Exception as e:  # noqa: BLE001
        log.warning("video-observer compression failed: %s", e)
        return None


async def caption_ensemble_frames(
    frames: list[Path], styles: list[str], video_b64: str | None = None
) -> dict[str, str]:
    """Run the observe->cross-reference->write ensemble on already-extracted frames."""
    content = _frames_content(frames)
    async with httpx.AsyncClient(timeout=httpx.Timeout(240.0)) as client:
        async def observe(model: str) -> tuple[str, list[str]]:
            try:
                return model, _parse_list(await _call(client, model, OBSERVE_SYSTEM, content, 4000))
            except Exception as e:  # noqa: BLE001
                log.warning("observer %s failed: %s", model, e)
                return model, []

        async def observe_video(model: str) -> tuple[str, list[str]]:
            vid_content = [
                {"type": "text", "text": "The full video clip:"},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            ]
            try:
                return f"{model} (video)", _parse_list(
                    await _call(client, model, OBSERVE_SYSTEM, vid_content, 4000))
            except Exception as e:  # noqa: BLE001
                log.warning("video observer %s failed: %s", model, e)
                return model, []

        observers = [observe(m) for m in OBSERVERS]
        if VIDEO_OBSERVER and video_b64:
            observers.append(observe_video(VIDEO_OBSERVER))
        obs = await asyncio.gather(*observers)
        blocks = [
            f"### {m.split('/')[-1]} ({len(d)} details):\n" + "\n".join(f"- {x}" for x in d)
            for m, d in obs if d
        ]
        if not blocks:
            raise RuntimeError("all ensemble observers failed")
        write_content = (
            "Independent observation lists from several vision models for ONE clip. "
            "Cross-reference and write the four captions.\n\n" + "\n\n".join(blocks)
        )
        system = _writer_system()
        if W4_STYLE_SPLIT:
            per_style_tokens = _WRITER_TOTAL_MAX_TOKENS // len(_STYLE_WRITER_STYLES)

            async def write_style(style: str) -> tuple[str, str]:
                style_system = _style_writer_system(style, system)
                caption = ""
                for attempt in range(2):
                    try:
                        raw = await _call(
                            client,
                            WRITER,
                            style_system,
                            write_content,
                            per_style_tokens,
                            temperature=WRITER_TEMP,
                        )
                        caption = str(_parse_obj(raw).get("caption", ""))
                        break
                    except (
                        httpx.HTTPStatusError,
                        httpx.TransportError,
                        ValueError,
                    ) as e:
                        if attempt == 1:
                            raise
                        log.warning(
                            "writer %s attempt 1 failed (%s), retrying once", style, e
                        )
                        await asyncio.sleep(2)
                return style, caption

            caps = dict(
                await asyncio.gather(
                    *(write_style(style) for style in _STYLE_WRITER_STYLES)
                )
            )
        else:
            # 3000 tokens: 4 rich captions can exceed 2000 and a mid-JSON truncation
            # discards the whole ensemble. One retry on transient writer failure -
            # cheaper than the alternative (a full 150s single-model pipeline rerun).
            raw = ""
            for attempt in range(2):
                try:
                    raw = await _call(
                        client,
                        WRITER,
                        system,
                        write_content,
                        _WRITER_TOTAL_MAX_TOKENS,
                        temperature=WRITER_TEMP,
                    )
                    break
                except (httpx.HTTPStatusError, httpx.TransportError) as e:
                    if attempt == 1:
                        raise
                    log.warning("writer attempt 1 failed (%s), retrying once", e)
                    await asyncio.sleep(2)
            caps = _parse_obj(raw)
    return {k: str(caps.get(k, "")) for k in styles}


async def caption_ensemble(video_url: str, styles: list[str]) -> dict[str, str]:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        vp = await P._download(video_url, wd / "clip.mp4")
        frames = P._extract_keyframes(vp, wd, P.NUM_FRAMES, P.FRAME_MAX_EDGE)
        video_b64 = None
        if VIDEO_OBSERVER:
            video_b64 = await asyncio.to_thread(_compress_for_video_observer, vp, wd)
        return await caption_ensemble_frames(frames, styles, video_b64)
