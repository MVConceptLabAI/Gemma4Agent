# Track 2 v38 Leader-Parity Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan one task at a time. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement reversible v38 causal profiles that restore factual recall and style strength while preserving the controlled C0 source fallback, immutable R0 whole-run rollback, and Track 2 contract.

**Architecture:** Preserve an immutable v36 reference and implement independent experimental profiles behind environment switches. Controlled profiles change frame geometry, description richness, caption length, joint generation, model choice, and repair one lever at a time; only the measured winning combination becomes the production candidate. The DescribeX OCI profile is explicitly an unverified public-image hypothesis, the official-repository profile is separate, and MiniMax has faithful and hardened profiles.

**Tech Stack:** Python 3.11, asyncio, httpx, Pydantic, Pillow, FFmpeg/ffprobe, OpenAI-compatible Fireworks endpoints, Docker Buildx, plain assertion-based test scripts.

---

## Non-negotiable guardrails

- Do not submit or retag the public competition image while v37 is still unevaluated.
- Pin the official reference to ghcr.io/theskygold/track2-captioner@sha256:161efc8b098a6a46f395f01fb83ce7c41a9e71c61a51205b59934393bac5f19d and assert its linux/amd64 child sha256:1b39f65c4b99642a318b353a2e0281c4d5ddb4d346510766e40c27ae5ed0ac07.
- Do not call a source reconstruction exact v36. R0 is the immutable image; C0 is a controlled source reconstruction from commit 283ce7f.
- Do not use a same-model verifier or a generated fact list as the sole factual judge.
- Do not print, commit, or inspect provider secret values.
- Keep every new behavior behind V38_PROFILE. The value c0 must preserve the reconstructed v36 path; every other value names one documented ablation.
- Stop if the paired evaluation plan does not clear every gate in the companion plan.

### Task 0: Freeze R0 and create the controlled C0 worktree

**Files:**
- Create: scripts/baseline_manifest.py
- Create: scripts/test_baseline_manifest.py
- Create: docs/evals/v36-baseline-manifest.json

- [ ] Write a failing test that rejects a mutable tag, an unknown/unknown attestation child, a non-amd64 image child, or a mismatched digest.

    EXPECTED_INDEX = "sha256:161efc8b098a6a46f395f01fb83ce7c41a9e71c61a51205b59934393bac5f19d"
    EXPECTED_AMD64 = "sha256:1b39f65c4b99642a318b353a2e0281c4d5ddb4d346510766e40c27ae5ed0ac07"

    def fixture_manifest(
        index_digest: str,
        child_digest: str,
        os_name: str,
        architecture: str,
    ) -> dict[str, object]:
        return {
            "index_digest": index_digest,
            "manifests": [{
                "digest": child_digest,
                "platform": {"os": os_name, "architecture": architecture},
            }],
        }

    def test_v36_manifest_requires_exact_index_and_amd64_child() -> None:
        manifest = fixture_manifest(EXPECTED_INDEX, EXPECTED_AMD64, "linux", "amd64")
        assert validate_v36_manifest(manifest) == EXPECTED_AMD64

    def test_v36_manifest_rejects_attestation_child() -> None:
        manifest = fixture_manifest(EXPECTED_INDEX, "sha256:09b370", "unknown", "unknown")
        try:
            validate_v36_manifest(manifest)
        except ValueError:
            return
        raise AssertionError("attestation child was accepted")

- [ ] Run the test.

    python scripts/test_baseline_manifest.py

    Expected: FAIL because scripts/baseline_manifest.py does not exist.

- [ ] Implement registry-manifest validation without reading image environment values.

    def validate_v36_manifest(payload: dict[str, object]) -> str:
        if payload["index_digest"] != EXPECTED_INDEX:
            raise ValueError("v36 index digest mismatch")
        children = payload.get("manifests", [])
        matches = [
            child for child in children
            if child.get("platform") == {"os": "linux", "architecture": "amd64"}
        ]
        if len(matches) != 1 or matches[0].get("digest") != EXPECTED_AMD64:
            raise ValueError("v36 linux/amd64 child mismatch")
        return EXPECTED_AMD64

- [ ] Save only digests, platform, source commit 283ce7f, displayed_historical_score 0.9133, resubmission/evaluation timestamps, and score_attribution in docs/evals/v36-baseline-manifest.json. Because Lablab does not expose the evaluated image digest in the row, set attributed_digest to null and label the v36 association timestamp-inferred, not cryptographically proven. Never serialize Config.Env.

- [ ] At execution time, use the worktree skill to create codex/v38-causal-eval from commit 283ce7f unless the resolved v37 score is above the leader; record any different base explicitly.

- [ ] Verify the controlled C0 profile has CAPTION_ENGINE=ensemble, ten frames, 896-pixel maximum edge, and MAX_CAPTION_CHARS=1600. R0 remains the whole-run rollback; C0 is a source profile selected at task initialization for controlled ablations, never a late fallback.

- [ ] Run the test and commit.

    python scripts/test_baseline_manifest.py
    git add scripts/baseline_manifest.py scripts/test_baseline_manifest.py docs/evals/v36-baseline-manifest.json
    git commit -m "test: pin immutable v36 reference"

    Expected: PASS.

### Task 1: Add exact and endpoint-aware frame profiles

**Files:**
- Modify: app/pipeline.py
- Create: scripts/test_leader_parity.py

- [ ] Add failing pure-function tests for the exact public DescribeX OCI geometry and the independent endpoint-aware variant.

    from app.pipeline import _ratio_timestamps

    def test_oci_exact_timestamps_match_observed_code() -> None:
        assert _ratio_timestamps(100.0, "describex_oci_hypothesis") == [
            5.0, 16.25, 27.5, 38.75, 50.0, 61.25, 72.5, 83.75
        ]

    def test_endpoint_variant_includes_near_end_evidence() -> None:
        assert _ratio_timestamps(100.0, "endpoint_aware") == [
            5.0, 17.857, 30.714, 43.571, 56.429, 69.286, 82.143, 95.0
        ]

    def test_official_repo_sampler_matches_sixty_frame_case() -> None:
        assert _official_repo_indices(60) == [
            0, 3, 7, 11, 15, 18, 22, 26, 30, 33, 37, 41, 45, 48, 52, 59
        ]

- [ ] Run the focused test and confirm the import fails.

    python scripts/test_leader_parity.py

    Expected: FAIL because _ratio_timestamps does not exist.

- [ ] Implement a pure timestamp helper and a dedicated extractor in app/pipeline.py.

    FRAME_PROFILES = {
        "describex_oci_hypothesis": tuple(0.05 + index * (0.90 / 8) for index in range(8)),
        "endpoint_aware": tuple(0.05 + index * (0.90 / 7) for index in range(8)),
    }

    def _ratio_timestamps(
        duration: float,
        profile: str,
    ) -> list[float]:
        if duration <= 0:
            raise ValueError("video duration must be positive")
        try:
            ratios = FRAME_PROFILES[profile]
        except KeyError as error:
            raise ValueError(f"unknown frame profile: {profile}") from error
        return [round(float(duration) * ratio, 3) for ratio in ratios]

    def _official_repo_indices(total_frames: int, target: int = 16) -> list[int]:
        if total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if total_frames <= target:
            return list(range(total_frames))
        step = total_frames / target
        indices = [int(index * step) for index in range(target)]
        indices[0] = 0
        indices[-1] = total_frames - 1
        return sorted(set(indices))

    def _extract_official_repo_frames(
        video: Path,
        workdir: Path,
        video_fps: float,
        duration: float,
    ) -> list[Path]:
        extraction_fps = min(video_fps, 60.0 / duration)
        extracted = _ffmpeg_fps_extract(
            video=video,
            workdir=workdir,
            fps=extraction_fps,
            qscale=2,
            total_timeout_s=12.0,
        )
        return [extracted[index] for index in _official_repo_indices(len(extracted))]

    def _ffmpeg_fps_extract(
        video: Path,
        workdir: Path,
        fps: float,
        qscale: int,
        total_timeout_s: float,
    ) -> list[Path]:
        output_dir = workdir / "official_repo"
        output_dir.mkdir(exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-vf", f"fps={fps:.8f}",
            "-q:v", str(qscale), str(output_dir / "frame_%03d.jpg"),
        ], check=True, timeout=total_timeout_s)
        frames = sorted(output_dir.glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError("official-repository extraction produced no frames")
        return frames

    def _extract_ratio_frames(
        video: Path,
        workdir: Path,
        profile: str,
        max_edge: int = 768,
    ) -> list[Path]:
        duration = _ffprobe_duration(video)
        timestamps = _ratio_timestamps(duration, profile)
        return _extract_frames_at_timestamps(
            video=video,
            workdir=workdir,
            timestamps=timestamps,
            max_edge=max_edge,
            jpeg_quality=85,
        )

    def _extract_frames_at_timestamps(
        video: Path,
        workdir: Path,
        timestamps: Sequence[float],
        max_edge: int,
        jpeg_quality: int,
    ) -> list[Path]:
        frames: list[Path] = []
        deadline = time.monotonic() + 12.0
        qscale = max(2, min(31, round((100 - jpeg_quality) * 0.29 + 2)))
        for index, timestamp in enumerate(timestamps, start=1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("12-second frame extraction budget exhausted")
            target = workdir / f"leader_{index:02d}.jpg"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{timestamp:.3f}", "-i", str(video),
                "-frames:v", "1",
                "-vf", f"scale='min({max_edge},iw)':'min({max_edge},ih)':force_original_aspect_ratio=decrease",
                "-q:v", str(qscale), str(target),
            ], check=False, timeout=min(3.0, remaining))
            if target.exists() and target.stat().st_size > 0:
                frames.append(target)
        if len(frames) != 8:
            raise RuntimeError(f"only {len(frames)} leader-parity frames extracted")
        return frames

- [ ] Reuse the existing FFmpeg error handling, but do not route this helper through scene detection or duration divided by n plus one.

- [ ] Add tests for zero duration, invalid ffprobe output, a 30-second clip, monotonic ordering, exactly eight ratio positions, one missing frame, FFmpeg timeout, and values within the media duration. The official-repository profile separately extracts at fps=min(video_fps, 60/duration), caps near 60 JPEGs at qscale 2, then samples floor(i*N/16) with first=0 and last=N-1.

- [ ] Run the focused tests.

    python scripts/test_leader_parity.py

    Expected: PASS for all sampling tests.

- [ ] Commit the isolated sampling change.

    git add app/pipeline.py scripts/test_leader_parity.py
    git commit -m "feat: add endpoint-aware leader parity sampling"

### Task 2: Build strict parsing and narrow caption validation

**Files:**
- Create: app/leader_parity.py
- Modify: scripts/test_leader_parity.py

**Frozen R0 length evidence:** the exact v36 image produced 15 freshly executed
clips / 60 captions with formal min/median/mean/max of 125/167/166.2/202 words,
sarcastic 79/105/104.7/131, humorous_tech 83/104/103.6/130, and
humorous_non_tech 86/104/106.8/132. The official score attached to this
behavior is 0.9133. Consequently, this validator is a broad corruption/style
guard, not a concise-output optimizer. Any shorter length policy remains an
isolated L1 ablation and may not silently constrain every v38 arm.

- [ ] Add failing tests covering:
  - fenced JSON and prose around JSON;
  - alias keys mapped only to requested styles;
  - missing, empty, non-string, and extra keys;
  - formal: 3-9 complete sentences, 80-220 words;
  - creative styles: 2-6 complete sentences, 40-150 words;
  - at most 1600 characters for every style;
  - leaked analysis or repeated fragments;
  - technical jargon rejected from humorous_non_tech;
  - a natural technology or programming reference required in humorous_tech.

- [ ] Use adversarial examples that would have passed the v37 noun-list guard, including invented unicorn, forest, and mountain-lake claims. Confirm the new validator does not pretend to prove factual entailment; it must classify these as requiring independent factual review rather than silently declaring them grounded.

- [ ] Run the tests and confirm app.leader_parity is missing.

    python scripts/test_leader_parity.py

    Expected: FAIL with ModuleNotFoundError.

- [ ] Implement these public functions in app/leader_parity.py:

    import base64
    import difflib
    import hashlib
    import json
    import re
    import time
    from collections.abc import Awaitable, Callable
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any
    from urllib.parse import urlparse

    import httpx

    from app.prompts import STYLE_PROMPTS

    @dataclass(frozen=True)
    class ChatRequest:
        model: str
        messages: list[dict[str, Any]]
        temperature: float
        max_tokens: int | None = None
        reasoning_effort: str | None = None

    InvokeFn = Callable[[ChatRequest], Awaitable[str]]

    def remaining_seconds(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    REQUESTED_STYLES = (
        "formal",
        "sarcastic",
        "humorous_tech",
        "humorous_non_tech",
    )

    NON_TECH_BANNED = (
        "algorithm", "api", "app", "binary", "bug", "cache", "cloud",
        "code", "cpu", "database", "debug", "deploy", "docker", "gpu",
        "kernel", "latency", "loop", "network", "pixel", "reboot",
        "server", "software", "stack", "token",
    )
    TECH_MARKERS = (
        "algorithm", "api", "app", "bug", "cache", "cloud", "code",
        "cpu", "database", "debug", "deploy", "gpu", "loop", "network",
        "pixel", "server", "software", "stack",
    )
    LEAK_MARKERS = (
        "analysis:", "assistant:", "chain of thought", "reasoning:",
        "<think>", "</think>", "system prompt",
    )

    def parse_json_object(text: str) -> dict[str, object]:
        decoder = json.JSONDecoder()
        objects: list[dict[str, object]] = []
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects.append(value)
        if len(objects) != 1:
            raise ValueError(f"expected one JSON object, found {len(objects)}")
        return objects[0]

    def normalize_requested_captions(
        payload: dict[str, object],
        styles: list[str],
    ) -> dict[str, str]:
        aliases = {
            "humorous-tech": "humorous_tech",
            "humorous-non-tech": "humorous_non_tech",
        }
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            canonical = aliases.get(str(key), str(key))
            if canonical in normalized:
                raise ValueError(f"caption alias collision for {canonical}")
            normalized[canonical] = value
        if set(normalized) != set(styles):
            raise ValueError("caption keys do not exactly match requested styles")
        if any(not isinstance(normalized[style], str) for style in styles):
            raise ValueError("every caption must be a string")
        return {style: str(normalized[style]).strip() for style in styles}

    def caption_violations(style: str, caption: str) -> list[str]:
        clean = " ".join(caption.split())
        words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", clean)
        word_set = {word.casefold() for word in words}
        sentences = re.findall(r"[^.!?]+[.!?]", clean)
        violations: list[str] = []
        if not clean:
            violations.append("empty")
        sentence_min, sentence_max = ((3, 9) if style == "formal" else (2, 6))
        word_min, word_max = ((80, 220) if style == "formal" else (40, 150))
        if not sentence_min <= len(sentences) <= sentence_max:
            violations.append("sentence_count")
        if not word_min <= len(words) <= word_max:
            violations.append("word_count")
        if len(clean) > 1600:
            violations.append("character_count")
        lower = clean.casefold()
        if style in {"sarcastic", "humorous_non_tech"} and word_set.intersection(NON_TECH_BANNED):
            violations.append("technical_jargon")
        if style == "humorous_tech" and not word_set.intersection(TECH_MARKERS):
            violations.append("missing_tech_reference")
        if any(term in lower for term in LEAK_MARKERS):
            violations.append("leaked_reasoning")
        trigrams = [tuple(words[index:index + 3]) for index in range(max(0, len(words) - 2))]
        if any(trigrams.count(trigram) >= 3 for trigram in set(trigrams)):
            violations.append("repeated_fragment")
        letters = [character for character in clean if character.isalpha()]
        if letters and sum(character.isascii() for character in letters) / len(letters) < 0.8:
            violations.append("non_english")
        return violations

    def validate_caption_set(
        captions: dict[str, str],
        styles: list[str],
    ) -> dict[str, list[str]]:
        failures: dict[str, list[str]] = {}
        for style in styles:
            reasons = caption_violations(style, captions.get(style, ""))
            if reasons:
                failures[style] = reasons
        for left_index, left_style in enumerate(styles):
            for right_style in styles[left_index + 1:]:
                ratio = difflib.SequenceMatcher(
                    None,
                    captions.get(left_style, "").casefold(),
                    captions.get(right_style, "").casefold(),
                ).ratio()
                if ratio >= 0.92:
                    failures.setdefault(right_style, []).append(
                        f"near_duplicate:{left_style}"
                    )
        return failures

- [ ] Keep factual entailment out of caption_violations. It validates structure and style only; factual comparison belongs to the independent evaluation plan. Add explicit tests that appearance, happily, apple, within, and details do not trigger substring matches, that alias collisions fail, and that multiple JSON objects fail.

- [ ] Run the focused tests.

    python scripts/test_leader_parity.py

    Expected: PASS for parser and validator tests.

- [ ] Commit.

    git add app/leader_parity.py scripts/test_leader_parity.py
    git commit -m "feat: validate joint leader parity captions"

### Task 3: Implement separate DescribeX repository and OCI-hypothesis description profiles

**Files:**
- Modify: app/leader_parity.py
- Modify: scripts/test_leader_parity.py

- [ ] Add failing tests for two named profiles:
  - describex_official_repo uses 16 frames, Kimi K2.6, and GPT-OSS-120B later;
  - describex_oci_hypothesis uses eight OCI-exact frames, Kimi K2.6 with K2.5 fallback, temperature 0.3, and a 600-token scene bound.

- [ ] Assert that only the OCI-hypothesis prompt requests 150-250 factual words. The official-repository prompt has seven observed sections and no invented word limit. Keep both source-observed profiles free of added frame labels, reasoning controls, camera-motion/count/OCR requirements, or new prohibition clauses; those belong only to A_HARD.

- [ ] Add a separate A_HARD prompt test that requests counts, camera motion, exact OCR or uncertainty and forbids unsupported identity, causality, and off-screen events.

- [ ] Implement:

    OFFICIAL_REPO_SCENE_SYSTEM = (
        "Analyze the representative frames in clear numbered sections. Cover: "
        "the visible venue or setting; every focal person, animal, object, their "
        "appearance, position and distinguishing features; actions, movement and "
        "interactions; indoor/outdoor context, apparent time, weather and season; "
        "mood inferred from lighting, color grade, expressions, body language and "
        "pacing; prominent colors, objects, on-screen text, graphics, overlays and "
        "transitions; and the temporal progression from first to last frame. Keep "
        "the analysis factual and neutral, with no caption, humor or opinion."
    )

    OCI_SCENE_SYSTEM = (
        "Write a thorough factual 150-250 word description of the chronological "
        "frames. Cover location type, weather, apparent time and lighting; people, "
        "animals and main objects with appearance, action and position; movement "
        "and changes over time; visible signs, text, logos and branding; mood or "
        "atmosphere; and distinctive colors, textures and patterns."
    )

    HARDENED_SCENE_SYSTEM = (
        "Describe only visible evidence in 150-250 factual English words. Include "
        "clear colors and counts, chronology, camera motion, and exact legible text; "
        "mark unreadable text uncertain. Do not infer identity, intent, cause, "
        "location name, or off-screen events."
    )

    DESCRIPTION_PROFILES = {
        "describex_official_repo": {
            "frame_count": 16,
            "temperature": 0.3,
            "max_tokens": None,
            "word_range": None,
            "reject_leaks": False,
            "system_prompt": OFFICIAL_REPO_SCENE_SYSTEM,
        },
        "describex_oci_hypothesis": {
            "frame_count": 8,
            "temperature": 0.3,
            "max_tokens": 600,
            "word_range": None,
            "reject_leaks": False,
            "system_prompt": OCI_SCENE_SYSTEM,
        },
        "A_HARD": {
            "frame_count": 8,
            "temperature": 0.3,
            "max_tokens": 600,
            "word_range": (150, 250),
            "reject_leaks": True,
            "system_prompt": HARDENED_SCENE_SYSTEM,
        },
    }

    def build_scene_payload(
        frames: list[Path],
        system_prompt: str,
    ) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [{"type": "text", "text": system_prompt}]
        for frame in frames:
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        return content

    async def describe_frames(
        frames: list[Path],
        model: str,
        profile: str,
        invoke: InvokeFn,
    ) -> str:
        parameters = DESCRIPTION_PROFILES[profile]
        if len(frames) != parameters["frame_count"]:
            raise ValueError("frame count does not match description profile")
        request = ChatRequest(
            model=model,
            messages=[{
                "role": "user",
                "content": build_scene_payload(frames, parameters["system_prompt"]),
            }],
            temperature=parameters["temperature"],
            max_tokens=parameters["max_tokens"],
        )
        text = await invoke(request)
        clean = " ".join(text.split())
        count = len(clean.split())
        word_range = parameters["word_range"]
        if not clean:
            raise ValueError("invalid canonical description")
        if parameters["reject_leaks"] and any(
            marker in clean.casefold() for marker in LEAK_MARKERS
        ):
            raise ValueError("canonical description leaked control text")
        if word_range is not None and not word_range[0] <= count <= word_range[1]:
            raise ValueError("canonical description violates profile length")
        return clean

- [ ] The official-repository and OCI-hypothesis profiles accept any non-empty provider response, exactly as observed; the OCI length is prompt-only. Only A_HARD enforces 150-250 words and leak markers. Allow one provider failover only where that observed profile defines one; never add a same-model verify cycle.

- [ ] Add fake-invoker tests for a valid response, empty response, malformed response, timeout, 429, and fallback success.

- [ ] Run tests.

    python scripts/test_leader_parity.py

    Expected: PASS with no network calls.

- [ ] Commit.

    git add app/leader_parity.py scripts/test_leader_parity.py
    git commit -m "feat: add rich canonical video description"

### Task 4: Implement joint generation and hardening as separate profiles

**Files:**
- Modify: app/leader_parity.py
- Modify: scripts/test_leader_parity.py

- [ ] Add a failing orchestration test using a fake invoke function. In joint_only, one response contains all four styles and there is no repair. In joint_hardened, if and only if one style fails, a second call requests only that style.

- [ ] Add tests proving:
  - no response_format parameter and no repair in describex_oci_hypothesis;
  - GPT-OSS-120B in describex_official_repo and GPT-OSS-20B with DeepSeek V4 Flash fallback in describex_oci_hypothesis;
  - official repository uses one user message, temperature 0.3, no max_tokens, and only 2-4 sentences;
  - OCI hypothesis uses its observed creative-writer system message plus one user message, temperature 0.7, max_tokens 1528, and 2-4 sentences/40-120 words;
  - only A_HARD adds system-role grounding rules, R0-shaped style-specific
    length targets, a 1600-character cap, and repair;
  - no repair call when all requested styles pass;
  - exactly one targeted repair when one style fails;
  - one joint retry instead of multiple repairs when zero or at least two styles fail;
  - one orchestrator-owned fallback after API, empty, malformed, or validation failure;
  - deterministic emergency captions contain every requested style;
  - captions from one task never leak into another task.

- [ ] Implement the joint writer prompt and orchestration.

    HARDENED_JOINT_STYLE_SYSTEM = (
        "Return one strict JSON object with exactly the requested style keys. "
        "Formal must be 4-7 complete English sentences and normally 120-180 words. "
        "Each creative style must be 3-5 sentences and normally 75-115 words. "
        "No caption may exceed 1600 characters. Preserve the same visible "
        "facts in every style. A joke may add a comparison, never a new subject, "
        "object, action, place, identity, or event."
    )

    HARDENED_STYLE_DEFINITIONS = {
        "formal": "Professional, objective, factual tone with no joke.",
        "sarcastic": "Dry, ironic, lightly mocking tone with no technology jargon.",
        "humorous_tech": "Funny tone with a natural programming or technology comparison.",
        "humorous_non_tech": "Funny everyday tone with no technical jargon.",
    }

    def joint_messages(description: str, styles: list[str]) -> list[dict[str, str]]:
        requested = {style: HARDENED_STYLE_DEFINITIONS[style] for style in styles}
        return [
            {"role": "system", "content": HARDENED_JOINT_STYLE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Canonical visible scene:\n{description}\n\n"
                    f"Requested styles:\n{json.dumps(requested, ensure_ascii=False)}"
                ),
            },
        ]

    def joint_messages_oci(
        description: str,
        styles: list[str],
    ) -> list[dict[str, str]]:
        definitions = {
            "formal": (
                "Professional, objective and factual, like documentary or news "
                "narration, with precise measured language."
            ),
            "sarcastic": (
                "Dry, ironic and lightly mocking; find subtle humor in the scene "
                "without being mean-spirited."
            ),
            "humorous_tech": (
                "Funny through technology, programming or software-engineering "
                "metaphors tied to the scene."
            ),
            "humorous_non_tech": (
                "Funny, relatable everyday observational humor with no technical "
                "jargon."
            ),
        }
        requested = {style: definitions[style] for style in styles}
        return [
            {
                "role": "system",
                "content": (
                    "Act as a creative caption writer and return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the video description to produce the requested styles. "
                    "Each caption must be 2-4 sentences and 40-120 words, accurately "
                    "reflect the description, and sound distinctly different. Return "
                    "one JSON object with exactly the requested keys.\n\n"
                    f"Video description:\n{description}\n\n"
                    f"Styles:\n{json.dumps(requested, ensure_ascii=False)}"
                ),
            },
        ]

    def joint_messages_from_c0(
        description: str,
        styles: list[str],
    ) -> list[dict[str, str]]:
        rules = "\n\n".join(
            f"{style}: {STYLE_PROMPTS[style][0]}" for style in styles
        )
        return [
            {
                "role": "system",
                "content": (
                    "Return one JSON object with exactly the requested keys. "
                    "Apply these existing C0 style rules without adding a new "
                    f"length policy:\n{rules}"
                ),
            },
            {"role": "user", "content": f"Visible scene facts:\n{description}"},
        ]

    def joint_messages_official_repo(
        description: str,
        styles: list[str],
    ) -> list[dict[str, str]]:
        definitions = {
            "formal": (
                "Professional, clear and informative for business, education or "
                "official communication, with precise neutral authority."
            ),
            "sarcastic": (
                "Witty, ironic and tongue-in-cheek, using dry playful observations."
            ),
            "humorous_tech": (
                "Funny for a tech-savvy audience through programming, hardware, "
                "software, algorithms, internet culture or developer analogies."
            ),
            "humorous_non_tech": (
                "Funny through relatable daily life and common experiences, with "
                "no jargon."
            ),
        }
        requested = {style: definitions[style] for style in styles}
        return [{
            "role": "user",
            "content": (
                "Write one caption for every requested style, 2-4 sentences each. "
                "Return only one JSON object with exactly the four canonical keys; "
                "each value is one string. Do not use code fences, explanations or "
                "extra text.\n\n"
                f"Scene analysis:\n{description}\n\n"
                f"Styles:\n{json.dumps(requested, ensure_ascii=False)}"
            ),
        }]

    def repair_messages(
        description: str,
        style: str,
        rejected_caption: str,
        violations: list[str],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": HARDENED_JOINT_STYLE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Canonical visible scene:\n{description}\n\n"
                    f"Rewrite only {style}: {HARDENED_STYLE_DEFINITIONS[style]}\n"
                    f"Rejected caption:\n{rejected_caption}\n"
                    f"Fix these violations: {', '.join(violations)}\n"
                    "Return only the replacement caption."
                ),
            },
        ]

    WRITER_PROFILES = {
        "J1": {
            "model": "accounts/fireworks/models/gpt-oss-20b",
            "fallback_models": ["accounts/fireworks/models/deepseek-v4-flash"],
            "temperature": 0.7,
            "max_tokens": 1528,
            "repair": False,
            "prompt": "c0_joint",
        },
        "describex_official_repo": {
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "fallback_models": [],
            "temperature": 0.3,
            "max_tokens": None,
            "repair": False,
            "prompt": "official_repo",
        },
        "describex_oci_hypothesis": {
            "model": "accounts/fireworks/models/gpt-oss-20b",
            "fallback_models": ["accounts/fireworks/models/deepseek-v4-flash"],
            "temperature": 0.7,
            "max_tokens": 1528,
            "repair": False,
            "prompt": "oci_exact",
        },
        "joint_hardened": {
            "model": "accounts/fireworks/models/gpt-oss-20b",
            "fallback_models": ["accounts/fireworks/models/deepseek-v4-flash"],
            "temperature": 0.7,
            "max_tokens": 1528,
            "repair": True,
            "prompt": "hardened",
        },
    }

    async def write_joint_captions(
        description: str,
        styles: list[str],
        profile: str,
        invoke: InvokeFn,
    ) -> dict[str, str]:
        config = WRITER_PROFILES[profile]
        last_error: Exception | None = None
        schedule = [config["model"]]
        if config["fallback_models"]:
            schedule.append(config["fallback_models"][0])
        elif config["repair"]:
            schedule.append(config["model"])
        captions: dict[str, str] | None = None
        violations: dict[str, list[str]] = {}
        try:
            async with asyncio.timeout(22):
                for attempt, candidate_model in enumerate(schedule[:2]):
                    try:
                        request = ChatRequest(
                            model=candidate_model,
                            messages=(
                                joint_messages_from_c0(description, styles)
                                if config["prompt"] == "c0_joint"
                                else joint_messages_official_repo(description, styles)
                                if config["prompt"] == "official_repo"
                                else joint_messages_oci(description, styles)
                                if config["prompt"] == "oci_exact"
                                else joint_messages(description, styles)
                            ),
                            temperature=config["temperature"],
                            max_tokens=config["max_tokens"],
                        )
                        text = await invoke(request)
                        captions = normalize_requested_captions(
                            parse_json_object(text), styles
                        )
                        if not config["repair"]:
                            return captions
                        violations = validate_caption_set(captions, styles)
                        if len(violations) <= 1:
                            break
                        if attempt == 1:
                            raise ValueError("joint rewrite remains invalid")
                    except (ValueError, httpx.HTTPError) as error:
                        last_error = error
                        continue
        except TimeoutError as error:
            last_error = error
        if captions is None:
            raise RuntimeError("joint caption generation failed") from last_error
        if len(violations) == 1:
            style, reasons = next(iter(violations.items()))
            captions[style] = await repair_one_style(
                description,
                style,
                captions[style],
                reasons,
                schedule[min(len(schedule), 2) - 1],
                invoke,
            )
        if validate_caption_set(captions, styles):
            raise RuntimeError("joint caption validation failed")
        return captions

    async def repair_one_style(
        description: str,
        style: str,
        rejected_caption: str,
        violations: list[str],
        model: str,
        invoke: InvokeFn,
    ) -> str:
        async with asyncio.timeout(8):
            text = await invoke(ChatRequest(
                model=model,
                messages=repair_messages(description, style, rejected_caption, violations),
                temperature=0.45,
                max_tokens=400,
            ))
        repaired = " ".join(text.strip().strip('"').split())
        reasons = caption_violations(style, repaired)
        if reasons:
            raise ValueError(f"targeted repair failed: {','.join(reasons)}")
        return repaired

- [ ] Keep the canonical description unchanged during repair. Repair at most one style once. If two or more styles fail, spend the single retry on one joint rewrite instead of serial repairs.

- [ ] Run tests.

    python scripts/test_leader_parity.py

    Expected: PASS and fake call counts match the assertions.

- [ ] Commit.

    git add app/leader_parity.py scripts/test_leader_parity.py
    git commit -m "feat: generate and repair four styles jointly"

### Task 5: Add faithful and hardened MiniMax native-video profiles

**Files:**
- Modify: app/leader_parity.py
- Modify: scripts/test_leader_parity.py

- [ ] Add failing tests for B0_UNIKL_EXACT: original video URL, 5-7 factual sentences, MiniMax-M3, max_tokens 5000, temperature 0.2, 25-second timeout, then one joint MiniMax-M3 style call with max_tokens 5000 and temperature 0.7. It never downloads or base64-encodes the video.

- [ ] Add tests proving B0 has no 2-4 sentence/40-120 word gate, no response_format, no verifier, and no targeted repair; it permits two retries, for three attempts total, only inside this orchestrator. Add B1_UNIKL_HARDENED tests where the sole difference is validator plus one targeted repair.

- [ ] Add tests for native-description failure falling back to the frame arm only when at least 42 seconds remain, and for file, private, loopback, link-local, localhost, and non-HTTP URLs bypassing the native arm.

- [ ] Implement:

    def native_video_is_eligible(video_url: str) -> bool:
        parsed = urlparse(video_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        hostname = parsed.hostname.casefold()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return False
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return True
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        )

    def native_description_messages(video_url: str) -> list[dict[str, object]]:
        return [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this video in 5-7 factual English sentences. Cover "
                        "the setting or location and time of day; all subjects with "
                        "colors, clothing, quantities, and distinguishing features; "
                        "readable text, logos, signs, and labels quoted exactly; the "
                        "chronological sequence of actions or events; and the mood or "
                        "atmosphere. Be literal, do not speculate, and make this "
                        "description the sole factual source for the captions."
                    ),
                },
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }]

    async def describe_native_video(
        video_url: str,
        model: str,
        invoke: InvokeFn,
    ) -> str:
        if not native_video_is_eligible(video_url):
            raise ValueError("native video requires a public HTTP URL")
        async with asyncio.timeout(25):
            text = await invoke(ChatRequest(
                model=model,
                messages=native_description_messages(video_url),
                temperature=0.2,
                max_tokens=5000,
            ))
        clean = " ".join(text.split())
        if not clean:
            raise ValueError("native description is empty")
        return clean

    UNIKL_STYLE_DEFINITIONS = {
        "formal": "Professional, objective, and factual.",
        "sarcastic": "Dry, ironic, and lightly mocking.",
        "humorous_tech": "Funny with technology or programming references.",
        "humorous_non_tech": "Funny everyday humour with no technical jargon.",
    }

    UNIKL_ALIASES = {
        "formal": "formal",
        "formel": "formal",
        "sarcastic": "sarcastic",
        "sarcasm": "sarcastic",
        "sarcastik": "sarcastic",
        "sarcastc": "sarcastic",
        "humorous_tech": "humorous_tech",
        "tech_humour": "humorous_tech",
        "tech_humor": "humorous_tech",
        "humorous_non_tech": "humorous_non_tech",
        "nontech_humour": "humorous_non_tech",
        "nontech_humor": "humorous_non_tech",
    }

    def canonical_unikl_key(key: str) -> str:
        clean = key.strip().casefold()
        if "sarcas" in clean:
            return "sarcastic"
        return UNIKL_ALIASES.get(clean, clean)

    def map_unikl_partial(
        payload: dict[str, object],
        styles: list[str],
    ) -> dict[str, str]:
        mapped = {style: "" for style in styles}
        for raw_key, raw_value in payload.items():
            style = canonical_unikl_key(str(raw_key))
            if style in mapped and isinstance(raw_value, str) and raw_value.strip():
                mapped[style] = raw_value.strip()
        return mapped

    def unikl_style_messages(
        description: str,
        styles: list[str],
    ) -> list[dict[str, str]]:
        definitions = {style: UNIKL_STYLE_DEFINITIONS[style] for style in styles}
        return [{
            "role": "user",
            "content": (
                f"Factual video description:\n{description}\n\n"
                f"Requested styles:\n{json.dumps(definitions, ensure_ascii=False)}\n\n"
                "Write one grounded caption for every requested style. Preserve the "
                "description exactly as the factual source and add no new subject, "
                "action, object, or visual detail. Return JSON only, with exactly "
                "the requested style keys."
            ),
        }]

    def unikl_fallback(style: str, description: str) -> str:
        base = " ".join(description.split())[:280]
        prefixes = {
            "formal": "",
            "sarcastic": "Well, would you look at that: ",
            "humorous_tech": "Processing video.exe... output: ",
            "humorous_non_tech": "Basically what happened here: ",
        }
        return f"{prefixes[style]}{base}".strip()

    async def caption_native_video(
        video_url: str,
        styles: list[str],
        model: str,
        profile: str,
        invoke: InvokeFn,
        deadline: float,
    ) -> dict[str, str]:
        if profile not in {"B0_UNIKL_EXACT", "B1_UNIKL_HARDENED"}:
            raise ValueError(f"unknown native profile: {profile}")
        description = await describe_native_video(video_url, model, invoke)
        captions = {style: "" for style in styles}
        for _ in range(3):
            remaining = remaining_seconds(deadline)
            if remaining <= 8.0:
                break
            try:
                async with asyncio.timeout(min(25.0, remaining - 8.0)):
                    text = await invoke(ChatRequest(
                        model=model,
                        messages=unikl_style_messages(description, styles),
                        temperature=0.7,
                        max_tokens=5000,
                    ))
                partial = map_unikl_partial(parse_json_object(text), styles)
                for style, caption in partial.items():
                    if caption:
                        captions[style] = caption
                if all(captions.values()):
                    break
            except (ValueError, TimeoutError, httpx.HTTPError):
                continue
        for style in styles:
            if not captions[style]:
                captions[style] = unikl_fallback(style, description)
        if profile == "B1_UNIKL_HARDENED":
            failures = validate_caption_set(captions, styles)
            if len(failures) == 1 and remaining_seconds(deadline) >= 8.0:
                style, reasons = next(iter(failures.items()))
                captions[style] = await repair_one_style(
                    description, style, captions[style], reasons, model, invoke
                )
        return captions

- [ ] B0 describes once, then performs at most three writer/parser attempts, preserving non-empty styles between attempts and filling only missing styles with the observed 280-character templates. B0 runs with task concurrency 1. B1 calls the common validator and at most one targeted repair after that exact B0 result; no other parameter changes.

- [ ] Run tests.

    python scripts/test_leader_parity.py

    Expected: PASS without network access.

- [ ] Commit.

    git add app/leader_parity.py scripts/test_leader_parity.py
    git commit -m "feat: add MiniMax native video comparison arm"

### Task 6: Wire causal profiles, one retry layer, and absolute deadlines

**Files:**
- Modify: app/pipeline.py
- Modify: scripts/test_leader_parity.py
- Modify: scripts/test_429_retry.py

- [ ] Add failing routing tests for C0, M0, F1, D1, L1, J1, A_FULL, A_HARD, DESCRIBEX_OFFICIAL_REPO, B0_UNIKL_EXACT, and B1_UNIKL_HARDENED. Each test asserts the exact changed fields against the previous profile; no arm may change an undocumented lever.

    PROFILE_PARENTS = {
        "C0": None,
        "M0": "C0",
        "F1": "M0",
        "D1": "M0",
        "L1": "M0",
        "J1": "M0",
        "A_FULL": "M0",
        "A_HARD": "A_FULL",
        "DESCRIBEX_OFFICIAL_REPO": "M0",
        "B0_UNIKL_EXACT": None,
        "B1_UNIKL_HARDENED": "B0_UNIKL_EXACT",
    }

    PROFILE_FRAMES = {
        "C0": "c0",
        "M0": "c0",
        "F1": "describex_oci_hypothesis",
        "D1": "c0",
        "L1": "c0",
        "J1": "c0",
        "A_FULL": "describex_oci_hypothesis",
        "A_HARD": "describex_oci_hypothesis",
        "DESCRIBEX_OFFICIAL_REPO": "official_repo",
        "ENDPOINT_VARIANT": "endpoint_aware",
    }

- [ ] Encode exact unique differences:
  - M0 changes only models/fallbacks from C0;
  - F1 changes only frame geometry from M0;
  - D1 changes only description prompt and its generation parameters from M0;
  - L1 changes only caption length instructions from M0;
  - J1 changes only four per-style calls into one joint call from M0;
  - A_FULL combines F1+D1+L1+J1;
  - A_HARD adds only validator and targeted repair;
  - B1 adds only validator and targeted repair to B0.

- [ ] Add timeout and ladder tests. A frame profile gets one bounded extraction, one 25-second description stage, one 22-second writer stage covering all attempts, at most one 8-second repair, and 8 seconds reserve. Only a native-description failure before any usable description exists may switch to frames, and only with at least 42 seconds remaining. Once native writing starts, B0 preserves completed styles across attempts, reserves 8 seconds, then uses its deterministic templates. C0 is chosen before a task starts and is never entered after v38 has spent inference budget.

    TASK_BUDGET_S = 75.0
    STAGE_BUDGETS = {
        "extract": 12.0,
        "describe": 25.0,
        "write": 22.0,
        "repair": 8.0,
        "reserve": 8.0,
    }

    def can_enter_frames_fallback(deadline: float) -> bool:
        return remaining_seconds(deadline) >= 42.0

    from fractions import Fraction

    def _ffprobe_fps(video: Path) -> float:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nw=1:nk=1", str(video),
        ], text=True).strip()
        fps = float(Fraction(raw))
        if fps <= 0:
            raise ValueError("video FPS must be positive")
        return fps

    def _extract_profile_frames(
        video: Path,
        workdir: Path,
        frame_profile: str,
    ) -> list[Path]:
        if frame_profile == "official_repo":
            return _extract_official_repo_frames(
                video,
                workdir,
                _ffprobe_fps(video),
                _ffprobe_duration(video),
            )
        if frame_profile == "c0":
            return _extract_keyframes(video, workdir, n=10, max_edge=896)
        return _extract_ratio_frames(video, workdir, frame_profile)

    async def extract_for_profile(
        video: Path,
        workdir: Path,
        profile: str,
    ) -> list[Path]:
        async with asyncio.timeout(STAGE_BUDGETS["extract"]):
            return await asyncio.to_thread(
                _extract_profile_frames,
                video,
                workdir,
                PROFILE_FRAMES[profile],
            )

- [ ] Refactor the provider adapter into one network attempt. The profile orchestrator, and only it, owns model fallback and retry count.

    async def _invoke_chat_once(request: ChatRequest) -> str:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        return await _invoke_verified_payload_once(payload)

- [ ] Set PER_TASK_TIMEOUT_S=75 and GLOBAL_BUDGET_S=540. Use MAX_CONCURRENCY=1 for B0_UNIKL_EXACT/B1_UNIKL_HARDENED because the observed UniKL runtime is sequential; use 2 for other profiles. The frame candidate's twelve worst-case tasks complete in six waves, giving 450 seconds plus 90 seconds for startup, downloads, checkpoint writes, and queue jitter. The native profile must prove its own full-batch runtime rather than inherit that arithmetic.

- [ ] Assert that timeout, 429, invalid JSON, or exhausted fallbacks still produce every requested style. After a usable frame/native description exists, no engine switch is allowed: finish with bounded repair, B0 templates, or deterministic emergency captions. C0_source is selected only at task start; R0 is a whole-run operator rollback.

- [ ] Run the routing and retry suites.

    python scripts/test_leader_parity.py
    python scripts/test_429_retry.py

    Expected: PASS, including bounded Retry-After behavior.

- [ ] Commit.

    git add app/pipeline.py scripts/test_leader_parity.py scripts/test_429_retry.py
    git commit -m "feat: route causal v38 profiles under hard deadlines"

### Task 7: Add usage, latency, and cache observability without logging secrets

**Files:**
- Modify: app/leader_parity.py
- Modify: app/pipeline.py
- Modify: app/cache.py
- Modify: app/models.py
- Modify: app/main.py
- Modify: scripts/test_leader_parity.py

- [ ] Add failing tests for a frame-arm cache key containing content SHA-256, model, prompt revision, frame ratios, temperature, and token limit. For a native arm that does not download media, use a canonical URL hash and mark identity_kind=url; never claim it is a content hash. No key may contain API keys or authorization headers.

- [ ] Add tests that provider usage fields, elapsed seconds, retry count, selected profile, actual resolved models and fallbacks used per clip, and fallback reason are recorded as structured local metadata. The judge-independence filter consumes actual_models_used.

- [ ] Raise the legacy normalization ceiling to 1600 characters and keep the
  broad v38 corruption guard at the same ceiling. Length-selection arms enforce
  their own narrower targets without globally truncating rich R0-shaped output.
  Add an end-to-end test through app.main._amain proving a 660-character valid
  v38 caption remains 660 characters after both pipeline and main normalization.

    from app import main as M, models

    def test_main_does_not_retruncate_v38_caption(monkeypatch, tmp_path) -> None:
        caption = ("A" * 659) + "."
        input_path = tmp_path / "tasks.json"
        output_path = tmp_path / "results.json"
        input_path.write_text(json.dumps([{
            "task_id": "long",
            "video_url": "https://example.com/clip.mp4",
            "styles": list(REQUESTED_STYLES),
        }]), encoding="utf-8")

        async def fake_caption_one_video(
            video_url: str,
            styles: list[str],
        ) -> dict[str, str]:
            return {style: caption for style in styles}

        monkeypatch.setattr(M, "INPUT_PATH", input_path)
        monkeypatch.setattr(M, "OUTPUT_PATH", output_path)
        monkeypatch.setattr(M, "caption_one_video", fake_caption_one_video)
        monkeypatch.setattr(M, "MAX_CONCURRENCY", 1)
        monkeypatch.setattr(models, "MAX_CAPTION_CHARS", 1600)
        assert asyncio.run(M._amain()) == 0
        result = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(result[0]["captions"]["formal"]) == 660

- [ ] Implement a prompt revision constant and stable cache key:

    V38_PROMPT_REVISION = "v38-rich-joint-1"

    def leader_parity_cache_key(
        video_identity: str,
        identity_kind: str,
        arm: str,
        model: str,
        parameters: dict[str, object],
    ) -> str:
        payload = {
            "video_identity": video_identity,
            "identity_kind": identity_kind,
            "arm": arm,
            "model": model,
            "prompt_revision": V38_PROMPT_REVISION,
            "parameters": parameters,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

- [ ] Store experimental metadata under out during local runs. The competition result file remains exactly the required task_id and captions schema.

- [ ] Run tests and a local mock run.

    python scripts/test_leader_parity.py
    python scripts/mock_run.py --tasks data/sample_tasks.json --out out/mock_results.json

    Expected: PASS; output schema contains no experimental keys.

- [ ] Commit.

    git add app/leader_parity.py app/pipeline.py app/cache.py app/models.py app/main.py scripts/test_leader_parity.py
    git commit -m "feat: record leader parity usage and cache metadata"

### Task 8: Verify the container contract before evaluation

**Files:**
- Modify: Dockerfile
- Modify: scripts/preflight.py
- Modify: scripts/contract_test.py
- Modify: README.md

- [ ] Add preflight assertions for a valid V38_PROFILE, profile-specific frame count/ratios, bounded task/global timeouts, exact style keys, MAX_CAPTION_CHARS=1600, and non-empty dedicated contest credentials when building a submission candidate.

- [ ] Keep V38_PROFILE=C0 in the default source image until the companion evaluation gate chooses an arm. Add documented environment overrides rather than deleting C0. Set MAX_CAPTION_CHARS=1600, MAX_CONCURRENCY=2, PER_TASK_TIMEOUT_S=75, and GLOBAL_BUDGET_S=540.

- [ ] Run every offline regression suite.

    python scripts/test_verified_short.py
    python scripts/test_leader_parity.py
    python scripts/test_429_retry.py
    python scripts/test_hardening.py
    python scripts/test_style_filter.py
    python scripts/preflight.py

    Expected: every suite exits 0.

- [ ] Build a local linux/amd64 candidate without publishing it.

    docker buildx build --platform linux/amd64 --load -t track2-captioner:v38-local .

    Expected: build exits 0.

- [ ] Extend scripts/contract_test.py with --image, --profile, and --fake-base-url. Start a local fake OpenAI-compatible endpoint, run the container with V38_PROFILE enabled, and assert expected call counts so the contract cannot pass by silently using emergency captions.

- [ ] Run the exact three-task Docker contract twice with a warm cache disabled, once for A_FULL and once for B0_UNIKL_EXACT.

    python scripts/contract_test.py --image track2-captioner:v38-local --profile A_FULL --fake-base-url http://host.docker.internal:8877/v1
    python scripts/contract_test.py --image track2-captioner:v38-local --profile B0_UNIKL_EXACT --fake-base-url http://host.docker.internal:8877/v1

    Expected: 3/3 tasks, 12/12 requested captions, valid JSON, exit code 0 on both runs.

- [ ] Inspect the local image only for OS, architecture, and uncompressed local size.

    docker image inspect track2-captioner:v38-local

    Expected: linux/amd64. Do not label docker image inspect output as compressed registry size.

- [ ] After the evaluation gate authorizes a push, sum compressed blob sizes from the anonymous OCI registry manifest and verify the public index digest plus linux/amd64 child. That registry proof belongs to the final evaluation report.

- [ ] Verify that no personal or production credential is present. Only dedicated, revocable contest credentials may be used in a later public candidate, and their values must never appear in logs or commits.

- [ ] Commit the verified container wiring.

    git add Dockerfile scripts/preflight.py scripts/contract_test.py README.md
    git commit -m "chore: wire reversible v38 container profile"

## Engine completion condition

The engine work is complete only when Tasks 1-8 pass locally. It is not eligible for a public tag or Lablab resubmission until the companion MI300X and paired-panel plan produces a signed gate report with a positive lower confidence bound, no factual-accuracy regression, a full batch under 535 seconds, and a clean Docker contract.
