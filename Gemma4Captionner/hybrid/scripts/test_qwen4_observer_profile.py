"""Focused offline guard for the measured W4 submission profile.

The candidate keeps the measured v38 four-observer profile and enables the
validated per-style W4 writer path. Publication paths also disable BuildKit
provenance/SBOM attestations so GHCR exposes one amd64 image manifest rather
than an image plus unknown-platform attestation manifests.
"""

from __future__ import annotations

import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V36_PROFILE: list[tuple[str, str]] = [
    ("PYTHONPATH", "/app"),
    ("CAPTION_ENGINE", "ensemble"),
    (
        "ENSEMBLE_OBSERVERS",
        "openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.5",
    ),
    ("ENSEMBLE_WRITER", "anthropic/claude-opus-4.5"),
    ("STYLE_EXEMPLARS", "1"),
    ("STRICT_GROUNDING", "0"),
    ("WRITER_TEMP", "0.5"),
    ("W4_STYLE_SPLIT", "1"),
    ("TIMESTAMP_FRAMES", "0"),
    ("MAX_CAPTION_CHARS", "1600"),
    ("OPENROUTER_VLM_MODEL", "qwen/qwen3-vl-235b-a22b-instruct"),
    ("OPENROUTER_STYLE_MODEL", "google/gemma-4-31b-it"),
    ("PROVIDER_ORDER", "openrouter,groq,fireworks"),
    ("STYLE_PROVIDER_ORDER", "openrouter,fireworks,groq"),
    ("STYLE_MODEL", "accounts/fireworks/models/gpt-oss-120b"),
    ("STYLE_REASONING_EFFORT", "low"),
    ("STYLE_MAX_TOKENS", "1400"),
    ("DETERMINISTIC_FORMAL", "1"),
    ("NUM_FRAMES", "10"),
    ("FRAME_MAX_EDGE", "896"),
    ("GROQ_MAX_IMAGES", "4"),
    ("GROQ_FRAME_MAX_EDGE", "448"),
    ("HTTP_429_RETRIES", "5"),
    ("HTTP_429_MAX_WAIT_S", "45"),
    ("RETRY_AFTER_GIVEUP_S", "60"),
    ("DESCRIBE_MAX_TOKENS", "1300"),
    ("SCENE_DETECT_ENABLED", "0"),
    ("MAX_CONCURRENCY", "2"),
    ("PER_TASK_TIMEOUT_S", "150"),
    ("GLOBAL_BUDGET_S", "540"),
]

EXPECTED_OBSERVERS = [
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4.5",
    "qwen/qwen3-vl-235b-a22b-instruct",
]


def _env_assignments(dockerfile: str) -> list[tuple[str, str]]:
    instructions: list[str] = []
    fragments: list[str] = []
    for raw_line in dockerfile.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continued = stripped.endswith("\\")
        fragments.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            instructions.append(" ".join(fragments))
            fragments = []

    assignments: list[tuple[str, str]] = []
    for instruction in instructions:
        name, separator, arguments = instruction.partition(" ")
        if not separator or name.upper() != "ENV":
            continue
        for token in shlex.split(arguments, posix=True):
            key, equals, value = token.partition("=")
            if key and equals:
                assignments.append((key, value))
    return assignments


def _submission_profile() -> list[tuple[str, str]]:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assignments = _env_assignments(dockerfile)
    start = assignments.index(("PYTHONPATH", "/app"))
    end = next(
        index
        for index in range(start, len(assignments))
        if assignments[index][0] == "GLOBAL_BUDGET_S"
    )
    return assignments[start : end + 1]


def test_exactly_four_observers_in_measured_order() -> None:
    profile = dict(_submission_profile())
    observers = profile["ENSEMBLE_OBSERVERS"].split(",")
    assert observers == EXPECTED_OBSERVERS
    assert len(observers) == 4
    assert len(set(observers)) == 4


def test_w4_style_split_is_enabled() -> None:
    assert dict(_submission_profile())["W4_STYLE_SPLIT"] == "1"


def test_no_other_v38_submission_config_drift() -> None:
    expected = list(V36_PROFILE)
    observer_index = next(
        index for index, (key, _value) in enumerate(expected)
        if key == "ENSEMBLE_OBSERVERS"
    )
    expected[observer_index] = (
        "ENSEMBLE_OBSERVERS",
        ",".join(EXPECTED_OBSERVERS),
    )
    assert _submission_profile() == expected


def test_publish_paths_disable_provenance_and_sbom() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    assert "          provenance: false\n" in workflow
    assert "          sbom: false\n" in workflow

    for relative in (
        Path("scripts/build.sh"),
        Path("scripts/publish_image.sh"),
    ):
        script = (ROOT / relative).read_text(encoding="utf-8")
        assert "    --provenance=false \\\n" in script, relative
        assert "    --sbom=false \\\n" in script, relative


def main() -> None:
    test_exactly_four_observers_in_measured_order()
    test_w4_style_split_is_enabled()
    test_no_other_v38_submission_config_drift()
    test_publish_paths_disable_provenance_and_sbom()
    print("qwen4_observer_profile_ok")


if __name__ == "__main__":
    main()
