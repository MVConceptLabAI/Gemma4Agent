from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_manifest import (  # noqa: E402
    EXPECTED_AMD64,
    EXPECTED_INDEX,
    validate_v36_manifest,
)


def parse_dockerfile_env_assignments(dockerfile: str) -> list[tuple[str, str]]:
    logical_instructions: list[str] = []
    fragments: list[str] = []

    for raw_line in dockerfile.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        continued = stripped.endswith("\\")
        fragments.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            logical_instructions.append(" ".join(fragments))
            fragments = []

    if fragments:
        logical_instructions.append(" ".join(fragments))

    assignments: list[tuple[str, str]] = []
    for instruction in logical_instructions:
        name, separator, arguments = instruction.partition(" ")
        if not separator or name.upper() != "ENV":
            continue

        tokens = shlex.split(arguments, posix=True)
        if tokens and "=" not in tokens[0]:
            assignments.append((tokens[0], " ".join(tokens[1:])))
            continue

        for token in tokens:
            key, equals, value = token.partition("=")
            if key and equals:
                assignments.append((key, value))

    return assignments


def fixture_manifest(
    index_digest: str,
    child_digest: str,
    os_name: str,
    architecture: str,
) -> dict[str, object]:
    return {
        "index_digest": index_digest,
        "manifests": [
            {
                "digest": child_digest,
                "platform": {"os": os_name, "architecture": architecture},
            }
        ],
    }


def assert_rejected(payload: dict[str, object], message: str) -> None:
    try:
        validate_v36_manifest(payload)
    except ValueError:
        return
    raise AssertionError(message)


def test_v36_manifest_requires_exact_index_and_amd64_child() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, EXPECTED_AMD64, "linux", "amd64")
    assert validate_v36_manifest(manifest) == EXPECTED_AMD64


def test_v36_manifest_rejects_wrong_index() -> None:
    manifest = fixture_manifest("sha256:" + "0" * 64, EXPECTED_AMD64, "linux", "amd64")
    assert_rejected(manifest, "wrong index digest was accepted")


def test_v36_manifest_rejects_mutable_index_reference() -> None:
    manifest = fixture_manifest(
        "ghcr.io/theskygold/track2-captioner:latest",
        EXPECTED_AMD64,
        "linux",
        "amd64",
    )
    assert_rejected(manifest, "mutable index reference was accepted")


def test_v36_manifest_rejects_attestation_child() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, "sha256:" + "9" * 64, "unknown", "unknown")
    assert_rejected(manifest, "unknown/unknown attestation child was accepted")


def test_v36_manifest_rejects_valid_child_plus_attestation() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, EXPECTED_AMD64, "linux", "amd64")
    manifest["manifests"].append(
        {
            "digest": "sha256:" + "9" * 64,
            "platform": {"os": "unknown", "architecture": "unknown"},
        }
    )
    assert_rejected(manifest, "valid child plus unknown/unknown attestation was accepted")


def test_v36_manifest_rejects_valid_child_plus_wrong_platform() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, EXPECTED_AMD64, "linux", "amd64")
    manifest["manifests"].append(
        {
            "digest": "sha256:" + "a" * 64,
            "platform": {"os": "linux", "architecture": "arm64"},
        }
    )
    assert_rejected(manifest, "valid child plus wrong-platform child was accepted")


def test_v36_manifest_rejects_wrong_amd64_digest() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, "sha256:" + "f" * 64, "linux", "amd64")
    assert_rejected(manifest, "wrong linux/amd64 digest was accepted")


def test_v36_manifest_rejects_missing_amd64_child() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, EXPECTED_AMD64, "linux", "arm64")
    assert_rejected(manifest, "manifest without a linux/amd64 child was accepted")


def test_v36_manifest_rejects_duplicate_amd64_children() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, EXPECTED_AMD64, "linux", "amd64")
    child = manifest["manifests"][0]
    manifest["manifests"] = [child, dict(child)]
    assert_rejected(manifest, "duplicate linux/amd64 children were accepted")


def test_cli_writes_deterministic_sanitized_manifest() -> None:
    manifest = fixture_manifest(EXPECTED_INDEX, EXPECTED_AMD64, "linux", "amd64")
    manifest["metadata"] = {"untrusted": "omitted"}
    expected = json.dumps(
        {
            "index_digest": EXPECTED_INDEX,
            "manifests": [
                {
                    "digest": EXPECTED_AMD64,
                    "platform": {"architecture": "amd64", "os": "linux"},
                }
            ],
        },
        indent=2,
        sort_keys=True,
    ) + "\n"

    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "input.json"
        output_path = Path(directory) / "sanitized.json"
        input_path.write_text(json.dumps(manifest), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "baseline_manifest.py"),
                str(input_path),
                "--output",
                str(output_path),
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == f"{EXPECTED_AMD64}\n"
        assert output_path.read_text(encoding="utf-8") == expected


def assert_sanitized_cli_failure(input_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "baseline_manifest.py"),
            str(input_path),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "baseline manifest validation failed\n"
    assert "Traceback" not in completed.stderr
    for forbidden_path in (input_path.resolve(), ROOT.resolve(), Path(sys.executable).resolve()):
        assert str(forbidden_path).casefold() not in completed.stderr.casefold()


def test_cli_sanitizes_missing_input_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assert_sanitized_cli_failure(Path(directory) / "missing.json")


def test_cli_sanitizes_unreadable_input_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assert_sanitized_cli_failure(Path(directory))


def test_cli_sanitizes_malformed_json_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "malformed.json"
        input_path.write_text("{not-json", encoding="utf-8")
        assert_sanitized_cli_failure(input_path)


def test_cli_sanitizes_non_object_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "list.json"
        input_path.write_text("[]", encoding="utf-8")
        assert_sanitized_cli_failure(input_path)


def test_cli_sanitizes_recursive_json_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "deeply-nested.json"
        input_path.write_text("[" * 5000 + "0" + "]" * 5000, encoding="utf-8")
        assert_sanitized_cli_failure(input_path)


def test_cli_sanitizes_manifest_validation_failure() -> None:
    manifest = fixture_manifest(
        "sha256:" + "0" * 64,
        EXPECTED_AMD64,
        "linux",
        "amd64",
    )
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "wrong-digest.json"
        input_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert_sanitized_cli_failure(input_path)


def test_evidence_manifest_is_safe_and_explicitly_timestamp_inferred() -> None:
    evidence_path = ROOT / "docs" / "evals" / "v36-baseline-manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert set(evidence) == {
        "attributed_digest",
        "displayed_historical_score",
        "evaluation_timestamp",
        "image",
        "index_digest",
        "linux_amd64_digest",
        "note",
        "reference",
        "resubmission_timestamp",
        "schema",
        "score_attribution",
        "source_commit",
    }
    assert evidence["index_digest"] == EXPECTED_INDEX
    assert evidence["linux_amd64_digest"] == EXPECTED_AMD64
    assert evidence["source_commit"] == "283ce7f"
    assert evidence["displayed_historical_score"] == 0.9133
    assert evidence["score_attribution"] == "timestamp-inferred"
    assert evidence["attributed_digest"] is None
    assert evidence["resubmission_timestamp"] is None
    assert evidence["evaluation_timestamp"] is None
    assert "did not expose" in evidence["note"]
    assert "not cryptographically proven" in evidence["note"]


def test_dockerfile_env_parser_ignores_inactive_and_prefixed_text() -> None:
    dockerfile = """
# ENV CAPTION_ENGINE=ensemble
ENV NOT_CAPTION_ENGINE=ensemble
CAPTION_ENGINE=ensemble
RUN printf 'CAPTION_ENGINE=ensemble'
plain text CAPTION_ENGINE=ensemble
"""

    assert parse_dockerfile_env_assignments(dockerfile) == [
        ("NOT_CAPTION_ENGINE", "ensemble")
    ]


def test_dockerfile_env_parser_reads_multiline_key_value_pairs() -> None:
    dockerfile = (
        "ENV CAPTION_ENGINE=ensemble \\\n"
        "    MAX_CAPTION_CHARS=1600 \\\n"
        "    NUM_FRAMES=10 \\\n"
        "    FRAME_MAX_EDGE=896\n"
    )

    assert parse_dockerfile_env_assignments(dockerfile) == [
        ("CAPTION_ENGINE", "ensemble"),
        ("MAX_CAPTION_CHARS", "1600"),
        ("NUM_FRAMES", "10"),
        ("FRAME_MAX_EDGE", "896"),
    ]


def test_controlled_c0_dockerfile_pins() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    expected = {
        "CAPTION_ENGINE": "ensemble",
        "MAX_CAPTION_CHARS": "1600",
        "NUM_FRAMES": "10",
        "FRAME_MAX_EDGE": "896",
    }
    pins = [
        assignment
        for assignment in parse_dockerfile_env_assignments(dockerfile)
        if assignment[0] in expected
    ]

    assert len(pins) == len(expected), f"Dockerfile pins must be unique: {pins!r}"
    assert dict(pins) == expected, f"Dockerfile pins differ: {pins!r}"


def main() -> None:
    test_v36_manifest_requires_exact_index_and_amd64_child()
    test_v36_manifest_rejects_wrong_index()
    test_v36_manifest_rejects_mutable_index_reference()
    test_v36_manifest_rejects_attestation_child()
    test_v36_manifest_rejects_valid_child_plus_attestation()
    test_v36_manifest_rejects_valid_child_plus_wrong_platform()
    test_v36_manifest_rejects_wrong_amd64_digest()
    test_v36_manifest_rejects_missing_amd64_child()
    test_v36_manifest_rejects_duplicate_amd64_children()
    test_cli_writes_deterministic_sanitized_manifest()
    test_cli_sanitizes_missing_input_failure()
    test_cli_sanitizes_unreadable_input_failure()
    test_cli_sanitizes_malformed_json_failure()
    test_cli_sanitizes_non_object_failure()
    test_cli_sanitizes_recursive_json_failure()
    test_cli_sanitizes_manifest_validation_failure()
    test_evidence_manifest_is_safe_and_explicitly_timestamp_inferred()
    test_dockerfile_env_parser_ignores_inactive_and_prefixed_text()
    test_dockerfile_env_parser_reads_multiline_key_value_pairs()
    test_controlled_c0_dockerfile_pins()
    print("baseline_manifest_ok")


if __name__ == "__main__":
    main()
