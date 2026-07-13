"""Offline contract tests for the optional creative-discipline writer rule.

Run with::

    PYTHONPATH=. python scripts/test_creative_discipline.py
"""

from __future__ import annotations

import importlib
import os

from app.models import caption_passes_style_filter


_FLAG = "CREATIVE_DISCIPLINE"
_CONCISE_FLAG = "ENSEMBLE_CONCISE"
_SECRET_SENTINEL = "creative-discipline-test-secret"
_ACCEPTED_TECH_MARKERS = (
    "api",
    "latency",
    "cache",
    "runtime",
    "server",
    "pipeline",
    "scheduler",
)


def _load_ensemble(flag: str | None, *, concise: str | None = None):
    if flag is None:
        os.environ.pop(_FLAG, None)
    else:
        os.environ[_FLAG] = flag
    if concise is not None:
        os.environ[_CONCISE_FLAG] = concise
    os.environ["OPENROUTER_API_KEY"] = _SECRET_SENTINEL

    from app import ensemble

    return importlib.reload(ensemble)


def _legacy_writer_system(ensemble) -> str:
    """Reproduce the pre-ablation writer-system expression byte for byte."""
    return (
        ensemble.WRITE_SYSTEM
        + (ensemble._CONCISE_RULE if ensemble.CONCISE else "")
        + ((" " + ensemble.WRITER_LENGTH_HINT) if ensemble.WRITER_LENGTH_HINT else "")
        + (ensemble._GROUNDING_RULE if ensemble.STRICT_GROUNDING else "")
        + (ensemble._EXEMPLAR_BLOCK if ensemble.EXEMPLARS else "")
    )


class _RejectLogging:
    def __getattr__(self, name: str):
        raise AssertionError(f"writer-system construction unexpectedly accessed log.{name}")


def test_default_and_explicit_off_are_byte_identical_to_legacy_system() -> str:
    default = _load_ensemble(None)
    assert default.CREATIVE_DISCIPLINE is False
    default_system = default._writer_system()
    assert default_system == _legacy_writer_system(default)

    explicit_off = _load_ensemble("0")
    assert explicit_off.CREATIVE_DISCIPLINE is False
    explicit_off_system = explicit_off._writer_system()
    assert explicit_off_system.encode("utf-8") == default_system.encode("utf-8")
    assert explicit_off_system == _legacy_writer_system(explicit_off)
    return explicit_off_system


def test_on_appends_only_the_targeted_creative_rules(off_system: str) -> None:
    enabled = _load_ensemble("1")
    assert enabled.CREATIVE_DISCIPLINE is True

    enabled_system = enabled._writer_system()
    delta = enabled._CREATIVE_DISCIPLINE_RULE
    assert enabled_system == off_system + delta

    lowered = delta.casefold()
    required_rules = (
        "creative humor is framing only",
        "preserve literal scene claims",
        "unseen profession, intent, backstory, future action, or off-screen event",
        "explicit similes or metaphors",
        "never turn a person into a developer",
        "never turn typing into commits or code",
        "at most 2 metaphor or punchline devices per caption",
        'never open with "behold" or "ah yes"',
        "avoid repeating api, endpoint, deployment, or zero-latency patterns",
        "other precise terms such as throughput may accompany that marker",
        "apply these discipline rules only to sarcastic, humorous_tech, and humorous_non_tech",
        "this discipline adds no length limit",
        "preserve every already-active length instruction",
        "leave formal governed solely by the pre-existing formal rules",
    )
    for rule in required_rules:
        assert rule in lowered, f"missing creative-discipline rule: {rule!r}"


def test_common_boolean_spellings_are_parsed_strictly() -> None:
    for value in ("", "0", "false", "False", "off", "no", " 0 "):
        assert _load_ensemble(value).CREATIVE_DISCIPLINE is False, value

    for value in ("1", "true", "True", "on", "yes", " 1 "):
        assert _load_ensemble(value).CREATIVE_DISCIPLINE is True, value


def test_invalid_boolean_spelling_is_rejected() -> None:
    try:
        _load_ensemble("enable-maybe")
    except ValueError as error:
        assert _FLAG in str(error)
    else:
        raise AssertionError("invalid creative-discipline flag was accepted")
    finally:
        _load_ensemble("0")


def test_humorous_tech_rule_and_filter_share_natural_markers() -> None:
    enabled = _load_ensemble("1")
    lowered = enabled._CREATIVE_DISCIPLINE_RULE.casefold()
    assert "at least one" in lowered
    for marker in _ACCEPTED_TECH_MARKERS:
        assert marker in lowered, f"prompt omits accepted marker: {marker}"
        marker_caption = (
            f"Visible hands type while a {marker} metaphor keeps the keyboard joke "
            "tied to the on-screen keystrokes."
        )
        assert caption_passes_style_filter("humorous_tech", marker_caption), marker

    live_like = (
        "Visible hands keep a steady keystroke throughput while the keyboard pipeline "
        "turns that rhythm into one restrained technical punchline."
    )
    assert "throughput" in live_like
    assert caption_passes_style_filter("humorous_tech", live_like)


def test_concise_and_discipline_preserve_existing_length_instructions() -> None:
    concise_off = _load_ensemble("0", concise="1")
    concise_system = concise_off._writer_system()
    assert concise_off._CONCISE_RULE in concise_system

    enabled = _load_ensemble("1", concise="1")
    enabled_system = enabled._writer_system()
    delta = enabled._CREATIVE_DISCIPLINE_RULE
    assert enabled_system == concise_system + delta

    lowered = delta.casefold()
    assert "adds no length limit" in lowered
    assert "preserve every already-active length instruction" in lowered
    assert "leave formal governed solely by the pre-existing formal rules" in lowered
    assert "globally cap" not in lowered


def test_flag_does_not_touch_secrets_or_logging() -> None:
    enabled = _load_ensemble("1")
    secret_before = enabled.OR_KEY
    logger_sentinel = _RejectLogging()
    enabled.log = logger_sentinel

    enabled._writer_system()

    assert enabled.OR_KEY == secret_before == _SECRET_SENTINEL
    assert enabled.log is logger_sentinel


def main() -> None:
    original_flag = os.environ.get(_FLAG)
    had_flag = _FLAG in os.environ
    original_concise = os.environ.get(_CONCISE_FLAG)
    had_concise = _CONCISE_FLAG in os.environ
    original_secret = os.environ.get("OPENROUTER_API_KEY")
    had_secret = "OPENROUTER_API_KEY" in os.environ
    try:
        off_system = test_default_and_explicit_off_are_byte_identical_to_legacy_system()
        test_on_appends_only_the_targeted_creative_rules(off_system)
        test_common_boolean_spellings_are_parsed_strictly()
        test_invalid_boolean_spelling_is_rejected()
        test_humorous_tech_rule_and_filter_share_natural_markers()
        test_concise_and_discipline_preserve_existing_length_instructions()
        test_flag_does_not_touch_secrets_or_logging()
    finally:
        if had_flag:
            assert original_flag is not None
            os.environ[_FLAG] = original_flag
        else:
            os.environ.pop(_FLAG, None)
        if had_concise:
            assert original_concise is not None
            os.environ[_CONCISE_FLAG] = original_concise
        else:
            os.environ.pop(_CONCISE_FLAG, None)
        if had_secret:
            assert original_secret is not None
            os.environ["OPENROUTER_API_KEY"] = original_secret
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)

    print("creative_discipline_ok")


if __name__ == "__main__":
    main()
