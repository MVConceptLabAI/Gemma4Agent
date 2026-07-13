"""Strict parsing and broad structure/style guards for leader-parity captions.

These helpers deliberately do not compare captions with visual evidence. A
caption that passes here still requires independent factual review.
"""

from __future__ import annotations

import difflib
import json
import re

from app.models import TECH_KEYWORDS, TECH_PHRASES


REQUESTED_STYLES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)
FACTUAL_REVIEW_STATUS = "requires_independent_factual_review"

_HIGH_CONFIDENCE_TECH_WORDS = frozenset(TECH_KEYWORDS) | {"ai", "app"}
_REQUIRE_ONLY_TECH_WORDS = frozenset({"computer", "laptop"})
_CONTEXTUAL_TECH_PHRASES = frozenset(TECH_PHRASES) | {
    "cloud computing",
    "machine learning",
    "python code",
    "python module",
    "python script",
    "server side",
    "server-side",
    "stack trace",
}

NON_TECH_BANNED = tuple(sorted(_HIGH_CONFIDENCE_TECH_WORDS))
TECH_MARKERS = tuple(
    sorted(_HIGH_CONFIDENCE_TECH_WORDS | _REQUIRE_ONLY_TECH_WORDS)
)
_CONTEXTUAL_TECH_PATTERNS = tuple(
    re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for phrase in _CONTEXTUAL_TECH_PHRASES
)

LEAK_MARKERS = (
    "analysis:",
    "assistant:",
    "chain of thought",
    "let's analyze",
    "analyze the image",
    "here is my analysis",
    "i will analyze",
    "reasoning:",
    "<think>",
    "</think>",
    "system prompt",
)

_ALIASES = {
    "humorous-tech": "humorous_tech",
    "humorous-non-tech": "humorous_non_tech",
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_MARKER_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r'''[^.!?]+[.!?]+(?:["'\)\]”’]+)?(?=\s|$)''')
_NON_ENGLISH_FUNCTION_WORDS = (
    frozenset(
        {
            "el", "la", "los", "las", "una", "unas", "unos", "del",
            "por", "para", "con", "mientras", "sobre", "bajo", "desde",
        }
    ),
    frozenset(
        {
            "le", "les", "une", "des", "avec", "pour", "dans", "sous",
            "mais", "comme", "aux", "depuis", "pendant", "entre",
        }
    ),
    frozenset(
        {
            "der", "die", "das", "und", "mit", "fur", "ein", "eine",
            "einen", "auf", "uber", "wahrend", "den", "dem",
        }
    ),
    frozenset(
        {
            "il", "lo", "gli", "della", "delle", "con", "mentre",
            "sotto", "sopra", "senza", "nella", "dentro",
        }
    ),
    frozenset(
        {
            "uma", "com", "para", "pela", "pelo", "enquanto", "sob",
            "sobre", "das", "dos", "pelas", "pelos", "desde",
        }
    ),
)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


_STRICT_JSON_DECODER = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)


def _after_balanced_value(text: str, start: int) -> int:
    """Skip one malformed brace/bracket wrapper without scanning its children."""
    matching = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in matching:
            stack.append(character)
        elif character in "}]":
            if not stack or matching[stack[-1]] != character:
                return len(text)
            stack.pop()
            if not stack:
                return index + 1
    return len(text)


def _contains_json_object(value: object) -> bool:
    if isinstance(value, dict):
        return True
    if isinstance(value, list):
        return any(_contains_json_object(item) for item in value)
    return False


def parse_json_object(text: str) -> dict[str, object]:
    """Extract exactly one root JSON object from optional fences or prose."""
    if not isinstance(text, str):
        raise ValueError("caption response must be text")

    objects: list[dict[str, object]] = []
    arrays: list[list[object]] = []
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        try:
            value, consumed = _STRICT_JSON_DECODER.raw_decode(text[index:])
        except json.JSONDecodeError:
            index = _after_balanced_value(text, index)
            continue
        if isinstance(value, dict):
            objects.append(value)
        elif isinstance(value, list):
            arrays.append(value)
        index += consumed

    structured_array = any(_contains_json_object(value) for value in arrays)
    if len(objects) != 1 or structured_array:
        raise ValueError(f"expected one JSON object, found {len(objects)}")
    return objects[0]


def normalize_requested_captions(
    payload: dict[str, object],
    styles: list[str],
) -> dict[str, str]:
    """Map allowed aliases and require exactly the requested caption strings."""
    if not isinstance(payload, dict):
        raise ValueError("caption payload must be an object")
    if (
        not styles
        or len(styles) != len(set(styles))
        or any(style not in REQUESTED_STYLES for style in styles)
    ):
        raise ValueError("requested caption styles are invalid")

    requested = set(styles)
    normalized: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("caption keys must be strings")
        canonical = _ALIASES.get(key, key)
        if canonical not in requested:
            canonical = key
        if canonical in normalized:
            raise ValueError(f"caption alias collision for {canonical}")
        normalized[canonical] = value

    if set(normalized) != requested:
        raise ValueError("caption keys do not exactly match requested styles")

    captions: dict[str, str] = {}
    for style in styles:
        value = normalized[style]
        if not isinstance(value, str):
            raise ValueError("every caption must be a string")
        clean = value.strip()
        if not clean:
            raise ValueError("every caption must be non-empty")
        captions[style] = clean
    return captions


def caption_violations(style: str, caption: str) -> list[str]:
    """Return broad structure/style failures, never factual-grounding claims."""
    if style not in REQUESTED_STYLES:
        raise ValueError(f"unsupported caption style: {style}")
    if not isinstance(caption, str):
        raise ValueError("caption must be a string")

    clean = " ".join(caption.split())
    words = _WORD_RE.findall(clean)
    folded_words = [word.casefold() for word in words]
    marker_words = [
        word.casefold() for word in _MARKER_WORD_RE.findall(clean)
    ]
    marker_set = set(marker_words)
    has_contextual_tech = any(
        pattern.search(clean) for pattern in _CONTEXTUAL_TECH_PATTERNS
    )
    sentences = _SENTENCE_RE.findall(clean)
    violations: list[str] = []

    if not clean:
        violations.append("empty")
    sentence_min, sentence_max = ((3, 9) if style == "formal" else (2, 6))
    word_min, word_max = ((80, 220) if style == "formal" else (40, 150))
    has_unfinished_text = bool(_SENTENCE_RE.sub("", clean).strip())
    if (
        not sentence_min <= len(sentences) <= sentence_max
        or has_unfinished_text
    ):
        violations.append("sentence_count")
    if not word_min <= len(words) <= word_max:
        violations.append("word_count")
    if len(clean) > 1600:
        violations.append("character_count")

    if (
        style in {"sarcastic", "humorous_non_tech"}
        and (
            marker_set.intersection(NON_TECH_BANNED)
            or has_contextual_tech
        )
    ):
        violations.append("technical_jargon")
    if (
        style == "humorous_tech"
        and not marker_set.intersection(TECH_MARKERS)
        and not has_contextual_tech
    ):
        violations.append("missing_tech_reference")

    lower = clean.casefold()
    if any(marker in lower for marker in LEAK_MARKERS):
        violations.append("leaked_reasoning")

    repetition_detected = False
    for fragment_size, repetitions in ((8, 2), (3, 3)):
        fragments = [
            tuple(folded_words[index:index + fragment_size])
            for index in range(
                max(0, len(folded_words) - fragment_size + 1)
            )
        ]
        if any(
            fragments.count(fragment) >= repetitions
            for fragment in set(fragments)
        ):
            repetition_detected = True
            break
    if repetition_detected:
        violations.append("repeated_fragment")

    letters = [character for character in clean if character.isalpha()]
    if letters:
        ascii_ratio = sum(character.isascii() for character in letters) / len(letters)
        if ascii_ratio < 0.8:
            violations.append("non_english")
    if len(marker_words) >= 20 and any(
        sum(word in function_words for word in marker_words) >= 4
        and sum(word in function_words for word in marker_words) / len(marker_words)
        >= 0.1
        for function_words in _NON_ENGLISH_FUNCTION_WORDS
    ) and "non_english" not in violations:
        violations.append("non_english")
    return violations


def validate_caption_set(
    captions: dict[str, str],
    styles: list[str],
) -> dict[str, list[str]]:
    """Validate structure/style only; every result still needs factual review."""
    failures: dict[str, list[str]] = {}
    for style in styles:
        reasons = caption_violations(style, captions.get(style, ""))
        if reasons:
            failures[style] = reasons

    for left_index, left_style in enumerate(styles):
        for right_style in styles[left_index + 1:]:
            left_words = [
                word.casefold()
                for word in _WORD_RE.findall(captions.get(left_style, ""))
            ]
            right_words = [
                word.casefold()
                for word in _WORD_RE.findall(captions.get(right_style, ""))
            ]
            ratio = difflib.SequenceMatcher(
                None,
                left_words,
                right_words,
                autojunk=False,
            ).ratio()
            if ratio >= 0.92:
                failures.setdefault(right_style, []).append(
                    f"near_duplicate:{left_style}"
                )
    return failures
