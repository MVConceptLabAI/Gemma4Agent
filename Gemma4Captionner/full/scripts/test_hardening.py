"""v19-hardened-floor free gate: every audited defect repaired, floor intact.

Two parts:
1. Audit cases — each confirmed-by-execution defect must be fixed, and each
   guard the defect's fix could weaken must still fire.
2. Replay-diff — every archived known-good caption (previous gate/AB outputs)
   must pass the NEW filters unchanged: proves no previously-good caption is
   now rejected or altered (the 0.8908 floor invariant).

Run: PYTHONPATH=. python scripts/test_hardening.py
"""
import glob
import json
import sys

sys.path.insert(0, ".")
from app.models import (  # noqa: E402
    FALLBACK_CAPTIONS,
    _clean_caption,
    _has_tech_reference,
    caption_passes_style_filter,
    normalize_captions,
)

FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


print("1. _SUBJ_DIR garble fix")
c = _clean_caption("The dog rolls onto its right side and wags.")
check("'onto its right side' untouched", c == "The dog rolls onto its right side and wags.", c)
c = _clean_caption("The cat lies on its right side in the sun.")
check("'on its right side' untouched", c == "The cat lies on its right side in the sun.", c)
c = _clean_caption("The mouse sits to her left.")
check("'to her left' still neutralized", c == "The mouse sits beside her.", c)

print("2. everyday-word false bans lifted (sarcastic/humorous_non_tech)")
for cap in [
    "A python glides across the sunlit trail with zero urgency.",
    "The squirrel guards its cache of acorns like a tiny accountant.",
    "The peacock deploys its full tail, clearly aware of the audience.",
    "Dress code strictly enforced: mud, more mud, and one triumphant dog.",
    "Two lanes compile into one, and every driver acts surprised.",
    "The whole staging area is just geese now.",
]:
    check(f"sarcastic passes: {cap[:45]!r}", caption_passes_style_filter("sarcastic", cap))
    check(f"non_tech passes: {cap[:45]!r}", caption_passes_style_filter("humorous_non_tech", cap))
check("hard jargon still banned (api)", not caption_passes_style_filter("sarcastic", "An API of pigeons."))
check("hard jargon still banned (kubernetes)", not caption_passes_style_filter(
    "humorous_non_tech", "A kubernetes of ducks marches by."))

print("3. fat/disabled person-directed only")
check("'fat cat' passes", caption_passes_style_filter("humorous_non_tech", "A fat cat lounges on the windowsill."))
check("'disabled truck' passes", caption_passes_style_filter("formal", "A disabled truck sits on the shoulder."))
check("'fat man' still rejected", not caption_passes_style_filter("humorous_non_tech", "A fat man walks by."))
check("'disabled woman' still rejected", not caption_passes_style_filter("formal", "A disabled woman crosses."))

print("4. exclamation repair instead of template swap")
cap = ("Truly the pinnacle of athletic achievement, and the crowd of onlookers lining the "
       "muddy course clearly knows it, phones raised in unified devotion!")
out = normalize_captions({"sarcastic": cap}, ["sarcastic"], facts={"x": 1})
check("sarcastic '!' repaired to '.'", out["sarcastic"].endswith("devotion.")
      and out["sarcastic"] != FALLBACK_CAPTIONS["sarcastic"], out["sarcastic"][:80])

print("5. humorous_tech require-side accepts exemplar vocabulary")
check("'zero-downtime deployment' counts as tech", _has_tech_reference(
    "A zero-downtime deployment of ducks; telemetry says morale is up."))
check("'server' counts as tech", _has_tech_reference("The waiter is this scene's only reliable server."))

print("6. replay-diff: archived known-good captions unchanged and still passing")
seen = 0
diffs = 0
for path in glob.glob("out/**/results.json", recursive=True) + glob.glob("out/ab_*.json"):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    rows = data if isinstance(data, list) else data.get("results", [])
    for row in rows:
        caps = row.get("captions", {})
        if not isinstance(caps, dict):
            continue
        for style, text in caps.items():
            if style not in FALLBACK_CAPTIONS or not isinstance(text, str) or not text:
                continue
            if text == FALLBACK_CAPTIONS[style]:
                continue  # old template fallbacks are exactly what we're fixing
            seen += 1
            cleaned = _clean_caption(text)
            # Byte-identity is only meaningful for current-pipeline output,
            # which is pure ASCII by construction; pre-unicode-fix archives
            # legitimately change ('…' -> '...') and only need to PASS.
            if text == text.encode("ascii", "ignore").decode() and cleaned != text.strip():
                diffs += 1
                print(f"    DIFF [{style}] {path}: {text[:60]!r} -> {cleaned[:60]!r}")
            if not caption_passes_style_filter(style, cleaned):
                diffs += 1
                print(f"    NOW-REJECTED [{style}] {path}: {cleaned[:80]!r}")
check(f"replay-diff over {seen} archived captions", seen > 0 and diffs == 0, f"{diffs} diffs")

print()
if FAILURES:
    print(f"GATE FAILED: {len(FAILURES)} failure(s): {FAILURES}")
    sys.exit(1)
print("GATE PASSED: all hardening checks green, floor invariant proven on archive.")
