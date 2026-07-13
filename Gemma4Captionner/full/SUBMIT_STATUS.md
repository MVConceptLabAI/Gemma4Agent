# V20 submission status

## Implemented

- Branch: `codex/gemma4-submission-v20`
- Engine: `gemma_hybrid`
- V19 one-call 24-frame path with a 185-second fast budget
- bounded V18 evidence recovery for failed or unsafe tasks
- direct Gemma 4 video recovery for clips up to two 60-second segments
- final schema, grounding, corruption, voice-distinction and repetition checks
- maximum two batch repetition recoveries within the global budget
- V18 and V19 preserved as immutable rollback images

## Promotion gate

- 8-video long-form container run: exit 0 in 327.1 seconds, 32/32 captions
- targeted weak-case run: urban, people, sports and technology completed in 232.9 seconds
- final deterministic replay: 16/16 valid captions, zero exact duplicates and zero repeated four-grams
- final Gemma 4 batch audit: pass, style quality 0.90, diversity 0.90, accuracy risk 0.10
- local image: Linux amd64, 254 MB

The remaining publication gates are the GitHub Actions build, anonymous pull,
public manifest and clean-container contract verification of the tagged image.
