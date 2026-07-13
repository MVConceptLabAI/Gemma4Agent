# V19 submission status

## Implemented

- Branch: `codex/gemma4-submission-v19`
- Engine: `gemma_fast`
- 24 uniform frames at maximum edge 640
- exactly one Gemma multimodal request per successful clip
- four styles returned in one JSON object
- local JSON salvage, schema validation, style distinction and safe fallbacks
- V18 preserved as the rollback image

## Promotion gate

Passed on the complete 15-task `data/official_tasks.json` set:

- 60/60 valid captions and zero exact duplicates
- 161.4 seconds versus 406.0 seconds for the V18 comparison run
- Gemma 4 full-JSON audit: pass
- style quality 0.90, diversity 0.90, accuracy risk 0.00
- no shared templates or cliches reported by the final audit

V18 remains available as the conservative rollback image.
