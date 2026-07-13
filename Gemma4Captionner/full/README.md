# Gemma Fast V19

V19 is a deliberately compact Gemma-only Track 2 pipeline based on the common
architecture of two public 0.91 submissions: 24 bounded frames, one coherent
multimodal request returning all four styles, then strict local validation.

## Architecture

1. Download the clip and sample 24 uniformly spaced JPEG frames at a maximum
   edge of 640 pixels.
2. Send the ordered frame sequence to `google/gemma-4-31b-it` in exactly one
   OpenRouter request.
3. Ask for one strict JSON object containing `formal`, `sarcastic`,
   `humorous_tech`, and `humorous_non_tech` captions of 12-35 words.
4. Parse or salvage the response locally; no JSON-repair model call is made.
5. Locally enforce schema, non-empty styles, English/style filters, caption
   length, technology presence, style distinction, process-leak bans, and
   recycled-comedy bans.
6. Write validated `/output/results.json`.

Unlike V18, V19 does not make 24 independent observation requests, send the MP4
directly, consolidate evidence, or call separate humour/verifier stages. This
is an A/B variant, not an assertion that less grounding is always better.

## Docker

```bash
docker pull ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v19
docker run --rm \
  -v "$PWD/in:/input:ro" \
  -v "$PWD/out:/output" \
  ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v19
```

The published image contains the temporary judging key because the contest
injects no runtime environment variables. Rotate that key after judging.

## Local run

```bash
export OPENROUTER_API_KEY='...'
export CAPTION_ENGINE=gemma_fast
export INPUT_PATH=data/official_tasks.json
export OUTPUT_PATH=out/results-v19.json
python -u -m app.main
python eval/self_check.py --results out/results-v19.json
python scripts/repetition_audit.py out/results-v19.json --fail-on-exact
python scripts/audit_results_gemma.py out/results-v19.json --output out/gemma-audit-v19.json
```

PowerShell uses the same variable names through `$env:NAME = 'value'`.

## V19 profile

| Variable | Value |
| --- | --- |
| `CAPTION_ENGINE` | `gemma_fast` |
| `GEMMA_FAST_MODEL` | `google/gemma-4-31b-it` |
| `GEMMA_FAST_MAX_TOKENS` | `700` |
| `NUM_FRAMES` | `24` |
| `FRAME_MAX_EDGE` | `640` |
| `SCENE_DETECT_ENABLED` | `0` |
| `MAX_CONCURRENCY` | `3` |
| `PER_TASK_TIMEOUT_S` | `120` |
| `GLOBAL_BUDGET_S` | `480` |

## Tests

```bash
PYTHONPATH=. python scripts/test_gemma_fast.py
PYTHONPATH=. python scripts/test_demo_pipeline_quality.py
python scripts/contract_test.py
python -m compileall -q app scripts
```

`test_gemma_fast.py` verifies that one clip produces exactly one HTTP request
whose content is one prompt plus 24 images. It also tests malformed-JSON salvage,
style distinction, technology presence, and stale-template replacement.

`audit_results_gemma.py` is an offline promotion gate, not part of the submitted
runtime. It gives the complete final JSON to Gemma 4 to flag shared clichés,
semantic repetition, style failures, grammar defects, and facts introduced by a
styled caption but absent from its formal factual anchor.

## Validated V19 result

The full V19 run processed the 15 tasks in `data/official_tasks.json` in 161.4
seconds, wrote 60/60 captions, and produced no exact duplicates. The equivalent
V18 run took 406.0 seconds, so the single-call path was about 60% faster in this
test. An offline full-batch Gemma 4 review passed the promotion candidate with
0.90 style quality, 0.90 diversity, 0.00 accuracy risk, no shared patterns, and
no reported issues.
The audit checks cross-caption consistency against each formal factual anchor;
it complements rather than replaces visual grounding from the 24 input frames.
