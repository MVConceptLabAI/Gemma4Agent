# Gemma Hybrid V20.4

V20.4 is a Gemma 4-only Track 2 pipeline that processes one video per bounded
Gemma request. It preserves a hard 28-second task budget while grounding local
style repairs in the accepted formal visual caption.

## Architecture

1. Download the clip and sample 8 uniformly spaced JPEG frames at a maximum
   edge of 640 pixels.
2. Send the ordered sequence to `google/gemma-4-31b-it` in one multimodal call.
3. Parse and locally validate all four requested styles.
4. Accept the fast result when it is complete, concrete, grammatically intact,
   stylistically distinct, and consistent with its formal factual anchor.
5. Local checks reject malformed output, generic fallback, process leaks,
   lexical corruption and unsupported style claims; repairs retain the accepted
   formal visual anchor.
6. Deep and batch recovery reruns are disabled in the submission profile, so
   every video remains within the 28-second per-task guard and the full batch
   remains below the ten-minute limit.
7. Validate and write `/output/results.json`.

Every model request remains inside the Gemma 4 family.

## Docker

```bash
mkdir -p input output
cp data/sample_tasks.json input/tasks.json
docker pull mvconceptlab/gemma4-captioner:gemma4-submission-v20.4
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  mvconceptlab/gemma4-captioner:gemma4-submission-v20.4
python eval/self_check.py --results output/results.json
```

This is the Track 2 contract: each mounted task contains `task_id`, `video_url`
and requested `styles`; the output contains the same `task_id` plus a
`captions` object. It is intentionally different from Track 1's `prompt` /
`answer` schema.

```json
// input/tasks.json
[{"task_id":"v1","video_url":"https://example.test/clip.mp4","styles":["formal","sarcastic","humorous_tech","humorous_non_tech"]}]

// output/results.json
[{"task_id":"v1","captions":{"formal":"...","sarcastic":"...","humorous_tech":"...","humorous_non_tech":"..."}}]
```

The contest injects no runtime credentials, so the publish workflow embeds its
temporary OpenRouter key as a build secret argument. Rotate that key after the
evaluation window.

## Local run

```bash
export OPENROUTER_API_KEY='...'
export CAPTION_ENGINE=gemma_hybrid
export INPUT_PATH=data/official_tasks.json
export OUTPUT_PATH=out/results-v20.json
python -u -m app.main
python eval/self_check.py --results out/results-v20.json
python scripts/repetition_audit.py out/results-v20.json --fail-on-exact
python scripts/audit_results_gemma.py out/results-v20.json --output out/gemma-audit-v20.json
```

PowerShell uses the same variable names through `$env:NAME = 'value'`.

## V20 profile

| Variable | Value |
| --- | --- |
| `CAPTION_ENGINE` | `gemma_hybrid` |
| `GEMMA_FAST_MODEL` | `google/gemma-4-31b-it` |
| `DEMO_GEMMA_MODEL` | `google/gemma-4-31b-it` |
| `DEMO_VIDEO_MODEL` | `google/gemma-4-26b-a4b-it` |
| `NUM_FRAMES` | `8` |
| `FRAME_MAX_EDGE` | `640` |
| `GEMMA_FAST_MAX_TOKENS` | `500` |
| `HYBRID_FAST_TIMEOUT_S` | `24` |
| `HYBRID_RECOVERY_TIMEOUT_S` | `0` |
| `HYBRID_RECOVERY_MAX_PER_RUN` | `0` |
| `HYBRID_BATCH_RECOVERY_MAX` | `0` |
| `MAX_CONCURRENCY` | `1` |
| `PER_TASK_TIMEOUT_S` | `28` |
| `GLOBAL_BUDGET_S` | `540` |
| `GLOBAL_BUDGET_RESERVE_S` | `30` |

## Tests

```bash
PYTHONPATH=. python scripts/test_gemma_fast.py
PYTHONPATH=. python scripts/test_gemma_hybrid.py
PYTHONPATH=. python scripts/test_demo_pipeline_quality.py
PYTHONPATH=. python scripts/contract_test.py
python -m compileall -q app scripts
```

`test_gemma_hybrid.py` verifies fast-path acceptance, timeout recovery, unsafe
output recovery, cross-task repetition detection, and bounded batch recovery.
The older V18 and V19 images remain immutable rollback candidates.

The V20.4 submission profile completed a live 12-clip run in 134 seconds with
48/48 captions. Every task completed in 6.4-22.2 seconds; this is a runtime
smoke test, not a substitute for the hackathon's hidden-set judge score.
