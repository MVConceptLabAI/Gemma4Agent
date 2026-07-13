# Gemma Hybrid V20

V20 is a Gemma 4-only Track 2 pipeline that combines the speed of V19 with the
grounded recovery path of V18. It attempts one coherent 24-frame Gemma request
first and spends the additional V18 calls only when the fast result is unsafe.

## Architecture

1. Download the clip and sample 24 uniformly spaced JPEG frames at a maximum
   edge of 640 pixels.
2. Send the ordered sequence to `google/gemma-4-31b-it` in one multimodal call.
3. Parse and locally validate all four requested styles.
4. Accept the fast result when it is complete, concrete, grammatically intact,
   stylistically distinct, and consistent with its formal factual anchor.
5. On timeout, provider failure, malformed output, generic fallback, process
   leak, lexical corruption, contradiction, or near-duplicate voice, recover
   that task with the V18 evidence pipeline. V18 independently observes the
   frames, optionally inspects up to two 60-second MP4 segments with Gemma 4
   26B A4B, writes and verifies captions with Gemma 4 31B.
6. After all tasks, detect exact content-heavy five-word joke spans reused
   across videos. At most two affected tasks can receive bounded V18 recovery
   while the global runtime budget permits.
7. Validate and write `/output/results.json`.

Every model request remains inside the Gemma 4 family.

## Docker

```bash
docker pull mvconceptlab/gemma4-captioner:gemma4-submission-v20.2
docker run --rm \
  -v "$PWD/in:/input:ro" \
  -v "$PWD/out:/output" \
  mvconceptlab/gemma4-captioner:gemma4-submission-v20.2
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
| `NUM_FRAMES` | `24` |
| `FRAME_MAX_EDGE` | `640` |
| `HYBRID_FAST_TIMEOUT_S` | `185` |
| `HYBRID_RECOVERY_TIMEOUT_S` | `220` |
| `HYBRID_BATCH_RECOVERY_MAX` | `2` |
| `MAX_CONCURRENCY` | `3` |
| `PER_TASK_TIMEOUT_S` | `415` |
| `GLOBAL_BUDGET_S` | `540` |

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

The V20 promotion corpus completed eight 30-79 second category clips in 327.1
seconds with 32/32 captions. The final weak-case replay passed the Gemma 4 batch
audit with 0.90 style quality, 0.90 diversity, 0.10 accuracy risk, no shared
patterns, no exact duplicates, and no repeated four-word spans.
