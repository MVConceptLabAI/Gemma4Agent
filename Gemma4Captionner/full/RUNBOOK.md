# Gemma 4 Captioning V18 runbook

V18 uses OpenRouter only and routes every inference stage to Gemma 4. The active
container reads `/input/tasks.json`, writes `/output/results.json`, and uses the
`gemma_demo` engine configured in `Dockerfile`.

## Local Python run

```bash
export OPENROUTER_API_KEY='...'
export INPUT_PATH="$PWD/data/official_tasks.json"
export OUTPUT_PATH="$PWD/out/results.json"
python -m app.main
python eval/self_check.py --results out/results.json
python scripts/repetition_audit.py out/results.json --fail-on-exact
```

## Docker run

```bash
mkdir -p in out
cp data/official_tasks.json in/tasks.json
docker run --rm \
  -e OPENROUTER_API_KEY \
  -v "$PWD/in:/input:ro" \
  -v "$PWD/out:/output" \
  ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
python eval/self_check.py --results out/results.json
python scripts/repetition_audit.py out/results.json --fail-on-exact
```

For PowerShell, replace `$PWD` mounts with resolved Windows paths. A successful
run exits 0 and produces one non-empty caption for every requested style.

## Build and publish

```bash
export PUBLIC_IMAGE=ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
bash scripts/publish_image.sh
bash scripts/verify_public_image.sh
```

The image must be public, `linux/amd64`, below 10 GB compressed, and anonymously
pullable. Provider credentials are runtime environment variables; they are not
stored in Git.

## Important settings

| Variable | V18 default | Purpose |
| --- | ---: | --- |
| `DEMO_GEMMA_MODEL` | `google/gemma-4-31b-it` | Frame analysis, consolidation, writing and review |
| `DEMO_VIDEO_MODEL` | `google/gemma-4-26b-a4b-it` | Direct MP4 temporal/audio-aware observation |
| `NUM_FRAMES` | `24` | Representative frames per clip |
| `FRAME_MAX_EDGE` | `640` | Maximum sampled-frame edge |
| `DEMO_OBSERVATION_CONCURRENCY` | `9` | Parallel frame observations |
| `VIDEO_CONTEXT_MAX_SECONDS` | `60` | Maximum length per direct-video segment |
| `MAX_CONCURRENCY` | `3` | Parallel task limit |
| `PER_TASK_TIMEOUT_S` | `220` | Per-clip safety timeout |
| `GLOBAL_BUDGET_S` | `540` | Whole-run budget below the 10-minute rule |

Older ensemble, Fireworks, Groq, Qwen and Gemma 3 documents are retained only
as historical experiments and do not describe V18.
