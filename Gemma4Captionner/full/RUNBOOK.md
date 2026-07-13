# Gemma Hybrid V20 runbook

## Exact official test

```bash
export OPENROUTER_API_KEY='...'
export CAPTION_ENGINE=gemma_hybrid
export INPUT_PATH="$PWD/data/official_tasks.json"
export OUTPUT_PATH="$PWD/out/results-v20.json"
python -u -m app.main
python eval/self_check.py --results out/results-v20.json
python scripts/repetition_audit.py out/results-v20.json --fail-on-exact
```

Success requires exit code 0, 15 result rows, 60 non-empty captions, valid JSON,
no exact duplicates, and total runtime below 540 seconds.

## Publish

```bash
git tag gemma4-submission-v20.3
git push origin gemma4-submission-v20.3
```

The root GitHub workflow publishes a public single-manifest `linux/amd64` image
to GHCR and Docker Hub. Verify the exact mounted-I/O contract before submission:

```bash
PUBLIC_IMAGE=mvconceptlab/gemma4-captioner:gemma4-submission-v20.3 \
  bash scripts/verify_public_image.sh
```

## Rollback

V19 and V18 remain immutable at:

```text
ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v19
ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
```
