# Gemma Fast V19 runbook

## Exact official test

```bash
export OPENROUTER_API_KEY='...'
export CAPTION_ENGINE=gemma_fast
export INPUT_PATH="$PWD/data/official_tasks.json"
export OUTPUT_PATH="$PWD/out/results-v19.json"
python -u -m app.main
python eval/self_check.py --results out/results-v19.json
python scripts/repetition_audit.py out/results-v19.json --fail-on-exact
```

Success requires exit code 0, 15 result rows, 60 non-empty captions, valid JSON,
no exact duplicates, and total runtime below 480 seconds.

## Publish

```bash
git tag -f gemma4-submission-v19
git push origin refs/tags/gemma4-submission-v19 --force
```

The root GitHub workflow publishes a public single-manifest `linux/amd64` image.
Verify its anonymous registry response and architecture before submission.

## Rollback

V18 remains immutable at:

```text
ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
```
