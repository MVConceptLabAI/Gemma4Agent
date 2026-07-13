# V18 submission status

Updated 2026-07-13.

## Active artifact

- Branch: `codex/gemma4-submission-v18`
- Image: `ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18`
- Architecture: `linux/amd64`
- Engine: `gemma_demo`
- Provider: OpenRouter
- Models: Gemma 4 31B and Gemma 4 26B A4B only

## Completed gates

- Required input/output paths and JSON schema are enforced.
- All requested styles receive a non-empty caption, including failure paths.
- Direct videos longer than 60 seconds are split into segments.
- Invalid model JSON is retried with a strict repair request.
- Lexical corruption, unsupported stillness claims, speed inversions, generic
  humour and recurring comedy templates are repaired.
- A global 540-second budget leaves one minute below the contest limit.

## Submission checks

Before resubmitting, run the exact task set, structural validation, repetition
audit, and an anonymous image pull using the commands in `RUNBOOK.md`. Historical
benchmark notes elsewhere in this repository do not describe this V18 image.
