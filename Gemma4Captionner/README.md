# Gemma 4 Captioning V18

Gemma 4 Captioning V18 is an AMD Developer Hackathon Track 2 video-captioning
agent powered exclusively by models from the Gemma 4 family. It grounds every
caption in independently collected visual evidence and returns the four
required English styles: `formal`, `sarcastic`, `humorous_tech`, and
`humorous_non_tech`.

The active submission is [`full/`](full/). Historical experiment folders are
kept for comparison only and are not used by the V18 container.

## Submission image

```text
ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
```

The image is public, targets `linux/amd64`, reads `/input/tasks.json`, writes
`/output/results.json`, and is well below the 10 GB compressed-image limit.

## Gemma-only pipeline

1. FFmpeg samples 24 representative frames at a maximum edge of 640 pixels.
2. Gemma 4 31B analyses each frame independently with bounded parallelism.
3. Gemma 4 26B A4B inspects a compact MP4 for confirmed temporal or audible
   information. Clips longer than 60 seconds are covered by two segments.
4. Gemma 4 31B consolidates evidence and writes all four captions.
5. Dedicated sarcasm and humour passes strengthen style without changing facts.
6. Final grounding, lexical, OCR, motion, and strict-JSON checks repair risky
   outputs while preserving useful detail.

No GPT, Claude, Gemini, Qwen, Groq, or Fireworks model is used by the V18
submission path.

## Verified results

The current pipeline processed the 12-task AMD-hosted validation set in
`full/data/official_new12.json` in 340.7 seconds and wrote a valid 48-caption
result file with exit code 0. That run exposed one malformed model JSON response
on task `6023186`; V18 now retries strict JSON once. The corrected task was then
retested successfully in 69.9 seconds with four grounded captions and no task
fallback.

See [`full/README.md`](full/README.md) for commands, configuration, contract,
architecture, and reproducible validation details.
