# Gemma Hybrid V20

## Short description

A Gemma 4-only video-captioning agent that uses a fast 24-frame call first and
automatically activates a deeper Gemma evidence pipeline when quality is at risk.

## Long description

Gemma Hybrid V20 is a grounded AMD Developer Hackathon Track 2 agent powered
exclusively by Gemma 4 models. Its fast path samples 24 ordered frames at up to
640 pixels and asks Gemma 4 31B for formal, sarcastic, humorous-tech, and
humorous-non-tech captions in one multimodal JSON response.

V20 automatically activates its V18 evidence pipeline when the fast request
times out or returns malformed, generic, contradictory, corrupted, internally
leaked, or near-duplicate text. The recovery route independently analyzes the
frames, can inspect two 60-second MP4 segments with Gemma 4 26B A4B, and uses
Gemma 4 31B to write, strengthen, and verify every style. A final batch guard
also detects substantial joke wording reused across different videos and can
recover affected tasks while respecting the ten-minute global budget.

The public Linux amd64 container reads `/input/tasks.json`, guarantees every
requested style, validates the complete result, and writes strict JSON to
`/output/results.json`.

- Repository: `https://github.com/MVConceptLabAI/Gemma4Agent/tree/gemma4-submission-v20.2/Gemma4Captionner/full`
- Image: `mvconceptlab/gemma4-captioner:gemma4-submission-v20.2`
- Engine: `gemma_hybrid`
- Models: `google/gemma-4-31b-it`, `google/gemma-4-26b-a4b-it`
