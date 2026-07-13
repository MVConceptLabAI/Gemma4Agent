# Gemma 4 Captioning V18

## Short description

A video-captioning agent that uses only Gemma 4 models to ground each clip and
produce formal, sarcastic, humorous-tech and humorous-non-tech captions.

## Long description

Gemma 4 Captioning V18 is a grounded AMD Developer Hackathon Track 2 agent built
entirely on Gemma 4 through OpenRouter. It samples 24 representative frames at
up to 640 pixels and analyzes them independently in parallel with Gemma 4 31B.
For temporal context, each video is also sent directly to Gemma 4 26B A4B in
segments of at most 60 seconds; those observations may add only details that
remain consistent with the frame evidence.

Gemma 4 then consolidates supported facts, writes all four required English
styles, strengthens sarcasm and both humour variants, validates grounding, and
repairs malformed JSON or wording without dropping a requested caption. V18
also rejects recurring comedy templates so a batch does not repeat the same
"masterclass", GPU, or toddler joke across unrelated clips.

The Linux `amd64` container follows the exact contract: read
`/input/tasks.json`, write valid `/output/results.json`, return exit code 0 on
success, and stay within the 10-minute global budget.

## Submission fields

- Repository: `https://github.com/MVConceptLabAI/Gemma4Agent/tree/codex/gemma4-submission-v18/Gemma4Captionner/full`
- Image: `ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18`
- Runtime key: `OPENROUTER_API_KEY`
- Active engine: `gemma_demo`

No GPT, Claude, Gemini, Qwen, Groq, Fireworks model, Whisper, or local fallback
model is used by the V18 submission path.
