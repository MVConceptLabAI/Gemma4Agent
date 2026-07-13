# Gemma Fast V19

Gemma Fast V19 is the speed-focused Gemma-only A/B for AMD Developer
Hackathon Track 2. It sends 24 ordered video frames to one Gemma 4 multimodal
JSON request, then validates and repairs the four required styles locally.

The active implementation is [`full/`](full/). V18 remains available on branch
`codex/gemma4-submission-v18` as the grounded multi-stage comparison baseline.

## Submission image

```text
ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v19
```

V19 uses only `google/gemma-4-31b-it` through OpenRouter. It does not call GPT,
Claude, Gemini, Qwen, Groq, Fireworks models, or a second caption-rewrite model.

See [`full/README.md`](full/README.md) for the architecture, commands, tests and
benchmark evidence.
