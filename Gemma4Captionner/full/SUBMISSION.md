# Gemma Hybrid V20.4

## Short description

A Gemma 4-only video-captioning agent that uses one bounded 8-frame call per
clip, then applies local grounded quality repairs.

## Long description

Gemma Hybrid V20 is a grounded AMD Developer Hackathon Track 2 agent powered
exclusively by Gemma 4 models. Its fast path samples 8 ordered frames at up to
640 pixels and asks Gemma 4 31B for formal, sarcastic, humorous-tech, and
humorous-non-tech captions in one multimodal JSON response.

V20.4 uses a 24-second fast-model timeout inside a 28-second hard task budget.
Local quality gates repair style-level risks from an accepted formal caption.
Deep and batch recovery reruns are disabled, while clips are processed one at a
time, so every video request remains within the general 30-second response-time
rule.

The public Linux amd64 container reads `/input/tasks.json`, guarantees every
requested style, validates the complete result, and writes strict JSON to
`/output/results.json`.

- Repository: `https://github.com/MVConceptLabAI/Gemma4Agent/tree/gemma4-submission-v20.4/Gemma4Captionner/full`
- Image: `mvconceptlab/gemma4-captioner:gemma4-submission-v20.4`
- Engine: `gemma_hybrid`
- Models: `google/gemma-4-31b-it`, `google/gemma-4-26b-a4b-it`
