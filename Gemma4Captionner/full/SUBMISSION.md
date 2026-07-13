# Gemma Hybrid V20.3

## Short description

A Gemma 4-only video-captioning agent that uses a fast 24-frame call first and
automatically activates a deeper Gemma evidence pipeline when quality is at risk.

## Long description

Gemma Hybrid V20 is a grounded AMD Developer Hackathon Track 2 agent powered
exclusively by Gemma 4 models. Its fast path samples 24 ordered frames at up to
640 pixels and asks Gemma 4 31B for formal, sarcastic, humorous-tech, and
humorous-non-tech captions in one multimodal JSON response.

V20.3 uses a 55-second fast-model timeout and permits one 45-second bounded
V18 evidence recovery per batch. Local quality gates repair style-level risks
from an accepted formal caption; batch reruns stay disabled so one difficult
clip cannot consume the budget of later clips.

The public Linux amd64 container reads `/input/tasks.json`, guarantees every
requested style, validates the complete result, and writes strict JSON to
`/output/results.json`.

- Repository: `https://github.com/MVConceptLabAI/Gemma4Agent/tree/gemma4-submission-v20.3/Gemma4Captionner/full`
- Image: `mvconceptlab/gemma4-captioner:gemma4-submission-v20.3`
- Engine: `gemma_hybrid`
- Models: `google/gemma-4-31b-it`, `google/gemma-4-26b-a4b-it`
