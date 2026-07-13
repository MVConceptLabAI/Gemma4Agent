# Gemma Fast V19

## Short description

A Gemma 4-only video-captioning agent that turns 24 ordered frames into all four
required styles in one multimodal JSON call, followed by strict local validation.

## Long description

Gemma Fast V19 is a speed-focused AMD Developer Hackathon Track 2 agent powered
exclusively by Gemma 4 31B through OpenRouter. It samples 24 uniformly spaced
frames at up to 640 pixels and sends the entire ordered visual sequence in one
multimodal request. Gemma returns formal, sarcastic, humorous-tech, and
humorous-non-tech captions together in one JSON object, keeping every voice tied
to one coherent view of the clip.

After inference, deterministic local checks salvage malformed JSON, guarantee
every requested style, limit caption length, require a genuine technology
reference in tech humour, remove technology from non-tech humour, reject
near-duplicate voices, and replace internal pipeline wording or recycled comedy
templates. These checks make no additional model request.

The public Linux amd64 container reads `/input/tasks.json`, writes validated
`/output/results.json`, and keeps V18 available as a rollback baseline.

- Repository: `https://github.com/MVConceptLabAI/Gemma4Agent/tree/codex/gemma4-submission-v19/Gemma4Captionner/full`
- Image: `ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v19`
- Engine: `gemma_fast`
- Model: `google/gemma-4-31b-it`
