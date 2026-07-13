# Research Notes

Last updated: 2026-07-08.

## Local Archive Sweep

Archives found:

- `amd_hackathon_kit.zip`: official/local kit files `00` through `08`.
- `track2_starter.zip`: initial starter with app, Dockerfile, judge, seed data.
- `track2_starter_v2.zip`: richer starter with audio, frames, cache, self-check,
  mock run, dataset v2, LoRA notes, Makefile, and submission draft.

Decision: keep `track2_starter/` as the canonical project. The useful v2 assets
are already integrated, while flattened exports and extracted bundles remain
ignored local references.

## Web Findings

- Lablab Track 2 asks for captions/summaries in four styles: formal, sarcastic,
  humorous-tech, and humorous-non-tech; clips are 30 seconds to 2 minutes; judging
  is by LLM-Judge for accuracy and tone.
- Competitor pages converge on the same architecture: download video, sample
  frames, create factual understanding, then generate one caption per style.
- Fireworks lists Qwen2.5-VL 32B as a vision-language model with long video
  understanding and structured JSON strengths.
- Fireworks lists Gemma 3 27B as text-only on Fireworks, but supports
  fine-tuning and is suitable for the style layer and Gemma bonus positioning.
- Fireworks added direct video/audio inputs for models such as Qwen3 Omni, but
  this requires a dedicated deployment and preprocessing to 1 FPS at 360p, with
  Opus/Ogg audio and payload size discipline.
- Groq vision supports Llama 4 Scout/Maverick image understanding with JSON
  mode. Its request limits make five sampled frames the practical cap for the
  current fallback path.
- Groq Whisper large-v3-turbo is available for low-latency optional audio
  transcription.
- OpenRouter lists Qwen3-VL 8B Instruct as a multimodal vision-language model,
  which keeps it useful as a backup if network access and account limits allow.

## Implementation Decisions

- Keep the frame-based VLM path as the default because it works with serverless
  Fireworks VLM endpoints and has already passed Docker contract validation.
- Add optional `DIRECT_VIDEO_MODEL` as a future higher-accuracy path when a
  dedicated Qwen3 Omni/Molmo2 deployment exists.
- Keep Gemma 3 27B for text styling because Fireworks documents no image input
  support for that endpoint.
- Use Groq as the current live bridge: it produced valid sample outputs with the
  provided key, while Fireworks remains the intended competition path for real
  judging and local mirror scoring.
- Keep Ponytail as a coding method only: use existing dependencies and add the
  smallest checks that prove behavior.

## Sources

- https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii
- https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/veroai/clipforger-amd
- https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/stryvo/stryvo-vision
- https://fireworks.ai/models/fireworks/qwen2p5-vl-32b-instruct
- https://fireworks.ai/models/fireworks/gemma-3-27b-it
- https://fireworks.ai/models/fireworks/qwen3-omni-30b-a3b-instruct
- https://docs.fireworks.ai/guides/video-audio-inputs
- https://console.groq.com/docs/vision
- https://console.groq.com/docs/model/whisper-large-v3-turbo
- https://openrouter.ai/qwen/qwen3-vl-8b-instruct
- https://github.com/DietrichGebert/ponytail
