# Competitive Audit

Last audit: 2026-07-08.

## Goal

Win AMD Developer Hackathon ACT II Track 2 by maximizing both caption accuracy
and style match while avoiding elimination gates: malformed JSON, missing
styles, timeout, wrong Docker architecture, private image, or leaked secrets.

## Requirements Rechecked

- Read `/input/tasks.json` at container startup.
- Write strict `/output/results.json` before exit.
- Emit all four styles for every task:
  `formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`.
- Keep output English, non-empty, and style-distinct.
- Build and publish `linux/amd64`.
- Keep image under 10 GB compressed.
- Complete under 10 minutes for about 12 hidden clips.
- Avoid hardcoded captions and cross-run cache.
- Use credentials from runtime env only.

## Code Audit Findings

- fixed: video download now streams to disk and retries transient HTTP errors.
- fixed: Fireworks calls use configurable fallback model chains instead of a
  single brittle model.
- added: optional direct video/audio describe path for dedicated Fireworks
  multimodal deployments, with automatic fallback to frame-based analysis.
- added: Groq and OpenRouter provider routing. Groq is validated as the current
  live fallback; OpenRouter is configured but the latest local request timed out.
- fixed: runtime caption normalization matches the self-check gate, including
  300-character cap and style-specific bans.
- fixed: local judge now displays `--help` without requiring credentials and
  fails with a clear message when scoring is requested without a key.
- kept: two-stage pipeline because it is the highest-ROI architecture for the
  scoring rubric: understand once, style four times in parallel.
- kept: no new runtime dependency; standard library, httpx, tenacity, and
  existing Pydantic validation cover the needed behavior.

## Current Recommended Model Chain

- Temporary live bridge: Groq
  `meta-llama/llama-4-scout-17b-16e-instruct` for describe and
  `llama-3.3-70b-versatile` for style, with `MAX_CONCURRENCY=1`,
  `NUM_FRAMES=5`, and `FRAME_MAX_EDGE=512`.
- Describe primary: `accounts/fireworks/models/qwen2p5-vl-7b-instruct`
- Describe fallback: `accounts/fireworks/models/qwen2p5-vl-32b-instruct`
- Describe fallback: `accounts/fireworks/models/minimax-m3`
- Optional direct video/audio: `accounts/.../qwen3-omni-30b-a3b-instruct#accounts/.../deployments/...`
- Style primary: `accounts/fireworks/models/gemma-3-27b-it`
- Style fallback: `accounts/fireworks/models/minimax-m3`

Gemma stays in the style layer for bonus positioning; Qwen2.5-VL and MiniMax M3
cover accuracy and provider resilience.

## Remaining Proof Needed

- Run real Fireworks inference with `FIREWORKS_API_KEY`.
- Run mirror judge on at least the provided clips, ideally 20+ varied clips.
- Publish a public image and verify anonymous pull.
- Verify compressed registry size.
- Measure full 12-clip real-inference runtime and p95 per-clip latency.
