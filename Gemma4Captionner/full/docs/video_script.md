# Demo video script — Track 2 Captioner (2:30)

Format: screen recording + voiceover. Unlisted YouTube is fine. Keep it tight.

---

## 0:00–0:20 — Hook + problem
**Visual:** the three sample clips playing (traffic / kitten / office), four caption chips fading in per clip.
**VO:** "Every short video needs captions — for accessibility, for marketing, for reach. Track 2 asks for four *very different* tones of the same clip, and it scores you on two things at once: are you *accurate*, and does the *tone* land. Those pull against each other. Here's how we win both."

## 0:20–0:50 — The core idea: an ensemble that sees more, and lies less
**Visual:** animated diagram — one clip → keyframes → three model boxes (GPT‑5.5, Gemini 3.1 Pro, Claude Opus 4.5) each emitting a list of details → a "writer" box merging them → four styled captions.
**VO:** "Instead of one model, three frontier vision models watch the frames independently and each lists everything they see. A writer then cross‑references them: a detail two models agree on is trusted; a lone, unverifiable guess is dropped. Detection comes from the *union* of what they see. Precision comes from their *agreement*."

## 0:50–1:20 — Proof, on the judge's own clips
**Visual:** the benchmark table (BENCHMARK_LOG.md) highlighting: ENSEMBLE 0.942 accuracy, ~14.8 verified details/caption, ~3× any single model. Then the v1 caption zooming in on "KOREA ILLIES ENGINEERING", "Starbucks", "TAXPARK INSURANCE".
**VO:** "We didn't grade ourselves on a friendly judge. We built an adversarial *vision* audit — agents that read the real frames and try to *refute* every claim — and ran it on the fifteen clips in AMD's own bucket, the exact distribution the jury samples. The ensemble scores 0.94 accuracy with about fifteen verified details per caption — roughly three times any single model. It even reads a street sign none of the models get right alone, by agreement."

## 1:20–1:45 — It's not a demo, it's a product
**Visual:** the upload web app — paste a URL / drop a file → four captions appear. Then the side‑by‑side model comparison page.
**VO:** "It runs on *any* video, not just the samples — paste a URL or upload a file and get all four styles. Everything is a reproducible Docker image: read /input, write /output, no keys baked into the repo."

## 1:45–2:10 — Use of AMD
**Visual:** the JupyterLab terminal on the AMD Radeon/ROCm box — `rocm-smi`, then Qwen2.5‑VL loading weights 824/824, then the MI300X notebook.
**VO:** "And it's built for AMD. We stood Qwen‑VL up on AMD ROCm to caption real video the hosted APIs won't ingest, with a turnkey notebook to scale it to a single MI300X — 192 gigs, no sharding."

## 2:10–2:30 — Close
**Visual:** the four rich captions on screen, then the title card + repo URL.
**VO:** "Accurate, detailed, four distinct voices, verified on the real clips, deployable anywhere. That's Track 2 Captioner."

---

### Shot list / assets to capture
1. Cockpit `docs/demo-results.html` (the four styled captions, scored).
2. `docs/comparison.html` (six models side by side).
3. `docs/official.html` (captions on the official jury clips).
4. `eval/BENCHMARK_LOG.md` table.
5. The upload app at `:8799` — one live URL run.
6. AMD JupyterLab terminal: `rocm-smi`, Qwen weights loading.
7. `docker pull ghcr.io/theskygold/track2-captioner:latest` + a run writing results.json.
