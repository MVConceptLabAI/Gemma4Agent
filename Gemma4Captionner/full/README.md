# Track 2 Captioner - AMD Developer Hackathon ACT II

Dockerized video-captioning agent for Track 2. It reads `/input/tasks.json`,
generates four styled English captions per clip, and writes
`/output/results.json`.

## Run it in 60 seconds (judges start here)

```bash
docker pull ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v2
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v2
```

Public linux/amd64 image. `/input/tasks.json` in, `/output/results.json` out,
with all four requested styles present and validated. The public submission
reference is `ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v2`.

## Caption YOUR own video (beyond the sample clips)

```bash
pip install -r requirements.txt
python -m app.webapp        # -> http://127.0.0.1:8799
```

Paste any direct `.mp4` URL or upload a local file; get the four styled
captions in about a minute. Uses the full ensemble when API credits are
available and degrades automatically to the free-tier pipeline when they are
not — verified end-to-end on out-of-sample videos (`eval/BENCHMARK_LOG.md`
includes 12 Pexels stress clips beyond the official set).

## Architecture

**Submission engine = a vision-model ENSEMBLE** (`CAPTION_ENGINE=ensemble`).

1. Download each MP4 and extract keyframes (scene-change + uniform sampling).
2. **Three frontier vision models observe the frames independently** — GPT-5.5,
   Gemini 3.1 Pro, Claude Opus 4.5 — each returning an exhaustive detail list.
3. A **writer (Gemma 4 31B) cross-references all three lists**: a detail 2+ models
   agree on is trusted; a lone specific claim is dropped unless corroborated.
4. It writes the four required styles: `formal`, `sarcastic`, `humorous_tech`,
   `humorous_non_tech`. Normalized + Pydantic-validated before output.

This `gemma4-full-w4` experiment keeps the v40 observers and W4 per-style
writer split unchanged; only the writer model is Gemma 4 31B.

Detection comes from the *union* of what the models see; precision from their
*agreement*. Measured on the 15 official AMD sample clips via an adversarial
vision audit: **0.942 accuracy, ~14.8 verified details/caption**, ~3× a single
model — see `eval/BENCHMARK_LOG.md`.

A single-model fallback (`CAPTION_ENGINE=pipeline`: Qwen3-VL-235B describe →
writer) is available for lower-cost runs. The runtime degrades gracefully:
provider failover, non-empty styled fallbacks, never malformed output.

An experimental direct profile is available only when explicitly selected with
`CAPTION_ENGINE=qwen_direct`. It sends four uniform 1024px frames to four
independent Qwen multimodal calls (one per style), then falls back to the legacy
pipeline for any missing style. The Docker default remains `ensemble`.

**Dossier:** `SUBMISSION.md`, `docs/video_script.md`, `docs/slides.html`,
`docs/comparison.html` (models side by side), `docs/official.html` (captions on
the jury clips). Test any video: `python -m app.webapp` → upload or paste a URL.

## Quickstart

Offline preflight, no API keys required:

```bash
python scripts/preflight.py
```

Build and run the Docker contract check:

```bash
python scripts/preflight.py --docker-build --docker-run
```

Real inference run:

```bash
export FIREWORKS_API_KEY=fw_xxx
python scripts/preflight.py --strict --docker-build --docker-run
make submit-check
```

Run with OpenRouter first and Fireworks as fallback:

```bash
PROVIDER_ORDER=openrouter,fireworks \
DESCRIBE_PROVIDER_ORDER=openrouter,fireworks \
STYLE_PROVIDER_ORDER=openrouter,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=5 \
FRAME_MAX_EDGE=512 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/openrouter_results_final.json \
python -m app.main

python eval/self_check.py --results out/openrouter_results_final.json
python eval/quality_audit.py --results out/openrouter_results_final.json
python scripts/quality_gate.py --results out/demo_quality_results.json --scores eval/scores_quality_openrouter.json
```

Best measured quality profile on the public sample:

```bash
PROVIDER_ORDER=openrouter,fireworks \
DESCRIBE_PROVIDER_ORDER=openrouter,fireworks \
STYLE_PROVIDER_ORDER=openrouter,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=8 \
FRAME_MAX_EDGE=640 \
DESCRIBE_MAX_TOKENS=900 \
STYLE_MAX_TOKENS=180 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/demo_quality_results.json \
python -m app.main
```

The describe prompt asks for conservative fine-grained observations: approximate
traffic quantities, animal type without breed guesses, leaf/sunlight detail,
jewelry, hand position, nail color, cables, peripherals, and nearby objects.
Uncertain details are kept out of final captions unless they are visually clear.

Publish and verify a public image:

```bash
export PUBLIC_IMAGE=ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v2
make publish
make verify-public
```

## I/O Contract

Input: `/input/tasks.json`

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Output: `/output/results.json`

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Runtime Variables

Copy `.env.example` to `.env` for local development. Do not commit real keys.

| Variable | Default | Purpose |
|---|---|---|
| `CAPTION_ENGINE` | Docker: `ensemble`; source: `pipeline` | Select `ensemble`, `pipeline`, or the opt-in `qwen_direct` candidate. |
| `FIREWORKS_API_KEY` | empty | Required for real VLM/style inference. |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | OpenAI-compatible Fireworks endpoint. |
| `QWEN_DIRECT_MODEL` | `accounts/fireworks/models/qwen3p7-plus` | Model used only by `CAPTION_ENGINE=qwen_direct`. |
| `QWEN_DIRECT_HTTP_TIMEOUT_S` | `45` | Per-request network timeout for the opt-in direct engine. |
| `QWEN_DIRECT_STYLE_CONCURRENCY` | `1` | Shared Fireworks request limit across direct-engine styles and videos. |
| `QWEN_DIRECT_RETRY_MAX_DELAY_S` | `4` | Cap for direct-engine 429/transport retry backoff. |
| `QWEN_DIRECT_RETRY_JITTER_CAP_S` | `0.35` | Maximum random jitter added to direct-engine retries. |
| `QWEN_DIRECT_PROMPT_PROFILE` | `v1` | Select `strong_v2` for the opt-in short, conservative, strongly styled prompt ablation. |
| `PROVIDER_ORDER` | `openrouter,fireworks` | Default provider priority. |
| `DESCRIBE_PROVIDER_ORDER` | `PROVIDER_ORDER` | Provider priority for video understanding. |
| `STYLE_PROVIDER_ORDER` | `PROVIDER_ORDER` | Provider priority for style caption writing. |
| `VLM_MODEL` | `accounts/fireworks/models/qwen2p5-vl-7b-instruct` | Describe-stage VLM. |
| `VLM_FALLBACK_MODELS` | empty | Comma-separated describe-stage fallback models. |
| `DIRECT_VIDEO_MODEL` | empty | Optional dedicated Fireworks video/audio model deployment. |
| `DIRECT_VIDEO_MAX_SECONDS` | `60` | Max seconds sent to the direct video/audio path. |
| `OPENROUTER_API_KEY` | empty | Enables OpenRouter fallback. |
| `OPENROUTER_VLM_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter describe-stage model. |
| `DESCRIBE_MAX_TOKENS` | `700` | Token budget for rich scene-facts JSON. |
| `STYLE_MODEL` | `accounts/fireworks/models/gemma-3-27b-it` | Style rewriter, Gemma bonus path. |
| `STYLE_LORA` | empty | Optional deployed LoRA model id. |
| `STYLE_FALLBACK_MODELS` | empty | Comma-separated style fallback models. |
| `OPENROUTER_STYLE_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter style fallback. |
| `STYLE_MAX_TOKENS` | `140` | Token budget for one styled caption. |
| `EVIDENCE_LOCK_ENABLED` | `0` | Enables experimental candidate/repair pass against visual evidence. |
| `STYLE_CANDIDATES` | `2` | Candidate count when evidence-lock mode is enabled. |
| `STYLE_REPAIR_ENABLED` | `1` | Enables model/deterministic repair for thin evidence-lock captions. |
| `JUDGE_PROVIDER_ORDER` | `fireworks,openrouter` | Provider priority for local LLM-judge proxy. |
| `FIREWORKS_JUDGE_MODEL` | `accounts/fireworks/models/qwen3p7-plus` | Fireworks judge model when available. |
| `OPENROUTER_JUDGE_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter judge fallback. |
| `NUM_FRAMES` | `8` | Target number of keyframes. |
| `FRAME_MAX_EDGE` | `720` | Max frame edge before upload. |
| `VIDEO_CONTEXT_OBSERVER` | `google/gemma-4-31b-it` | Gemma receives a compact source video after the visual panel creates grounding context. |
| `VIDEO_CONTEXT_MAX_BYTES` | `14000000` | Upper bound for the compact video (with audio track) sent to Gemma. |
| `SCENE_DETECT_ENABLED` | `1` | Enable scene-change sampling before uniform fill. |
| `MAX_CONCURRENCY` | `3` | Parallel clips. |
| `PER_TASK_TIMEOUT_S` | `25` | Per-clip hard timeout. |

## Validation

Local gates:

```bash
python scripts/contract_test.py
python scripts/mock_run.py --tasks data/sample_tasks.json --out out/mock_results.json
python eval/self_check.py --results out/mock_results.json
python eval/quality_audit.py --results out/mock_results.json
python eval/detail_audit.py --results out/demo_quality_results.json
python finetune/train_gemma_lora.py --dataset finetune/dataset_v2.jsonl --dry-run
```

Current status and remaining submission gates are tracked in `SUBMIT_STATUS.md`.
Prompt/skill/plugin research is tracked in `docs/quality-research.md`.

Interactive project briefing:

```text
docs/mission-control.html
```

## Repository Layout

- `app/` - container runtime.
- `data/` - sample tasks.
- `docs/` - interactive briefing and project poster.
- `eval/` - zero-cost audits and provider-agnostic local judge.
- `finetune/` - synthetic dataset, scene generator, LoRA training/deploy notes.
- `scripts/` - preflight, Docker build/run/publish helpers.
- `SUBMISSION.md` - lablab.ai submission copy.
- `RUNBOOK.md` - operational commands.

## Notes

- Build target is `linux/amd64`.
- The image does not bake credentials.
- Generated `out/`, `in/`, caches, zips, and flattened duplicate exports are
  ignored by git.
