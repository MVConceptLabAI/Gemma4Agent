# Track 2 Runbook

## Current Hardening

- `/input/tasks.json` is parsed with Pydantic before execution.
- `/output/results.json` is validated with Pydantic before writing.
- All four mandatory styles are always emitted:
  `formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`.
- Timeouts, provider failures, and missing API keys produce non-empty styled fallbacks.
- `humorous_non_tech` is filtered for obvious technical jargon.
- Frame extraction is hybrid by default: scene-change frames first, then uniform sampling.
- Audio transcription is optional via Groq Whisper and feeds the VLM describe prompt.
- Fireworks VLM/style calls support comma-separated fallback model chains.
- Groq and OpenRouter can be used as temporary provider fallbacks through
  `PROVIDER_ORDER`; `DESCRIBE_PROVIDER_ORDER` and `STYLE_PROVIDER_ORDER` can split
  video understanding from style writing when quality mode is needed.
- Optional direct video/audio understanding is available through `DIRECT_VIDEO_MODEL`
  for dedicated Fireworks multimodal deployments; it falls back to frames when unset
  or unavailable.
- v2 assets are integrated: offline mock run, zero-cost self-check, dataset-v2 builder,
  balanced scene generator, LoRA training/deploy notes, Makefile, and submission notes.

## Local Checks

Run the no-network contract guard:

```bash
python scripts/contract_test.py
```

Run the full offline preflight evidence pack:

```bash
python scripts/preflight.py
```

Run a local degraded contract check without spending model credits:

```bash
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/results_contract.json \
PER_TASK_TIMEOUT_S=1 \
FIREWORKS_API_KEY= \
python -m app.main
```

Run the full Docker smoke test when `FIREWORKS_API_KEY` is available:

```bash
export FIREWORKS_API_KEY=fw_xxx
bash scripts/smoke_test.sh
```

Run the temporary live Groq path without waiting for Fireworks:

```bash
PROVIDER_ORDER=groq,fireworks,openrouter \
DESCRIBE_PROVIDER_ORDER=groq,openrouter,fireworks \
STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=5 \
FRAME_MAX_EDGE=512 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/groq_results_final.json \
python -m app.main

python eval/self_check.py --results out/groq_results_final.json
python eval/quality_audit.py --results out/groq_results_final.json
python scripts/quality_gate.py \
  --results out/demo_quality_results.json \
  --scores eval/scores_quality_openrouter.json
```

Run the best measured quality profile on the public sample:

```bash
PROVIDER_ORDER=openrouter,groq,fireworks \
DESCRIBE_PROVIDER_ORDER=openrouter,groq,fireworks \
STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=8 \
FRAME_MAX_EDGE=640 \
DESCRIBE_MAX_TOKENS=900 \
STYLE_MAX_TOKENS=180 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/demo_quality_results.json \
python -m app.main

python eval/self_check.py --results out/demo_quality_results.json
python eval/quality_audit.py --results out/demo_quality_results.json --strict
python eval/detail_audit.py --results out/demo_quality_results.json --strict
python eval/local_judge.py \
  --results out/demo_quality_results.json \
  --clips eval/clips.json \
  --out eval/scores_quality_openrouter.json
python scripts/quality_gate.py \
  --results out/demo_quality_results.json \
  --scores eval/scores_quality_openrouter.json
```

Run the strict pre-submit gate once Docker is running and credentials are set:

```bash
python scripts/preflight.py --strict --docker-build --docker-run
make submit-check
```

Publish and verify the public image:

```bash
export PUBLIC_IMAGE=ghcr.io/<user>/track2-captioner:final
make publish
make verify-public
```

## Runtime Knobs

| Variable | Default | Purpose |
|---|---:|---|
| `FIREWORKS_API_KEY` | empty | Required for real VLM/style calls. |
| `GROQ_API_KEY` | empty | Enables optional Whisper transcription. |
| `OPENROUTER_API_KEY` | empty | Enables OpenRouter fallback. |
| `PROVIDER_ORDER` | `groq,fireworks,openrouter` | Default provider priority. |
| `DESCRIBE_PROVIDER_ORDER` | `PROVIDER_ORDER` | Provider priority for video understanding. |
| `STYLE_PROVIDER_ORDER` | `PROVIDER_ORDER` | Provider priority for style captions. |
| `VLM_MODEL` | `accounts/fireworks/models/qwen2p5-vl-7b-instruct` | Describe-stage VLM. |
| `VLM_FALLBACK_MODELS` | empty | Comma-separated describe fallback models. |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq vision model. |
| `OPENROUTER_VLM_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter vision model. |
| `DESCRIBE_MAX_TOKENS` | `700` | Token budget for rich scene-facts JSON. |
| `DIRECT_VIDEO_MODEL` | empty | Optional dedicated video/audio model deployment id. |
| `DIRECT_VIDEO_MAX_SECONDS` | `60` | Max seconds sent to direct video/audio path. |
| `STYLE_MODEL` | `accounts/fireworks/models/gemma-3-27b-it` | Four style rewrites, Gemma bonus path. |
| `STYLE_LORA` | empty | Optional deployed LoRA model id for style rewrites. |
| `STYLE_FALLBACK_MODELS` | empty | Comma-separated style fallback models. |
| `GROQ_STYLE_MODEL` | `llama-3.3-70b-versatile` | Groq style model. |
| `OPENROUTER_STYLE_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter style fallback. |
| `STYLE_MAX_TOKENS` | `140` | Token budget for one styled caption. |
| `EVIDENCE_LOCK_ENABLED` | `0` | Optional candidate/repair mode that rejects visually thin captions during A/B runs. |
| `STYLE_CANDIDATES` | `2` | Candidate count when evidence-lock mode is enabled. |
| `STYLE_REPAIR_ENABLED` | `1` | Enables model and deterministic repair for evidence-lock captions. |
| `JUDGE_PROVIDER_ORDER` | `fireworks,openrouter,groq` | Provider priority for local LLM-judge proxy. |
| `GROQ_JUDGE_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq judge fallback. |
| `OPENROUTER_JUDGE_MODEL` | `qwen/qwen3-vl-8b-instruct` | OpenRouter judge fallback. |
| `AUDIO_TRANSCRIBE_ENABLED` | `1` | Uses Groq Whisper when `GROQ_API_KEY` is set. |
| `WHISPER_MODEL` | `whisper-large-v3-turbo` | Groq transcription model. |
| `NUM_FRAMES` | `8` | Target frames sent to the VLM. |
| `FRAME_MAX_EDGE` | `720` | Max frame edge before upload. |
| `SCENE_DETECT_ENABLED` | `1` | Enable scene-change sampling before uniform fill. |
| `SCENE_THRESHOLD` | `0.35` | FFmpeg scene threshold. |
| `MAX_CONCURRENCY` | `3` | Parallel clips. |
| `PER_TASK_TIMEOUT_S` | `25` | Per-clip hard timeout. |

For the current Groq key/plan, the safest live settings are `MAX_CONCURRENCY=1`,
`NUM_FRAMES=5`, and `FRAME_MAX_EDGE=512`. Groq vision currently accepts up to
five images per request, so the pipeline caps Groq describe calls accordingly.

## Submit Gates

- Build with `docker buildx build --platform linux/amd64`.
- Confirm public image pull works anonymously.
- Confirm compressed image size is under 10 GB.
- Confirm startup is under 60 s.
- Confirm a 12-clip run is under 10 min.
- Confirm every internal clip finishes under the 25 s target.
- Confirm `out/results.json` has four non-empty English captions per task.
- Run mirror judging before burning official submissions.
- Push `PUBLIC_IMAGE` to GHCR/Docker Hub and verify anonymous pull.

## v2 Offline Tools

```bash
python scripts/mock_run.py --tasks data/sample_tasks.json --out out/mock_results.json
python eval/self_check.py --results out/mock_results.json
python eval/quality_audit.py --results out/mock_results.json
python finetune/train_gemma_lora.py --dataset finetune/dataset_v2.jsonl --dry-run
```

Run the provider-agnostic local judge when at least one judge key is available:

```bash
JUDGE_PROVIDER_ORDER=openrouter,groq,fireworks \
python eval/local_judge.py \
  --results out/demo_quality_results.json \
  --clips eval/clips.json \
  --out eval/scores_quality_openrouter.json
```
