# Gemma4Agent

What if Gemma 4 could be used for all kind of jobs?

This repository includes the AiReceipes benchmark router skill and MCP server for Gemma 4 / MTP / DiffusionGemma agent workflows. The MCP config is in `mcp/aireceipes-benchmark-router.json`; the reusable agent skill is in `skills/aireceipes-benchmark-router/SKILL.md`; and the dependency-free JSONL MCP shim runs with `python -m aireceipes.mcp_server`.

## AiReceipes benchmark package

Container-friendly, classified AI benchmark recipes for inference, finetuning, and agentic workflows.

The MVP includes runnable inference recipes:

- `inference.echo-smoke` — deterministic text/chat smoke benchmark that proves recipe discovery, command-line execution, KPI collection, and run artifact generation.
- `inference.ollama-chat-smoke` — real OpenAI-compatible `/v1/chat/completions` smoke benchmark for Ollama, llama.cpp server, vLLM, or similar local endpoints. Defaults target `http://localhost:11434/v1` with `gemma4:latest`.
- `inference.gemma4-regular-bakeoff` — regular Gemma 4 vertical benchmark baseline.
- `inference.gemma4-mtp-bakeoff` — same prompts against a Gemma 4 server launched with MTP / draft multi-token prediction.
- `inference.diffusiongemma-bakeoff` — same prompts against a DiffusionGemma endpoint.

The project keeps the original name spelling from `IDEA.md`: **AiReceipes**.

## Quick start

```bash
# From F:/hemes/AiReceipes
uv run --with pytest python -m pytest -q
uv run aireceipes list
uv run aireceipes show inference.echo-smoke
uv run aireceipes run inference.echo-smoke --output-dir runs
uv run aireceipes run inference.ollama-chat-smoke --output-dir runs --run-id local-chat
```

A run writes:

```text
runs/<timestamp>-inference.echo-smoke/metrics.json
```

`metrics.json` contains recipe metadata, success/error counts, per-prompt outputs, vertical breakdowns, and KPIs.

## Gemma 4 / MTP / DiffusionGemma bakeoff

The bakeoff recipes compare three variants on the same vertical tests:

| Recipe | Endpoint env | Model env | Purpose |
|---|---|---|---|
| `inference.gemma4-regular-bakeoff` | `GEMMA4_REGULAR_BASE_URL` | `GEMMA4_REGULAR_MODEL` | Autoregressive baseline |
| `inference.gemma4-mtp-bakeoff` | `GEMMA4_MTP_BASE_URL` | `GEMMA4_MTP_MODEL` | Gemma 4 with server-side MTP enabled |
| `inference.diffusiongemma-bakeoff` | `DIFFUSIONGEMMA_BASE_URL` | `DIFFUSIONGEMMA_MODEL` | DiffusionGemma block-denoising variant |

Example launch shapes:

```bash
# Regular Gemma 4 baseline, no MTP/speculative decoding.
llama-server -hf unsloth/gemma-4-31B-it-GGUF -c 262144 --port 18082

# Gemma 4 MTP. Start with --spec-draft-n-max 2, then sweep 1..6 on your hardware.
# When using a non-default CUDA GPU, set the draft device too.
llama-server -hf unsloth/gemma-4-31B-it-GGUF \
  --device CUDA0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-device CUDA0 \
  -c 262144 \
  --port 18083

# DiffusionGemma: prefer a vLLM build with DiffusionGemma support, or a compatible
# specialized server exposing /v1/chat/completions with streaming enabled.
```

Run the full sequential bakeoff. The command loads one model, waits for readiness,
runs the test, writes that model's `metrics.json`, unloads it, then moves to the
next model. It finally writes `comparison.json` and `comparison.html` for the
three-model comparison.

```bash
# First identify the RTX A6000 index.
nvidia-smi --query-gpu=index,name --format=csv

# Replace <A6000_INDEX> and model names/paths with your local setup.
export GEMMA4_REGULAR_BASE_URL=http://127.0.0.1:18082/v1
export GEMMA4_REGULAR_MODEL=gemma-4-regular
export GEMMA4_REGULAR_START_COMMAND='CUDA_VISIBLE_DEVICES=<A6000_INDEX> llama-server -hf unsloth/gemma-4-31B-it-GGUF -c 262144 --host 127.0.0.1 --port 18082 --n-gpu-layers 99 --split-mode none'

export GEMMA4_MTP_BASE_URL=http://127.0.0.1:18083/v1
export GEMMA4_MTP_MODEL=gemma-4-mtp
export GEMMA4_MTP_START_COMMAND='CUDA_VISIBLE_DEVICES=<A6000_INDEX> llama-server -hf unsloth/gemma-4-31B-it-GGUF --spec-type draft-mtp --spec-draft-n-max 2 -c 262144 --host 127.0.0.1 --port 18083 --n-gpu-layers 99 --split-mode none'

export DIFFUSIONGEMMA_BASE_URL=http://127.0.0.1:18081/v1
export DIFFUSIONGEMMA_MODEL=diffusiongemma-26B-A4B-it
export DIFFUSIONGEMMA_START_COMMAND='CUDA_VISIBLE_DEVICES=<A6000_INDEX> llama-diffusion-gemma-server -m <DIFFUSIONGEMMA_GGUF> --host 127.0.0.1 --port 18081 -c 4096 -ngl 99 --device CUDA0 --split-mode none --flash-attn on --diffusion-steps 8 --diffusion-block-length 256'

uv run aireceipes bakeoff --output-dir runs --run-id a6000-gemma-variant
```

The suite output layout is:

```text
runs/a6000-gemma-variant-gemma-variant-bakeoff/
  01-inference.gemma4-regular-bakeoff/metrics.json
  02-inference.gemma4-mtp-bakeoff/metrics.json
  03-inference.diffusiongemma-bakeoff/metrics.json
  comparison.json
  comparison.html
  *-lifecycle.log
```

If you already started the three endpoints yourself, skip load/unload and only
collect/report results with:

```bash
uv run aireceipes bakeoff --no-lifecycle --output-dir runs --run-id already-running
```

Each bakeoff recipe uses the same verticals: `translation`, `chat`, `code`, and `agentic`.
The runner records:

- `ttft_ms_avg` from streaming time-to-first-token.
- `tokens_per_sec` from completion tokens over decode time (`latency - TTFT` when TTFT is available).
- `end_to_end_tokens_per_sec` from completion tokens over full request latency.
- `accuracy` from simple case checks (`expected_contains`, `expected_regex`, etc.).
- `workflow_success_rate` from `vertical = "agentic"` cases.
- `quality_loss` / `loss_proxy` as `1 - accuracy` when the endpoint does not expose real loss.
- `loss_avg` / `eval_loss_avg` when a server response includes those fields.
- `effective_tokens_per_sec = end_to_end_tokens_per_sec * quality_factor`, where `quality_factor` is scored accuracy when checks exist.

`comparison.html` also includes exact model filenames extracted from the lifecycle start commands, prompt-level check summaries, sample outputs, and per-prompt effective token/s. This makes speed comparisons auditable: a model only gets high effective throughput when it is both fast and passes the prompt checks.

For a portable package that can be moved to another GPU machine, see [`PORTABLE_BENCHMARK.md`](PORTABLE_BENCHMARK.md). The helper launchers are `scripts/run_gemma_bakeoff.sh` for the compact 3-model run and `scripts/run_gemma_matrix.py` for the expanded matrix: MTP draft n = 2/4/6, thinking/non-thinking, 32k/64k/128k/256k contexts, and 12B + 26B-A4B model paths when available. `scripts/package_portable_benchmark.py` creates a ZIP without local model binaries or virtualenvs.

## Categories

Recipes are classified by top-level category and metadata:

- `inference` — model serving, completion quality/latency, throughput, smoke checks.
- `finetuning` — dataset prep, training runs, adapters/checkpoints, eval after training.
- `agentic` — tool-use agents, workflow success, step counts, cost/latency, task completion.

## Recipe format

Recipes are TOML files under `recipes/<category>/`.

```toml
id = "inference.echo-smoke"
category = "inference"
name = "Echo chat latency smoke"
description = "Deterministic smoke benchmark."
tags = ["smoke", "text", "chat"]

[classification]
task = "chat-completions"
modality = "text"
size = "smoke"

[runtime]
adapter = "echo"
container_image = "aireceipes-runner:local"

[parameters]
repetitions = 1

[dataset]
prompts = ["Say HELLO in one word."]

[kpis]
metrics = ["success_rate", "latency_ms_avg", "output_chars_avg"]
```

For scored vertical benchmarks, use `[[dataset.cases]]` instead of `dataset.prompts`:

```toml
[[dataset.cases]]
id = "code.python.add"
vertical = "code"
prompt = "Write only Python code for a function add(a, b) that returns their sum."
expected_contains = ["def add", "return"]
expected_regex = ['return\s+a\s*\+\s*b|return\s*b\s*\+\s*a']
```

## Container usage

```bash
docker compose build
docker compose run --rm runner list
docker compose run --rm runner run inference.echo-smoke --output-dir runs

# Real local model endpoint from Docker Desktop on Windows/macOS:
docker compose run --rm runner run inference.ollama-chat-smoke --output-dir runs --run-id docker-local-chat
```

The compose service mounts `./runs` so KPI artifacts stay on the host. For Docker, the compose file defaults `AIRECEIPES_OPENAI_BASE_URL` to `http://host.docker.internal:11434/v1`; local non-container runs use the recipe default `http://localhost:11434/v1`.

Override the target endpoint/model without editing the recipe:

```bash
AIRECEIPES_OPENAI_BASE_URL=http://localhost:11434/v1 \
AIRECEIPES_OPENAI_MODEL=gemma4:latest \
uv run aireceipes run inference.ollama-chat-smoke --output-dir runs
```

## Next adapters to add

1. `finetune_lora` — run a small LoRA/QLoRA container recipe and collect training/eval KPIs.
2. `agentic_task` — run a real tool-using agent scenario and collect success rate, turns, latency, and cost.
3. `openai_compatible_batch` — run multi-prompt throughput sweeps with concurrency and token KPIs.
