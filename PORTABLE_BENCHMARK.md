# Portable Gemma 4 / MTP / DiffusionGemma GPU bakeoff

This package contains the AiReceipes benchmark runner, recipes, report generator, tests, and helper scripts needed to run the same 3-model bakeoff on another machine/GPU.

Model binaries are **not** included in the ZIP. Put the GGUF files on the target machine and point the script at them.

## What the benchmark runs

All three variants receive the same 4 prompts, repeated 3 times:

| Case | Vertical | Evaluation checks |
|---|---|---|
| `translation.fr.simple` | translation | French translation must contain `chat`, `dort`, `table`, `bleue`; must not contain `explanation` or the English source. |
| `chat.thanks` | chat | One short sentence thanking a teammate; must contain `thank` and `test`; must end with punctuation. |
| `code.python.add` | code | Python `add(a, b)` function; must contain `def add`, `return`, match `return a + b` / `return b + a`, and avoid Markdown fences. |
| `agentic.cleanup.plan` | agentic | Exactly numbered safe-cleanup plan with `1.`, `2.`, `3.`, final `ANSWER: ready`; must not include `rm -rf /`. |

Each case produces `accuracy`, `case_success_rate`, latency, TTFT, token counts, and prompt-level `effective_tokens_per_sec`.

`effective_tokens_per_sec = (completion_tokens_total * 1000 / latency_ms_total) * quality_factor`

For checked prompts, `quality_factor` is the prompt accuracy. This means a very fast model is penalized if it fails the checks.

## Model versions used in the reference A6000 run

- Regular Gemma 4: `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
- MTP primary: `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
- MTP draft: `mtp-gemma-4-12B-it.gguf`
- DiffusionGemma: `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`

The enhanced report extracts exact `.gguf` filenames from the server start command and includes them in `comparison.json` and `comparison.html`.

## Target machine setup

```bash
cd AiReceipes
python -m venv .venv
source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
pip install -e . pytest
pytest
```

You also need compatible CUDA llama.cpp binaries:

- `llama-server` for regular Gemma 4 and MTP.
- `llama-diffusion-gemma-server` for DiffusionGemma.

## Run on another GPU

Set paths and the GPU device, then run the compact 3-model launcher:

```bash
export GPU_DEVICE=CUDA0
export LLAMA_SERVER=/path/to/llama-server
export DIFFUSION_SERVER=/path/to/llama-diffusion-gemma-server

export REGULAR_MODEL_GGUF=/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
export MTP_DRAFT_GGUF=/models/mtp-gemma-4-12B-it.gguf
export DIFFUSION_GGUF=/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf

bash scripts/run_gemma_bakeoff.sh
```

## Full matrix requested

The full matrix launcher generates unique recipes and a consolidated report for:

- Regular autoregressive Gemma 4.
- MTP with `--spec-draft-n-max` = `2`, `4`, and `6`.
- DiffusionGemma via `llama-diffusion-gemma-server`.
- Optional external Unsloth Studio/OpenAI-compatible DiffusionGemma endpoint.
- Thinking and non-thinking variants (`--reasoning on/off` for llama.cpp regular/MTP; payload-level reasoning flag for endpoints that support it).
- Context sizes `32k`, `64k`, `128k`, and `256k`.
- 12B QAT and 26B-A4B Gemma 4 regular/MTP models when both model paths are present.

Download the reference GGUFs:

```bash
bash scripts/download_gemma_matrix_models.sh
```

Or set paths manually:

```bash
export GEMMA12_MODEL_GGUF=/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
export GEMMA12_MTP_DRAFT_GGUF=/models/mtp-gemma-4-12B-it.gguf
export GEMMA26_MODEL_GGUF=/models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
export GEMMA26_MTP_DRAFT_GGUF=/models/mtp-gemma-4-26B-A4B-it.gguf
export DIFFUSION_GGUF=/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```

Plan the complete matrix without launching models:

```bash
python scripts/run_gemma_matrix.py --dry-run --run-id matrix-plan
```

Run the complete matrix:

```bash
export GPU_DEVICE=CUDA0
export TARGET_GPU="RTX 6000 / CUDA0"
export LLAMA_SERVER=/path/to/llama-server
export DIFFUSION_SERVER=/path/to/llama-diffusion-gemma-server
python scripts/run_gemma_matrix.py --result runs --run-id matrix-full
```

The default 72-spec matrix now runs in lifecycle order by default: specs are grouped by backend/model/mode before context/reasoning to reduce unnecessary model-load churn. Use `--orchestration-order matrix` to keep the older generation order. Adjacent recipes with the same explicit `runtime.lifecycle_key` / `runtime.model_memory_key` reuse the already-loaded server; otherwise the runner stops the current server and waits for the readiness URL to go down before starting the next server.

Per-test wrapper hook: edit `scripts/wrap_each_test.cmd` and add one non-comment command. It is executed before every test/recipe. The command receives `$result` as the concrete suite folder containing all logs/results for the run and `$test_result` as that test's metrics folder. Because Windows `shell=True` uses `cmd.exe`, wrap Bash snippets as `bash -lc '...'`:

```cmd
bash -lc 'mkdir -p "$result/wrapper-logs" && { date; echo "$recipe_index $recipe_id"; nvidia-smi; } >> "$result/wrapper-logs/${recipe_index}-${recipe_id//[^A-Za-z0-9_.-]/-}.log" 2>&1'
```

Or select a different wrapper file:

```bash
python scripts/run_gemma_matrix.py --test-wrapper-cmd scripts/wrap_each_test.cmd --result runs --run-id matrix-full
```

For 256k DiffusionGemma on 48 GB RTX 6000/A6000, default F16 KV cache can exceed VRAM. Retry that slice with quantized KV cache if the lifecycle log shows a ~51,200 MiB KV allocation failure:

```bash
python scripts/run_gemma_matrix.py \
  --mode diffusion \
  --context 262144 \
  --reasoning off --reasoning on \
  --diffusion-extra-args "--cache-type-k q8_0 --cache-type-v q8_0" \
  --run-id matrix-diffusion-256k-q8kv
```

Run only the 26B-A4B matrix:

```bash
python scripts/run_gemma_matrix.py --model 26b --run-id matrix-26b-a4b
```

Include an already-running Unsloth Studio / compatible DiffusionGemma endpoint:

```bash
export UNSLOTH_DIFFUSION_BASE_URL=http://127.0.0.1:18084/v1
export UNSLOTH_DIFFUSION_MODEL=diffusiongemma-26B-A4B-it
python scripts/run_gemma_matrix.py --mode diffusion-unsloth --run-id unsloth-diffusion
```

Run the full 72-row Unsloth-version matrix (regular + MTP + DiffusionGemma slices all reported with the Unsloth-compatible backend):

```bash
python scripts/run_gemma_matrix.py \
  --mode unsloth \
  --unsloth-base-url http://127.0.0.1:18084/v1 \
  --result runs \
  --run-id unsloth-full-72
```

If local Unsloth dependencies are missing and the endpoint is unreachable, the runner does **not** turn all 72 rows into failed prompt attempts. It writes all 72 rows as `skipped preflight` and records the missing dependency/endpoint status in `comparison.html` under `Backend preflight checks`. Use `--force-unsloth-run-on-missing` only when you intentionally want to send prompts despite a failed preflight.

If the local Unsloth Python/API packages are missing and you want the launcher to install them before testing the Unsloth backend, add:

```bash
python scripts/run_gemma_matrix.py \
  --mode diffusion-unsloth \
  --install-unsloth-if-required \
  --unsloth-diffusion-base-url http://127.0.0.1:18084/v1 \
  --run-id unsloth-diffusion
```

`UNSLOTH_PYTHON` can point at a dedicated Unsloth/Studio Python. The install step installs `unsloth`, `torch`, and the API dependencies needed by the Unsloth-compatible endpoint path; it does not invent an endpoint URL, so keep `UNSLOTH_DIFFUSION_BASE_URL` or `--unsloth-diffusion-base-url` set.

Unsloth's DiffusionGemma documentation states that DiffusionGemma can reach `2000+ tokens/s` on an RTX 6000 via Unsloth Studio or llama.cpp; the matrix report records the backend so local llama.cpp results and Unsloth endpoint results remain separated.

Optional overrides:

```bash
export RUN_ID=a100-test-01
export OUTPUT_DIR=runs
export REGULAR_PORT=18082
export MTP_PORT=18083
export DIFFUSION_PORT=18081
```

On a non-default GPU, MTP must set both `--device` and `--spec-draft-device`. The launcher does this automatically:

```text
--device CUDA2 ... --spec-draft-device CUDA2
```

Without the draft device flag, some Windows/CUDA llama.cpp builds abort with:

```text
pre-allocated tensor (cache_k_l46) in a buffer (CUDA2) that cannot run the operation (NONE)
```

## Output

A run writes:

```text
runs/<RUN_ID>-gemma-variant-bakeoff/
  01-inference.gemma4-regular-bakeoff/metrics.json
  02-inference.gemma4-mtp-bakeoff/metrics.json
  03-inference.diffusiongemma-bakeoff/metrics.json
  comparison.json
  comparison.html
  *-lifecycle.log
```

Open `comparison.html` for the full dark report with exact model filenames, metric definitions, prompt-level checks, and effective token/s.

## Regenerate a report without rerunning models

```bash
python scripts/regenerate_comparison.py runs/<RUN_ID>-gemma-variant-bakeoff
```

## Create a portable ZIP

```bash
python scripts/package_portable_benchmark.py --output dist/aireceipes-gemma-bakeoff-portable.zip
```

To include a small example report:

```bash
python scripts/package_portable_benchmark.py \
  --include-run runs/<RUN_ID>-gemma-variant-bakeoff \
  --output dist/aireceipes-gemma-bakeoff-portable-with-example.zip
```
