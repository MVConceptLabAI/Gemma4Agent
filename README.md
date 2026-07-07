# Gemma4Agent RouteZero

Hybrid Token-Efficient Routing Agent for AMD Developer Hackathon ACT II — Track 1.

Gemma4Agent RouteZero is packaged for the official Track 1 validation flow: the container reads `/input/tasks.json`, answers every task, and writes `/output/results.json` before exiting with code `0`.

The default submission path is deliberately token-efficient:

- deterministic local solvers handle simple arithmetic and sentiment tasks with zero remote tokens;
- low-confidence or knowledge-heavy tasks fall back to the sponsor-provided Fireworks-compatible endpoint;
- fallback calls are restricted to `ALLOWED_MODELS` and always use `FIREWORKS_BASE_URL`;
- the optional Gemma 4 / MTP / DiffusionGemma benchmark-router MCP technology from `feat/benchmark-router-mcp` remains included, but only as off-contest tooling.

## Competition container contract

The Docker image defaults to the Track 1 agent:

```bash
docker build --platform linux/amd64 -t gemma4agent-track1:local .
docker run --rm \
  -v "$PWD/samples/track1:/input:ro" \
  -v "$PWD/output:/output" \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  gemma4agent-track1:local
```

Expected output:

```text
/output/results.json
```

`results.json` is a JSON array with one object per input task:

```json
[
  {"task_id": "math-1", "answer": "42"}
]
```

A non-scored `/output/route_audit.json` is also written for transparency during local smoke tests. The official scorer should only need `/output/results.json`.

## Required environment variables

| Variable | Purpose |
|---|---|
| `FIREWORKS_API_KEY` | Bearer token for fallback model calls. |
| `FIREWORKS_BASE_URL` | Base URL for all fallback calls, for example `https://api.fireworks.ai/inference/v1`. |
| `ALLOWED_MODELS` | Comma-separated model allowlist supplied by the harness. The router prefers an allowed Gemma model when present, otherwise uses the first allowed model. |

If these are absent, local smoke tests still write a valid result for every task; unknown tasks receive `I don't know.` instead of making any remote call. In official validation, the harness is expected to provide the variables.

## Local validation commands

```bash
# Unit tests
UV_LINK_MODE=copy uv run pytest -q

# Docker smoke without secrets: local solvers answer math/sentiment and fallback-safe unknown tasks.
rm -rf output && mkdir -p output
docker build --platform linux/amd64 -t gemma4agent-track1:local .
docker run --rm \
  -v "$PWD/samples/track1:/input:ro" \
  -v "$PWD/output:/output" \
  gemma4agent-track1:local
python -m json.tool output/results.json
```

With Fireworks credentials:

```bash
docker run --rm \
  -v "$PWD/samples/track1:/input:ro" \
  -v "$PWD/output:/output" \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  gemma4agent-track1:local
```

## Docker Compose

The `competition` service is the submission path:

```bash
docker compose build competition
docker compose run --rm competition
```

The `benchmark` service is optional and off contest:

```bash
docker compose run --rm benchmark list
```

## Off-contest option: local Gemma 4 benchmark and route MCP

This repository keeps the installation and local benchmark technology as an optional add-on, not as the default validation flow.

Included optional components:

- CLI: `aireceipes list`, `aireceipes run`, `aireceipes bakeoff`, `aireceipes route`.
- Benchmark recipes for regular Gemma 4, Gemma 4 MTP, DiffusionGemma, Ollama/OpenAI-compatible smoke tests.
- Matrix launcher: `scripts/run_gemma_matrix.py`.
- Portable benchmark packaging: `scripts/package_portable_benchmark.py`.
- MCP JSONL shim: `python -m aireceipes.mcp_server`.
- MCP config: `mcp/aireceipes-benchmark-router.json`.
- Reusable agent skill: `skills/aireceipes-benchmark-router/SKILL.md`.

Install for off-contest local exploration:

```bash
uv sync --dev
uv run aireceipes list
uv run aireceipes run inference.echo-smoke --output-dir runs
uv run aireceipes bakeoff --no-lifecycle --output-dir runs --run-id already-running
```

For the portable Gemma 4 / MTP / DiffusionGemma matrix and ZIP workflow, see [`PORTABLE_BENCHMARK.md`](PORTABLE_BENCHMARK.md). Model binaries, local `runs/`, virtual environments, caches, and GGUF files are intentionally excluded from Git and Docker context.

## Architecture

```text
/input/tasks.json
      |
      v
src/aireceipes/track1_agent.py
      |-- classify task category
      |-- local zero-token solvers for confident tasks
      |-- Fireworks fallback through FIREWORKS_BASE_URL for uncertain tasks
      v
/output/results.json
```

The benchmark-router MCP remains available for developers who want to generate route plans from local Gemma-family performance reports, but the competition image does not require local GGUF downloads or GPU servers to answer the official Track 1 input file.

## Repository hygiene

- Branch for validation: `main`.
- Default image platform: `linux/amd64`.
- License: MIT.
- Ignored artifacts: `.venv/`, `.pytest_cache/`, `__pycache__/`, `dist/`, `runs/`, `models/`, `*.gguf`, `.env`.
