#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

: "${REGULAR_MODEL_GGUF:?Set REGULAR_MODEL_GGUF to the Gemma 4 primary GGUF path}"
: "${MTP_DRAFT_GGUF:?Set MTP_DRAFT_GGUF to the Gemma 4 MTP draft GGUF path}"
: "${DIFFUSION_GGUF:?Set DIFFUSION_GGUF to the DiffusionGemma GGUF path}"

GPU_DEVICE="${GPU_DEVICE:-CUDA0}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
DIFFUSION_SERVER="${DIFFUSION_SERVER:-llama-diffusion-gemma-server}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
RUN_ID="${RUN_ID:-gemma-bakeoff-$(date -u +%Y%m%dT%H%M%SZ)}"

REGULAR_PORT="${REGULAR_PORT:-18082}"
MTP_PORT="${MTP_PORT:-18083}"
DIFFUSION_PORT="${DIFFUSION_PORT:-18081}"

REGULAR_MODEL_NAME="${REGULAR_MODEL_NAME:-$(basename "${REGULAR_MODEL_GGUF%.gguf}")}"
MTP_MODEL_NAME="${MTP_MODEL_NAME:-gemma-4-mtp}"
DIFFUSION_MODEL_NAME="${DIFFUSION_MODEL_NAME:-$(basename "${DIFFUSION_GGUF%.gguf}")}"

python scripts/kill_port.py "$REGULAR_PORT" || true
python scripts/kill_port.py "$MTP_PORT" || true
python scripts/kill_port.py "$DIFFUSION_PORT" || true

export GEMMA4_REGULAR_BASE_URL="http://127.0.0.1:${REGULAR_PORT}/v1"
export GEMMA4_REGULAR_MODEL="$REGULAR_MODEL_NAME"
export GEMMA4_REGULAR_READY_URL="http://127.0.0.1:${REGULAR_PORT}/v1/models"
export GEMMA4_REGULAR_START_COMMAND="\"$LLAMA_SERVER\" -m \"$REGULAR_MODEL_GGUF\" --alias \"$REGULAR_MODEL_NAME\" --host 127.0.0.1 --port ${REGULAR_PORT} -c 4096 -ngl 99 --device ${GPU_DEVICE} --split-mode none --flash-attn on -b 8192 -ub 2048"
export GEMMA4_REGULAR_STOP_COMMAND="python \"$ROOT_DIR/scripts/kill_port.py\" ${REGULAR_PORT}"

export GEMMA4_MTP_BASE_URL="http://127.0.0.1:${MTP_PORT}/v1"
export GEMMA4_MTP_MODEL="$MTP_MODEL_NAME"
export GEMMA4_MTP_READY_URL="http://127.0.0.1:${MTP_PORT}/v1/models"
export GEMMA4_MTP_START_COMMAND="\"$LLAMA_SERVER\" -m \"$REGULAR_MODEL_GGUF\" --alias \"$MTP_MODEL_NAME\" --host 127.0.0.1 --port ${MTP_PORT} -c 4096 -ngl 99 --device ${GPU_DEVICE} --split-mode none --flash-attn on -b 8192 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 2 --model-draft \"$MTP_DRAFT_GGUF\" --spec-draft-ngl 99 --spec-draft-device ${GPU_DEVICE}"
export GEMMA4_MTP_STOP_COMMAND="python \"$ROOT_DIR/scripts/kill_port.py\" ${MTP_PORT}"

export DIFFUSIONGEMMA_BASE_URL="http://127.0.0.1:${DIFFUSION_PORT}/v1"
export DIFFUSIONGEMMA_MODEL="$DIFFUSION_MODEL_NAME"
export DIFFUSIONGEMMA_READY_URL="http://127.0.0.1:${DIFFUSION_PORT}/v1/models"
export DIFFUSIONGEMMA_START_COMMAND="\"$DIFFUSION_SERVER\" -m \"$DIFFUSION_GGUF\" --host 127.0.0.1 --port ${DIFFUSION_PORT} -c 4096 -ngl 99 --device ${GPU_DEVICE} --split-mode none --flash-attn on -b 8192 -ub 2048 --diffusion-steps 8 --diffusion-block-length 256"
export DIFFUSIONGEMMA_STOP_COMMAND="python \"$ROOT_DIR/scripts/kill_port.py\" ${DIFFUSION_PORT}"

echo "GPU_DEVICE=${GPU_DEVICE}"
echo "RUN_ID=${RUN_ID}"
echo "Regular=${REGULAR_MODEL_GGUF}"
echo "MTP draft=${MTP_DRAFT_GGUF}"
echo "Diffusion=${DIFFUSION_GGUF}"
python -m aireceipes bakeoff --run-id "$RUN_ID" --output-dir "$OUTPUT_DIR"
