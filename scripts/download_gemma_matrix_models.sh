#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
mkdir -p "$MODEL_ROOT/gemma-4-12B-it-qat-GGUF" "$MODEL_ROOT/gemma-4-26B-A4B-it-GGUF" "$MODEL_ROOT/diffusiongemma-26B-A4B-it-GGUF"

: "${HF:=hf}"

$HF download unsloth/gemma-4-12B-it-qat-GGUF \
  gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  mtp-gemma-4-12B-it.gguf \
  --local-dir "$MODEL_ROOT/gemma-4-12B-it-qat-GGUF"

$HF download unsloth/gemma-4-26B-A4B-it-GGUF \
  gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  mtp-gemma-4-26B-A4B-it.gguf \
  --local-dir "$MODEL_ROOT/gemma-4-26B-A4B-it-GGUF"

$HF download unsloth/diffusiongemma-26B-A4B-it-GGUF \
  diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  --local-dir "$MODEL_ROOT/diffusiongemma-26B-A4B-it-GGUF"

cat <<EOF
Downloaded matrix models under: $MODEL_ROOT

Suggested environment:
export GEMMA12_MODEL_GGUF="$MODEL_ROOT/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
export GEMMA12_MTP_DRAFT_GGUF="$MODEL_ROOT/gemma-4-12B-it-qat-GGUF/mtp-gemma-4-12B-it.gguf"
export GEMMA26_MODEL_GGUF="$MODEL_ROOT/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
export GEMMA26_MTP_DRAFT_GGUF="$MODEL_ROOT/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it.gguf"
export DIFFUSION_GGUF="$MODEL_ROOT/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
EOF
