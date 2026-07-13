#!/usr/bin/env bash
set -euo pipefail

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"
IMAGE="${IMAGE:-gemma4-captioner:v18-dev}"
mkdir -p in out
cp "${TASKS_FILE:-data/sample_tasks.json}" in/tasks.json

docker run --rm \
  -e OPENROUTER_API_KEY \
  -v "$(pwd)/in:/input:ro" \
  -v "$(pwd)/out:/output" \
  "$IMAGE"
