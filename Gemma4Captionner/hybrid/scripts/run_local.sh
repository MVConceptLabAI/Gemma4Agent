#!/usr/bin/env bash
# Run the container locally on data/sample_tasks.json.
set -euo pipefail

IMAGE="${IMAGE:-track2-captioner:dev}"

if [[ -z "${FIREWORKS_API_KEY:-}" ]]; then
    echo "FIREWORKS_API_KEY is not set. Export it before running." >&2
    exit 1
fi

mkdir -p in out
cp data/sample_tasks.json in/tasks.json

echo ">>> Running ${IMAGE}"
time docker run --rm \
    -v "$(pwd)/in:/input:ro" \
    -v "$(pwd)/out:/output" \
    -e FIREWORKS_API_KEY \
    -e FIREWORKS_BASE_URL \
    -e VLM_MODEL \
    -e STYLE_MODEL \
    "${IMAGE}"

echo ">>> results.json:"
if command -v jq >/dev/null; then
    jq . out/results.json
else
    cat out/results.json
fi
