#!/usr/bin/env bash
# Start the GPU-only API that is reached through the Cloudflare Worker.
# Secrets must be exported by the operator, never saved in this repository.
set -Eeuo pipefail

: "${DEMO_ORIGIN_TOKEN:?Set a random shared token before starting the service.}"
: "${OPENROUTER_API_KEY:?Set the OpenRouter key in the GPU environment.}"

PORT="${DEMO_PORT:-8799}"
LOG_FILE="${DEMO_LOG_FILE:-/workspace/gemma4-demo-api.log}"

python -c 'import flask' 2>/dev/null || python -m pip install -r requirements.txt
nohup env DEMO_PORT="$PORT" python -u -m app.demo_api >"$LOG_FILE" 2>&1 &
echo "Gemma 4 demo API started on port $PORT (PID $!)."
echo "Private log: $LOG_FILE"
