#!/usr/bin/env bash
# Pull a public image and run the degraded contract check through mounted I/O.
set -euo pipefail

if [[ -z "${PUBLIC_IMAGE:-}" ]]; then
    echo "PUBLIC_IMAGE is required, e.g. ghcr.io/<user>/track2-captioner:final" >&2
    exit 2
fi

WORK="${WORK:-public_verify}"
rm -rf "${WORK}"
mkdir -p "${WORK}/in" "${WORK}/out"
cp data/sample_tasks.json "${WORK}/in/tasks.json"

echo ">>> Pulling ${PUBLIC_IMAGE}"
docker pull "${PUBLIC_IMAGE}"

echo ">>> Inspecting architecture"
docker inspect "${PUBLIC_IMAGE}" --format 'architecture={{.Architecture}} os={{.Os}} size={{.Size}}'

echo ">>> Running degraded contract check"
docker run --rm \
    -v "$(pwd)/${WORK}/in:/input:ro" \
    -v "$(pwd)/${WORK}/out:/output" \
    -e PER_TASK_TIMEOUT_S=1 \
    -e FIREWORKS_API_KEY= \
    -e GROQ_API_KEY= \
    "${PUBLIC_IMAGE}"

python eval/self_check.py --results "${WORK}/out/results.json"
echo "PUBLIC IMAGE VERIFY OK"
