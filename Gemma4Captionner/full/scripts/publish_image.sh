#!/usr/bin/env bash
# Build and push the public linux/amd64 image for submission.
set -euo pipefail

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required to publish a judge-ready image}"

if [[ -z "${PUBLIC_IMAGE:-}" ]]; then
    echo "PUBLIC_IMAGE is required, e.g. ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v19" >&2
    exit 2
fi

echo ">>> Building and pushing ${PUBLIC_IMAGE} for linux/amd64"
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    --build-arg "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
    --tag "${PUBLIC_IMAGE}" \
    --push \
    .

echo ">>> Pushed ${PUBLIC_IMAGE}"
echo ">>> Verify anonymous pull from a clean environment:"
echo "    PUBLIC_IMAGE=${PUBLIC_IMAGE} bash scripts/verify_public_image.sh"
