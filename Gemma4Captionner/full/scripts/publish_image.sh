#!/usr/bin/env bash
# Build and push the public linux/amd64 image for submission.
set -euo pipefail

if [[ -z "${PUBLIC_IMAGE:-}" ]]; then
    echo "PUBLIC_IMAGE is required, e.g. ghcr.io/<user>/track2-captioner:final" >&2
    exit 2
fi

echo ">>> Building and pushing ${PUBLIC_IMAGE} for linux/amd64"
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    --tag "${PUBLIC_IMAGE}" \
    --push \
    .

echo ">>> Pushed ${PUBLIC_IMAGE}"
echo ">>> Verify anonymous pull from a clean environment:"
echo "    PUBLIC_IMAGE=${PUBLIC_IMAGE} bash scripts/verify_public_image.sh"
