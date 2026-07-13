#!/usr/bin/env bash
# Build the Track 2 image for linux/amd64.
# The judging VM is linux/amd64 — any other arch will fail to pull and score 0.
set -euo pipefail

IMAGE="${IMAGE:-track2-captioner:dev}"

echo ">>> Building ${IMAGE} for linux/amd64"
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    --tag "${IMAGE}" \
    --load \
    .

echo ">>> Done. Compressed size:"
docker image ls "${IMAGE}" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}'
