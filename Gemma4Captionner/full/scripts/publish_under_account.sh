#!/usr/bin/env bash
# Publish the current V18 tag through the repository-level GitHub workflow.
set -euo pipefail

EXPECTED="${1:-MVConceptLabAI}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

WHO="$(gh api user --jq .login 2>/dev/null || true)"
if [[ -z "$WHO" ]]; then
  echo "ERROR: authenticate GitHub CLI first with: gh auth login" >&2
  exit 1
fi
if ! gh repo view "$EXPECTED/Gemma4Agent" >/dev/null 2>&1; then
  echo "ERROR: $WHO cannot access $EXPECTED/Gemma4Agent" >&2
  exit 1
fi

if [[ -f Gemma4Captionner/full/.env ]]; then
  KEY="$(grep -E '^OPENROUTER_API_KEY=' Gemma4Captionner/full/.env | head -1 | cut -d= -f2- | tr -d '"')"
  [[ -n "$KEY" ]] && printf '%s' "$KEY" | gh secret set OPENROUTER_API_KEY \
    --repo "$EXPECTED/Gemma4Agent" --env OPENROUTER_API_KEY
fi

TAG=gemma4-submission-v18
git tag -f "$TAG"
git push origin "refs/tags/$TAG" --force

cat <<EOF
Published tag $TAG. The repository workflow builds:
  ghcr.io/mvconceptlabai/gemma4-captioner:$TAG

Watch it with:
  gh run watch --repo $EXPECTED/Gemma4Agent

The key is embedded only in the temporary judge image because the hackathon
injects no runtime secrets. Rotate it after judging.
EOF
