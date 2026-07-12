#!/usr/bin/env bash
# One-shot: publish a PUBLIC, jury-pullable image under an account YOU control.
#
# Prereq: you are authenticated as the target account:
#   Option A (GHCR):  gh auth login   (as e.g. TheSkyGold, HTTPS, web browser)
#
# Usage:
#   bash scripts/publish_under_account.sh <github-username>
#   e.g. bash scripts/publish_under_account.sh TheSkyGold
#
# What it does (all under YOUR account, nothing under devopsm3):
#   1. creates a PUBLIC repo <user>/track2-captioner (code, so no keys in it)
#   2. sets OPENROUTER/GROQ/FIREWORKS as CI secrets (read from local .env)
#   3. pushes this code + the publish workflow
#   4. runs the workflow -> builds linux/amd64 and pushes
#        ghcr.io/<user>/track2-captioner:latest  (keys baked from CI secrets)
#   5. prints the exact "make package public" URL + the docker pull line
#
# After it finishes you do ONE click (make the package public), then submit.
set -euo pipefail

USER_ACCT="${1:?usage: publish_under_account.sh <github-username>}"
REPO="track2-captioner"
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

# --- sanity: authed as the target account? ---
who="$(gh api user --jq .login 2>/dev/null || true)"
if [[ "$who" != "$USER_ACCT" ]]; then
  echo "ERROR: gh is authed as '${who:-none}', not '$USER_ACCT'. Run: gh auth login" >&2
  exit 1
fi

# --- load key VALUES from .env (never echoed) ---
get_env() { grep -E "^$1=" .env | head -1 | cut -d= -f2- | tr -d '"'\' ; }
OR="$(get_env OPENROUTER_API_KEY)"; GK="$(get_env GROQ_API_KEY)"; FW="$(get_env FIREWORKS_API_KEY)"

# --- 1. create the public repo (idempotent) ---
if ! gh repo view "$USER_ACCT/$REPO" >/dev/null 2>&1; then
  gh repo create "$USER_ACCT/$REPO" --public \
    --description "AMD Track 2 video-captioning agent (ensemble + hardened Groq floor)"
fi

# --- 2. CI secrets (so the image bakes working keys; repo code stays keyless) ---
[[ -n "$OR" ]] && printf '%s' "$OR" | gh secret set OPENROUTER_API_KEY --repo "$USER_ACCT/$REPO"
[[ -n "$GK" ]] && printf '%s' "$GK" | gh secret set GROQ_API_KEY      --repo "$USER_ACCT/$REPO"
[[ -n "$FW" ]] && printf '%s' "$FW" | gh secret set FIREWORKS_API_KEY --repo "$USER_ACCT/$REPO"

# --- 3. push code (new remote, keeps origin=devopsm3 intact) ---
git remote remove pub 2>/dev/null || true
git remote add pub "https://github.com/$USER_ACCT/$REPO.git"
git push pub HEAD:main --force

# --- 4. run the publish workflow ---
gh workflow run publish.yml --repo "$USER_ACCT/$REPO" --ref main
echo "Publish workflow dispatched. Watch: gh run watch --repo $USER_ACCT/$REPO"

cat <<EOF

============================================================
NEXT (your 1 click), once the workflow shows 'completed success':
  Make the package public:
    https://github.com/users/$USER_ACCT/packages/container/$REPO/settings
    -> Danger Zone -> Change visibility -> Public -> confirm '$REPO'

  Jury pull line (for lablab):
    docker pull ghcr.io/${USER_ACCT,,}/track2-captioner:latest
    docker run --rm -v \$PWD/in:/input -v \$PWD/out:/output ghcr.io/${USER_ACCT,,}/track2-captioner:latest

  Then verify anonymously:
    python scripts/../verify (I'll run: python /tmp/verify_public.py after editing the image name)

  SECURITY: rotate the Groq key after judging (it's baked in the public image).
============================================================
EOF
