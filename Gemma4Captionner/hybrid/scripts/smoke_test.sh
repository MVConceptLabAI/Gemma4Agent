#!/usr/bin/env bash
# End-to-end sanity check: builds, runs, validates JSON contract, then tries the judge.
set -euo pipefail

echo "[0/4] Local contract test"
python scripts/contract_test.py

echo "[1/4] Build"
bash scripts/build.sh

echo "[2/4] Run"
bash scripts/run_local.sh

echo "[3/4] Validate results.json contract"
python - <<'PY'
import json
import pathlib

p = pathlib.Path("out/results.json")
data = json.loads(p.read_text(encoding="utf-8"))
assert isinstance(data, list) and data, "results.json must be a non-empty list"

required_styles = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
tech_keywords = {
    "api", "backend", "bug", "code", "coding", "commit", "compile", "cpu",
    "database", "deploy", "developer", "docker", "frontend", "git", "gpu",
    "kubernetes", "llm", "model", "programming", "python", "server", "software",
}

for row in data:
    assert "task_id" in row and "captions" in row, row
    missing = required_styles - set(row["captions"])
    assert not missing, f"Missing styles for {row['task_id']}: {missing}"
    for style, cap in row["captions"].items():
        assert isinstance(cap, str), f"{row['task_id']}.{style} is not a string"
        assert cap.strip(), f"{row['task_id']}.{style} is empty"
        if style == "humorous_non_tech":
            words = {
                w.strip(".,!?;:()[]{}\"'`").lower()
                for w in cap.replace("/", " ").replace("-", " ").split()
            }
            leaked = words & tech_keywords
            assert not leaked, (
                f"{row['task_id']}.{style} leaked tech jargon: {sorted(leaked)}"
            )

print(f"OK - {len(data)} clip(s), 4 styles each, non-empty captions")
PY

python eval/self_check.py --results out/results.json

echo "[4/4] Optional: local judge (needs FIREWORKS_API_KEY)"
python eval/local_judge.py \
    --results out/results.json \
    --clips eval/clips.json || true

echo "SMOKE TEST DONE."
