# Track 2 - orchestrated commands. Every target is idempotent and safe to
# re-run.
#
# Env you probably want set:
#   FIREWORKS_API_KEY   (required for real inference + teacher-mode dataset)
#   IMAGE               (defaults to track2-captioner:dev)

IMAGE ?= track2-captioner:dev
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} \
	      /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
.PHONY: scenes
scenes:  ## Generate 200 balanced scenes -> finetune/scenes.jsonl
	$(PYTHON) finetune/generate_scenes.py --n 200 --out finetune/scenes.jsonl

.PHONY: dataset
dataset: scenes  ## Build dataset_v2.jsonl (offline mode, ~800 rows)
	$(PYTHON) finetune/build_dataset_v2.py \
	  --scenes finetune/scenes.jsonl \
	  --out finetune/dataset_v2.jsonl \
	  --mode offline

.PHONY: dataset-teacher
dataset-teacher: scenes  ## Build dataset via a teacher LLM (needs FIREWORKS_API_KEY)
	$(PYTHON) finetune/build_dataset_v2.py \
	  --scenes finetune/scenes.jsonl \
	  --out finetune/dataset_v2.jsonl \
	  --mode teacher \
	  --model accounts/fireworks/models/gemma-3-27b-it

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
.PHONY: train-dryrun
train-dryrun:  ## Validate the dataset by printing one training example
	$(PYTHON) finetune/train_gemma_lora.py \
	  --dataset finetune/dataset_v2.jsonl --dry-run

.PHONY: train
train:  ## Real LoRA training (needs Unsloth + a GPU)
	$(PYTHON) finetune/train_gemma_lora.py \
	  --dataset finetune/dataset_v2.jsonl \
	  --model unsloth/gemma-3-4b-it \
	  --output finetune/out --epochs 3

# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
.PHONY: build
build:  ## docker buildx for linux/amd64
	bash scripts/build.sh

.PHONY: run
run:  ## Run the container against data/sample_tasks.json
	bash scripts/run_local.sh

.PHONY: mock
mock:  ## Full end-to-end pipeline WITHOUT any API call
	mkdir -p out
	$(PYTHON) scripts/mock_run.py --tasks data/sample_tasks.json --out out/mock_results.json

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
.PHONY: self-check
self-check:  ## Structural + style-ban check on out/results.json
	$(PYTHON) eval/self_check.py --results out/results.json

.PHONY: self-check-mock
self-check-mock:  ## Structural + style-ban check on out/mock_results.json
	$(PYTHON) eval/self_check.py --results out/mock_results.json

.PHONY: judge
judge:  ## LLM-Judge proxy on out/results.json (costs a few Fireworks tokens)
	$(PYTHON) eval/local_judge.py \
	  --results out/results.json --clips eval/clips.json

# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
.PHONY: smoke
smoke: mock self-check-mock  ## Offline sanity: mock pipeline + self-check
	@echo "Smoke (offline) OK."

.PHONY: preflight
preflight:  ## Offline pre-submit evidence pack
	$(PYTHON) scripts/preflight.py

.PHONY: preflight-strict
preflight-strict:  ## Strict pre-submit gate (requires Docker + FIREWORKS_API_KEY)
	$(PYTHON) scripts/preflight.py --strict --docker-build --docker-run

.PHONY: submit-check
submit-check: preflight-strict run self-check  ## Final pre-submit gate (real inference)
	@echo "Submit-check OK - image is ready to push."

.PHONY: publish
publish:  ## Build and push PUBLIC_IMAGE for linux/amd64
	bash scripts/publish_image.sh

.PHONY: verify-public
verify-public:  ## Pull and contract-test PUBLIC_IMAGE anonymously
	bash scripts/verify_public_image.sh

.PHONY: clean
clean:  ## Remove out/, in/, finetune/out/
	rm -rf out in out_docker public_verify finetune/out
