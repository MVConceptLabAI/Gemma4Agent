# Active V20 commands. V18 and V19 remain in Git history.
IMAGE ?= gemma4-captioner:v20-dev
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help build run self-check repetition-audit contract smoke publish verify-public clean

help:  ## Show commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build the linux/amd64 V20 image
	bash scripts/build.sh

run:  ## Run V20 against data/sample_tasks.json (requires OPENROUTER_API_KEY)
	bash scripts/run_local.sh

self-check:  ## Validate out/results.json contract and style rules
	$(PYTHON) eval/self_check.py --results out/results.json

repetition-audit:  ## Reject exact repeated captions and report repeated phrases
	$(PYTHON) scripts/repetition_audit.py out/results.json --fail-on-exact

contract:  ## Offline input/output contract tests
	$(PYTHON) scripts/contract_test.py

smoke: contract  ## Build, run and validate V20
	bash scripts/smoke_test.sh

publish:  ## Push PUBLIC_IMAGE as linux/amd64
	bash scripts/publish_image.sh

verify-public:  ## Pull and contract-test PUBLIC_IMAGE
	bash scripts/verify_public_image.sh

clean:  ## Remove generated local output
	rm -rf out in out_docker public_verify
