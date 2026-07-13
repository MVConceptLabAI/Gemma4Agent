# Track 2 v38 MI300X and Independent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan one task at a time. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select a v38 arm only when it beats the exact v36 baseline under a reproducible, position-balanced, multi-judge evaluation and a zero-Fireworks MI300X factual benchmark.

**Architecture:** Build a blind paired evaluator that balances order within clip and style, records both orientations without double-counting them, and aggregates to one delta per clip before an exact sign-flip test. Select profiles on the public plus stress development corpus, freeze all hashes, then open the 12-clip confirmation corpus once. Every scoring judge is filtered against models actually used by both arms, including fallbacks; full-video or dense independent evidence adjudicates claims absent from sparse frames.

**Tech Stack:** Python 3.11 for the Track 2 app/evaluator, Python 3.12 inside a versioned vllm/vllm-openai-rocm container for MI300X, asyncio, httpx, FFmpeg, OpenAI-compatible local and OpenRouter endpoints, pinned Hugging Face revisions, Jupyter, hash-chained JSON Lines, exact randomization inference.

---

## Auditor independence rule

Independence is evaluated per clip/pair from actual_models_used, including every fallback:

    independent(judge, pair) = (
        model_family(judge)
        not in model_families(pair.baseline_actual_models | pair.candidate_actual_models)
    )

A same-family judge remains diagnostic but does not count toward the two independent votes. Build the scoring set per pair from this replacement pool: local Qwen3-VL, Gemini, GPT, Claude, and Mistral visual. Only qualified and family-independent entries count; model availability is checked before result unblinding. Deterministic checks are additional diagnostics. If fewer than two independent model votes remain for a pair, two blinded humans must review that full pair or the panel is incomplete.

If one judge fails or disagrees systematically, report it; do not silently average it away.

Before opening candidate results, qualify each judge on at least 20 synthetic or human-authored benchmark pairs covering false color, OCR, action, chronology, style permutation, identical captions, and order inversion. A scoring judge requires at least 90 percent correct winners, at most 10 percent false severe-error flags, and at least 80 percent inversion consistency.

    FAMILY_PATTERNS = {
        "openai": ("openai/", "gpt-", "gpt_"),
        "google": ("google/", "gemini", "gemma"),
        "anthropic": ("anthropic/", "claude"),
        "qwen": ("qwen/", "qwen"),
        "moonshot": ("moonshotai/", "kimi"),
        "minimax": ("minimax/", "minimax"),
        "deepseek": ("deepseek/", "deepseek"),
        "mistral": ("mistralai/", "mistral", "pixtral"),
        "xai": ("x-ai/", "grok"),
        "meta": ("meta-llama/", "llama"),
    }

    def model_family(model_id: str) -> str:
        normalized = model_id.casefold().replace("accounts/fireworks/models/", "")
        matches = [
            family for family, patterns in FAMILY_PATTERNS.items()
            if any(pattern in normalized for pattern in patterns)
        ]
        if len(matches) != 1:
            raise ValueError(f"unresolved model family for {model_id}")
        return matches[0]

    JUDGE_POOL = [
        "Qwen/Qwen3-VL-4B-Instruct",
        "google/gemini-3.1-pro-preview",
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.5",
        "mistralai/mistral-medium-3.1",
    ]

## Corpus separation

- data/public3.json contains only 13825391, 1860079, and 3044693 and is smoke-only.
- data/stress_tasks.json and data/official_new12.json are development-only because this repository and prior runs may already have exposed them.
- data/confirmation_fresh12.json must contain 12 new direct-download clips absent from the repository history, selected before inference by category constraints: nature, urban, animals, people, sports, food, weather, technology, multi-shot, text-heavy, vertical, and low-light. Each must be 30-120 seconds, legally usable for evaluation, checksum-pinned, and human-annotated before model output.
- data/official_tasks.json contains 15 clips and must never be mislabeled public3.

### Task 1: Implement deterministic blind A/B assignment

**Files:**
- Create: eval/paired_panel_judge.py
- Create: scripts/test_paired_panel.py

- [ ] Add failing tests for stable, balanced assignment keyed by run seed, task_id, style, and judge model.

    from eval.paired_panel_judge import blind_assignment

    def test_blind_assignment_is_stable_and_reversible() -> None:
        first = blind_assignment("v38-gate-1", "v1", "formal", "judge-a")
        second = blind_assignment("v38-gate-1", "v1", "formal", "judge-a")
        assert first == second
        assert set(first) == {"baseline", "candidate"}

- [ ] Add corpus-level tests requiring exactly 2/2 positions inside every clip and a baseline-first/candidate-first difference at most one for every judge by style.

- [ ] Run the test and confirm the module is missing.

    python scripts/test_paired_panel.py

    Expected: FAIL with ModuleNotFoundError.

- [ ] Implement assignment from SHA-256, not process-random hash, and persist a concealed mapping in the run cache.

    def blind_assignment(
        seed: str,
        task_id: str,
        style: str,
        judge_model: str,
    ) -> tuple[str, str]:
        material = "|".join((seed, task_id, style, judge_model)).encode("utf-8")
        bit = hashlib.sha256(material).digest()[0] & 1
        return ("baseline", "candidate") if bit == 0 else ("candidate", "baseline")

    def build_blind_assignments(
        seed: str,
        judge_model: str,
        pairs: list[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[str, str]]:
        style_order = [
            "formal", "sarcastic", "humorous_tech", "humorous_non_tech"
        ]
        clips = sorted(
            {task_id for task_id, _ in pairs},
            key=lambda task_id: hashlib.sha256(
                f"{seed}|{judge_model}|{task_id}".encode("utf-8")
            ).digest(),
        )
        assignments: dict[tuple[str, str], tuple[str, str]] = {}
        for clip_index, task_id in enumerate(clips):
            pattern = (0, 1, 0, 1) if clip_index % 2 == 0 else (1, 0, 1, 0)
            for style_index, style in enumerate(style_order):
                if (task_id, style) not in pairs:
                    continue
                assignments[(task_id, style)] = (
                    ("baseline", "candidate")
                    if pattern[style_index] == 0
                    else ("candidate", "baseline")
                )
        return assignments

- [ ] Use build_blind_assignments for complete runs so position counts are balanced. blind_assignment is only the deterministic single-pair fallback.

- [ ] The judge-facing prompt may use only Caption A and Caption B. It must not reveal version names, model names, cost, or leaderboard score.

- [ ] Run tests.

    python scripts/test_paired_panel.py

    Expected: PASS.

- [ ] Commit.

    git add eval/paired_panel_judge.py scripts/test_paired_panel.py
    git commit -m "test: add deterministic blind caption assignment"

### Task 2: Define the official-like two-axis judge schema

**Files:**
- Modify: eval/paired_panel_judge.py
- Modify: scripts/test_paired_panel.py

- [ ] Add failing parser tests for valid JSON, fenced JSON, missing scores, scores outside 0-1, ties, contradictory winner and scores, and judge failure.

- [ ] Define one strict response schema:

    {
      "a": {
        "accuracy": 0.0,
        "style": 0.0,
        "severe_factual_error": false,
        "flaw": "",
        "claim_issues": []
      },
      "b": {
        "accuracy": 0.0,
        "style": 0.0,
        "severe_factual_error": false,
        "flaw": "",
        "claim_issues": []
      },
      "winner": "tie",
      "confidence": 0.0,
      "reason": ""
    }

- [ ] Use the challenge definitions literally. Accuracy rewards correct specific coverage and penalizes unsupported subjects, actions, objects, places, counts, colors, OCR, and chronology. Style is judged independently against exactly one requested style.

- [ ] Each claim issue has claim, classification, timestamp_start, timestamp_end, and evidence_frame_ids. classification is contradicted, unsupported_after_full_review, or not_visible_in_review_evidence. Do not deduct accuracy for not_visible_in_review_evidence; route it to dense-video or human adjudication.

- [ ] Do not hard-code arbitrary deductions such as 0.15 per flaw. Ask each judge to calibrate 0-1 directly and retain raw reasoning under 30 words. winner accepts A, B, or tie and is treated as a diagnostic derived from the same call, never independent evidence from the axis scores.

- [ ] Mark a response invalid if the declared winner contradicts both caption axis means by more than 0.05. Retry once with a format-only repair prompt; otherwise record judge_failed.

    def parse_one_json_object(text: str) -> dict[str, object]:
        decoder = json.JSONDecoder()
        values: list[dict[str, object]] = []
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                values.append(value)
        if len(values) != 1:
            raise ValueError(f"expected one JSON object, found {len(values)}")
        return values[0]

    def parse_verdict(text: str) -> dict[str, object]:
        payload = parse_one_json_object(text)
        if payload.get("winner") not in {"A", "B", "tie"}:
            raise ValueError("winner must be A, B, or tie")
        means: dict[str, float] = {}
        for key in ("a", "b"):
            item = payload.get(key)
            if not isinstance(item, dict):
                raise ValueError(f"missing {key} score object")
            accuracy = float(item["accuracy"])
            style = float(item["style"])
            if not 0.0 <= accuracy <= 1.0 or not 0.0 <= style <= 1.0:
                raise ValueError("judge scores must be within 0-1")
            means[key.upper()] = (accuracy + style) / 2
        score_winner = (
            "tie" if abs(means["A"] - means["B"]) <= 0.05
            else "A" if means["A"] > means["B"] else "B"
        )
        if payload["winner"] != score_winner:
            raise ValueError("declared winner contradicts axis scores")
        confidence = float(payload["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within 0-1")
        return payload

- [ ] Run tests.

    python scripts/test_paired_panel.py

    Expected: PASS for every parser case.

- [ ] Commit.

    git add eval/paired_panel_judge.py scripts/test_paired_panel.py
    git commit -m "feat: score blind pairs on official caption axes"

### Task 3: Add independent judge adapters and immutable caching

**Files:**
- Modify: eval/paired_panel_judge.py
- Modify: scripts/test_paired_panel.py
- Modify: .gitignore

- [ ] Add fake-server tests for:
  - local OpenAI-compatible Qwen visual endpoint;
  - OpenRouter Gemini endpoint;
  - OpenRouter GPT endpoint;
  - timeout, 429, invalid JSON, and one retry;
  - no authorization value in cache or console output.

- [ ] Configure judges explicitly:

    LOCAL_VISUAL_JUDGE_MODEL=Qwen/Qwen3-VL-4B-Instruct
    LOCAL_VISUAL_JUDGE_BASE_URL=http://127.0.0.1:8000/v1
    REMOTE_VISUAL_JUDGES=google/gemini-3.1-pro-preview,openai/gpt-5.5,anthropic/claude-opus-4.5,mistralai/mistral-medium-3.1

- [ ] For remote judges, extract 24 timestamped review frames using positions shifted from every generator profile, maximum edge 640. For the local judge, provide the full video at one frame per second capped at 128. Sparse-frame judges may emit not_visible_in_review_evidence but cannot call that claim false.

- [ ] Cache a verdict by video digest, caption hashes, style, resolved judge model, prompt revision, assignment seed, presentation_id, ordered_A_hash, ordered_B_hash, frame_manifest_hash, and preprocessor version. Store append-only, file-locked, hash-chained JSON Lines under out/judge-cache, which remains gitignored.

- [ ] Never read .env by iterating and printing lines. Use python-dotenv and access only the needed variable names.

- [ ] Load actual_models_used from both result manifests, normalize with model_family, and mark each verdict scoring or diagnostic before inference. Add qualification fixtures and refuse to count an unavailable, unqualified, unresolved, or same-family judge. For every clip/style with fewer than two valid model votes, require two blinded human votes or mark panel_incomplete.

- [ ] Run tests with local fake servers only.

    python scripts/test_paired_panel.py

    Expected: PASS and no external calls.

- [ ] Commit.

    git add eval/paired_panel_judge.py scripts/test_paired_panel.py .gitignore
    git commit -m "feat: add cached independent visual judge adapters"

### Task 4: Aggregate at clip level with uncertainty and judge diagnostics

**Files:**
- Modify: eval/paired_panel_judge.py
- Modify: scripts/test_paired_panel.py

- [ ] Add failing tests for:
  - exact mean accuracy, style, and final score per arm;
  - paired delta per clip/style;
  - judge-specific delta and win rate;
  - ties counting as half a win;
  - severe-error rate;
  - orientation collapse before repetition, judge, and style aggregation;
  - exactly one delta per clip;
  - exact sign-flip p-value over all 2^n clip sign assignments;
  - deterministic clip-cluster bootstrap as a secondary interval;
  - a known positive data set with a confidence interval above zero;
  - a mixed data set with a confidence interval crossing zero.

- [ ] Implement:

    @dataclass(frozen=True)
    class GateSummary:
        baseline_accuracy: float
        candidate_accuracy: float
        baseline_style: float
        candidate_style: float
        paired_proxy_delta: float
        ci95_low: float
        ci95_high: float
        exact_sign_flip_p: float
        candidate_win_rate: float
        severe_error_rate: float
        judge_deltas: dict[str, float]

    def exact_sign_flip_pvalue(clip_deltas: list[float]) -> float:
        observed = abs(statistics.mean(clip_deltas))
        extreme = 0
        total = 1 << len(clip_deltas)
        for mask in range(total):
            permuted = [
                delta if mask & (1 << index) else -delta
                for index, delta in enumerate(clip_deltas)
            ]
            if abs(statistics.mean(permuted)) >= observed - 1e-12:
                extreme += 1
        return extreme / total

    def bootstrap_clip_delta(
        clip_deltas: list[float],
        seed: int = 38,
        samples: int = 10_000,
    ) -> tuple[float, float]:
        rng = random.Random(seed)
        draws = sorted(
            statistics.mean(rng.choice(clip_deltas) for _ in clip_deltas)
            for _ in range(samples)
        )
        return draws[int(samples * 0.025)], draws[int(samples * 0.975) - 1]

- [ ] Implement this aggregation order exactly: remap A/B; average two orientations by clip/style/judge/repetition; average both repetitions; average valid independent judges with equal weights; average four styles with equal weights; retain separate accuracy and style deltas; produce one proxy delta per clip only after labeling the 50/50 mean as non-official.

- [ ] Report pairwise winner agreement and Gwet AC1 for a two-arm final. Use Kendall or Spearman only when every judge evaluated the same three or more arms.

- [ ] Flag any judge whose mean delta differs in sign from every other valid judge; never delete its records automatically. Add leave-one-judge-out summaries and make a clip/style incomplete when fewer than two independent valid judges remain unless a blinded human adjudication replaces the missing vote.

- [ ] Run tests.

    python scripts/test_paired_panel.py

    Expected: PASS with stable numeric fixtures.

- [ ] Commit.

    git add eval/paired_panel_judge.py scripts/test_paired_panel.py
    git commit -m "feat: bootstrap paired caption evaluation by clip"

### Task 5: Implement the machine-enforced submission gate

**Files:**
- Create: scripts/ab_gate.py
- Create: scripts/test_ab_gate.py

- [ ] Add failing tests for each independent failure:
  - malformed result JSON;
  - any missing requested caption;
  - batch runtime above 535 seconds;
  - nearest-rank p95 over 24 warm task measurements above 90 seconds or maximum task above 125 seconds;
  - candidate accuracy or style below baseline in aggregate;
  - any style/axis failing the declared non-inferiority margin;
  - paired proxy delta below the empirically measured MDE;
  - exact sign-flip p-value above 0.05;
  - human-confirmed candidate-only severe factual error;
  - fewer than two agreeing independent auditors;
  - Docker contract or manifest failure.
  - baseline_kind other than R0_exact, wrong R0 index/amd64 digest, or missing baseline-noise hash.

- [ ] Define the exact gate:

    def nearest_rank(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        rank = max(1, math.ceil(percentile * len(ordered)))
        return ordered[rank - 1]

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    EXPECTED_R0_INDEX = "sha256:161efc8b098a6a46f395f01fb83ce7c41a9e71c61a51205b59934393bac5f19d"
    EXPECTED_R0_AMD64 = "sha256:1b39f65c4b99642a318b353a2e0281c4d5ddb4d346510766e40c27ae5ed0ac07"
    baseline_test_retest_deltas = baseline_noise["proxy_clip_deltas"]
    mde = max(
        0.01,
        nearest_rank(
            [abs(delta) for delta in baseline_test_retest_deltas],
            0.95,
        ),
    )
    accuracy_noninferior_all_styles = all(
        accuracy_delta_by_style[style]
        >= -float(baseline_noise["ni_margin"]["accuracy"][style])
        for style in REQUIRED_STYLES
    )
    style_noninferior_all_styles = all(
        style_delta_by_style[style]
        >= -float(baseline_noise["ni_margin"]["style"][style])
        for style in REQUIRED_STYLES
    )
    eligible = (
        baseline_manifest["baseline_kind"] == "R0_exact"
        and baseline_manifest["index_digest"] == EXPECTED_R0_INDEX
        and baseline_manifest["amd64_digest"] == EXPECTED_R0_AMD64
        and baseline_manifest["noise_artifact_sha256"] == sha256_file(baseline_noise_path)
        and captions_present == captions_requested
        and batch_runtime_s <= 535
        and task_p95_s <= 90
        and max_task_s < 125
        and candidate_accuracy >= baseline_accuracy
        and candidate_style >= baseline_style
        and accuracy_noninferior_all_styles
        and style_noninferior_all_styles
        and paired_proxy_delta >= mde
        and exact_sign_flip_p <= 0.05
        and human_confirmed_candidate_only_severe_errors == 0
        and independent_agreeing_auditors >= 2
        and all_scoring_judges_qualified
        and docker_contract_passed
        and linux_amd64_manifest
        and compressed_size_bytes < 10_000_000_000
    )

    diagnostics = {
        "bootstrap_ci95": [ci95_low, ci95_high],
        "candidate_win_rate": candidate_win_rate,
        "leave_one_judge_out": leave_one_judge_out,
    }

- [ ] The arithmetic mean of accuracy and style is only a 50/50 proxy because the organizer has not published the axis weights. Gate both axes separately; do not describe the proxy as the official formula. Win rate and bootstrap interval remain diagnostics, not independent proof.

- [ ] Make the CLI consume only explicit JSON artifacts and emit one signed summary JSON plus a readable Markdown report. It must never infer missing evidence as pass.

    python scripts/ab_gate.py \
      --baseline out/ab/v36/results.json \
      --baseline-manifest out/ab/v36/manifest.json \
      --baseline-noise out/ab/v36/baseline-noise.json \
      --candidate out/ab/a1/results.json \
      --panel out/ab/a1/panel.json \
      --runtime out/ab/a1/runtime.json \
      --docker out/ab/a1/docker.json \
      --human out/ab/a1/human-review.json \
      --json-out out/ab/a1/gate.json \
      --report-out out/ab/a1/gate.md

- [ ] If R0 cannot execute, emit baseline_kind=C0_reconstructed and exit with exploratory_only rather than success. A C0 comparison may guide engineering but cannot satisfy the exact-R0 gate.

- [ ] Run tests.

    python scripts/test_ab_gate.py

    Expected: PASS for one all-green fixture and every single-failure fixture.

- [ ] Commit.

    git add scripts/ab_gate.py scripts/test_ab_gate.py
    git commit -m "feat: enforce v38 submission evidence gate"

### Task 6: Create the zero-credit MI300X model bake-off harness

**Files:**
- Create: notebooks/mi300x_model_bakeoff.py
- Modify: notebooks/qwen3vl_video_mi300x.ipynb
- Create: scripts/test_mi300x_bakeoff.py
- Create: eval/gold_video_ledger.schema.json

- [ ] Add failing unit tests for pinned model manifests, valid factual-ledger JSON, frame-budget configuration, runtime/HBM parsing, resume behavior, and refusal to run an unpinned revision.

- [ ] Define gold annotations before any model output is revealed:

    {
      "task_id": "stress_city_banner",
      "claims": [{
        "claim_id": "stress_city_banner-c001",
        "type": "action",
        "text": "normalized visible claim",
        "start_s": 0.0,
        "end_s": 3.0,
        "evidence": ["frame:1.0", "frame:2.0"],
        "accepted_aliases": ["equivalent normalized wording"],
        "uncertain": false,
        "annotator_ids": ["h1", "h2"],
        "adjudicated": true
      }]
    }

- [ ] Require double human annotation and adjudication for all development clips used to report recall/false-detail rates. Model-produced ledgers are predictions, never truth.

- [ ] Pin the manifest exactly:

    MODELS = {
        "qwen3vl4b": {
            "repo": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        },
        "minicpmv46": {
            "repo": "openbmb/MiniCPM-V-4.6",
            "revision": "8169864629825dc1d755a5aa1cd8b5935dcbc83f",
        },
        "internvideo25": {
            "repo": "OpenGVLab/InternVideo2_5_Chat_8B",
            "revision": "87680fe232af8681b55322f68541679b91f99df1",
        },
    }

- [ ] Make the harness produce the same factual ledger for every model:

    {
      "setting": [],
      "subjects": [],
      "actions": [],
      "chronology": [],
      "objects": [],
      "ocr": [],
      "colors_and_counts": [],
      "uncertain": []
    }

- [ ] Support three sampling modes: eight uniform endpoint-aware frames, sixteen uniform frames, and native one frame per second capped at 128 when the model supports it.

- [ ] Record repository, revision, runtime versions, ROCm version, GPU name, sampling mode, warm latency, peak HBM, valid-output flag, raw response hash, and ledger JSON. Never record Hub tokens.

- [ ] Use Python 3.12 inside vllm/vllm-openai-rocm:v0.24.0@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28, not the app's Python 3.11. Record vLLM, ROCm, PyTorch, transformers, video processor, qwen-vl-utils, and glibc versions before inference. Use a separate container or Python 3.12 environment per model.

- [ ] Update the notebook to launch one pinned model environment at a time, perform a real video request, call the harness, stop the server process, and verify HBM returns within 1 GiB of its pre-load level before the next model.

- [ ] Run unit tests locally without downloading model weights.

    python scripts/test_mi300x_bakeoff.py

    Expected: PASS using fixture responses.

- [ ] Commit.

    git add notebooks/mi300x_model_bakeoff.py notebooks/qwen3vl_video_mi300x.ipynb scripts/test_mi300x_bakeoff.py eval/gold_video_ledger.schema.json
    git commit -m "feat: add pinned MI300X video model bakeoff"

### Task 7: Run the MI300X factual benchmark in a controlled sequence

**Files:**
- Create locally: out/mi300x/qwen3vl4b.json
- Create locally: out/mi300x/minicpmv46.json
- Create locally only if needed: out/mi300x/internvideo25.json
- Create: docs/evals/2026-07-12-v38-mi300x-summary.md

- [ ] Confirm the AMD notebook GPU, ROCm, available HBM, disk space, and network before downloading weights.

    rocm-smi --showproductname --showmeminfo vram
    python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
    df -h

- [ ] Run Qwen3-VL-4B first on data/public3.json, data/stress_tasks.json, and data/official_new12.json with the three sampling modes. All are development-only; do not open data/confirmation_fresh12.json during model or prompt selection.

- [ ] Run MiniCPM-V-4.6 second on exactly the same media, order, prompt revision, and two warm repetitions.

- [ ] Score factual ledgers through deterministic claim matching plus blind human review. Do not ask either candidate model to grade itself.

- [ ] Run InternVideo2.5 only if Qwen and MiniCPM both miss temporal events on at least two reviewed clips or fail the visible-detail target.

- [ ] Record cold model-load time, first-request latency, warm median, nearest-rank p95, peak HBM, valid-output rate, gold-ledger recall, false-detail count, temporal accuracy, OCR accuracy, and any crash. A conditional InternVideo run must later complete the same benchmark before it can be called a winner.

- [ ] Write docs/evals/2026-07-12-v38-mi300x-summary.md with exact commands, revisions, results, and a factual winner. Do not promote a local model to production unless its endpoint is guaranteed reachable throughout official judging and the full batch runtime passes.

- [ ] Commit only the summary and hashes, not model weights, raw video, secrets, or large caches.

    git add docs/evals/2026-07-12-v38-mi300x-summary.md
    git commit -m "docs: record v38 MI300X factual benchmark"

### Task 8: Generate immutable R0, controlled C0, and candidate outputs on the same corpus

**Files:**
- Create: scripts/run_ab_matrix.py
- Create: scripts/test_run_ab_matrix.py
- Create: data/public3.json
- Create: data/confirmation_fresh12.json
- Create locally: out/ab/v36
- Create locally: out/ab/development
- Create locally: out/ab/confirmation

- [ ] Add fake-runner tests proving identical task order, two post-warm repetitions, separate output directories, no result overwrite, exact environment capture with secret values redacted, and per-call cost/latency aggregation.

- [ ] Implement the immutable and controlled references:

    R0 = ghcr.io/theskygold/track2-captioner@sha256:161efc8b098a6a46f395f01fb83ce7c41a9e71c61a51205b59934393bac5f19d
    R0_LINUX_AMD64 = sha256:1b39f65c4b99642a318b353a2e0281c4d5ddb4d346510766e40c27ae5ed0ac07
    C0 = source commit 283ce7f reconstructed profile, never labeled exact v36

- [ ] Implement the causal matrix. Against M0, F1 changes geometry only, D1 description only, L1 length only, and J1 joint call only. A_FULL combines the winning known effects; A_HARD adds repair only. Model changes are separate arms.

    ARMS = {
        "C0": {"parent": None, "changes": ["controlled_v36_source"]},
        "M0": {"parent": "C0", "changes": ["describex_models_only"]},
        "F1": {"parent": "M0", "changes": ["oci_exact_geometry_only"]},
        "D1": {"parent": "M0", "changes": ["rich_description_only"]},
        "L1": {"parent": "M0", "changes": ["long_caption_only"]},
        "J1": {"parent": "M0", "changes": ["joint_json_only"]},
        "A_FULL": {"parent": "M0", "changes": ["F1", "D1", "L1", "J1"]},
        "A_HARD": {"parent": "A_FULL", "changes": ["targeted_repair_only"]},
        "DESCRIBEX_OFFICIAL_REPO": {
            "parent": "M0",
            "changes": ["official_repo_16_frames_temp_0p3_gpt_oss_120b"],
        },
        "B0_UNIKL_EXACT": {"parent": None, "changes": ["faithful_unikl_architecture"]},
        "B1_UNIKL_HARDENED": {
            "parent": "B0_UNIKL_EXACT",
            "changes": ["validator_and_repair_only"],
        },
    }

- [ ] Create data/public3.json by selecting only IDs 13825391, 1860079, and 3044693 from data/official_tasks.json. Use public3 only for contract/smoke; use stress_tasks plus official_new12 for development.

- [ ] Before any candidate inference, create data/confirmation_fresh12.json from 12 new direct MP4 URLs meeting the category/duration/license rules above. Verify SHA-256, duration, resolution, reachability, duplicate perceptual hashes, and absence from every existing task file. Two humans create the gold ledgers before the file is sealed read-only.

- [ ] Run every simple ablation once on development. Interleave arms per clip with a seeded permutation to neutralize provider congestion. Run two post-warm repetitions only for C0, R0 when its embedded credentials still work, and the finalists. Cross execution order: repetition 1 R0 then candidate, repetition 2 candidate then R0.

- [ ] Before finalist confirmation, blind-judge R0 repetition 1 against R0 repetition 2 on the development clips. Aggregate to clip deltas with the same panel code, then freeze out/ab/v36/baseline-noise.json containing:
  - proxy_clip_deltas;
  - per-style accuracy/style test-retest deltas;
  - ni_margin for every style/axis as nearest-rank p95 absolute delta;
  - mde=max(0.01, nearest-rank p95 absolute proxy delta);
  - judge/prompt/frame hashes and the artifact SHA-256.

- [ ] Write out/ab/v36/manifest.json with baseline_kind=R0_exact only after the immutable index and amd64 child both match. If R0 cannot run, write baseline_kind=C0_reconstructed; downstream gate must return exploratory_only.

- [ ] Capture results, runtime, fallback reasons, actual resolved models/fallbacks per clip, provider usage, estimated cost, source commit, image digest, prompt hash, and environment names with values redacted. Use isolated caches per arm.

- [ ] If R0 credentials no longer function, mark its fresh run unavailable and use baseline_reconstructed for C0; do not claim an exact live v36 comparison. The official 0.9133 score remains historical evidence only.

- [ ] Commit the runner and tests, not generated out artifacts.

    git add scripts/run_ab_matrix.py scripts/test_run_ab_matrix.py
    git commit -m "feat: run reproducible v36 versus v38 matrix"

### Task 9: Run finalist judging and blind human adjudication

**Files:**
- Create locally: out/ab/finalist/panel.json
- Create locally: out/ab/finalist/human-review.json
- Create: docs/evals/2026-07-12-v38-finalist-report.md

- [ ] Before opening confirmation results, freeze finalist profile, prompt hashes, source commit, container digest, model/fallback list, retry policy, frame geometry, and validation policy in out/ab/finalist/freeze.json. A changed hash starts a new development cycle.

- [ ] Run both finalist repetitions and the baseline on data/confirmation_fresh12.json exactly once. No prompt/model selection may use these results.

- [ ] First run a qualified independent local dense-video judge over every confirmation clip/style/repetition pair. Any claim not visible in sparse evidence goes to this full-video review, not an automatic error.

- [ ] Only for the frozen finalist, select at least two qualified family-independent judges per pair from JUDGE_POOL. Run them over the same blinded pairs with the 24-frame pack or dense local video, then run the exact complementary presentation. Cache both presentations with distinct presentation_id values. A disqualified Gemini/GPT/Claude is replaced rather than averaged.

- [ ] Remap and average the two orientations before repetition, judge, style, or clip aggregation. Never count the inversion as a second observation. Mark a remapped preference that changes with order order_sensitive and send it to human review.

- [ ] Blindly human-review:
  - every severe factual disagreement;
  - every pair where judges disagree on winner;
  - every order_sensitive pair;
  - a seeded, style-stratified 25 percent sample of all remaining clip/style pairs.

- [ ] Use two blind human reviewers on the stratified sample and all disputed/severe pairs, with the full video and sound available. Record visible-evidence errors and style compliance without arm identity; report raw agreement and Gwet AC1 or Cohen kappa, then adjudicate disagreements.

- [ ] Run the gate.

    python scripts/ab_gate.py \
      --baseline out/ab/v36/results.json \
      --baseline-manifest out/ab/v36/manifest.json \
      --baseline-noise out/ab/v36/baseline-noise.json \
      --candidate out/ab/finalist/results.json \
      --panel out/ab/finalist/panel.json \
      --runtime out/ab/finalist/runtime.json \
      --docker out/ab/finalist/docker.json \
      --human out/ab/finalist/human-review.json \
      --json-out out/ab/finalist/gate.json \
      --report-out out/ab/finalist/gate.md

    Expected for submission eligibility: gate exits 0, exact sign-flip p is at most 0.05, both axes pass, and every criterion is true.

- [ ] Write docs/evals/2026-07-12-v38-finalist-report.md with separate per-style accuracy/style deltas, the explicitly non-official 50/50 proxy, exact sign-flip result, MDE, secondary bootstrap interval, leave-one-judge-out summaries, judge qualification/agreement, order-sensitivity, human findings, runtime, cost, and Docker proof.

- [ ] Commit the report only after checking that it contains no secret values.

    git add docs/evals/2026-07-12-v38-finalist-report.md
    git commit -m "docs: record independent v38 finalist gate"

### Task 10: Publish and submit only after a green gate and resolved v37 score

**Files:**
- Modify: Dockerfile
- Modify: docs/evals/2026-07-12-v38-finalist-report.md

- [ ] Confirm the official v37 score is no longer pending and record whether v38 is based on the v36 floor, an isolated v37 improvement, or a frozen first-place behavior.

- [ ] Require a green gate from Task 9. If any criterion fails, stop and leave the current 0.9133 leaderboard result untouched; do not claim a cryptographically proven score-to-digest link and do not spend a submission.

- [ ] Set only the selected arm as the Docker default while leaving the exact legacy engine selectable by environment.

- [ ] Rebuild and publish linux/amd64 from the tested source commit using dedicated revocable contest credentials.

    docker buildx build \
      --platform linux/amd64 \
      --tag ghcr.io/theskygold/track2-captioner:submission-v38 \
      --push .

- [ ] Pull the public digest anonymously, run the exact three-task contract against that digest, verify size and linux/amd64, and compare the public digest to the recorded candidate.

- [ ] Submit once on Lablab, record the exact resubmission timestamp, image tag, index digest, source commit, gate-report hash, and cost.

- [ ] Start one heartbeat for the fresh score. Do not make another quality change until that score appears and its evidence is analyzed.

## Evaluation completion condition

Completion requires a reproducible green gate, resolved v37 score, anonymous public-image verification, and one recorded submission. A high single-judge score, a visually pleasing example, or a successful Docker build is not sufficient evidence.
