# Track 2 v38 First-Place Design

Date: 2026-07-12
Project: Verified Scene Gate / Mvconceptlab
Objective: exceed the current Track 2 leader score of 0.9217 without sacrificing the proven 0.9133 floor, the ten-minute batch limit, or output reliability.

## 1. Confirmed constraints

- The container reads `/input/tasks.json` at startup and writes valid `/output/results.json` before exiting.
- Every requested style must be present: `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech`.
- Hidden evaluation contains approximately 12 unseen clips, each 30 seconds to 2 minutes.
- Total runtime is at most 10 minutes.
- The compressed public image must remain below 10 GB and include a `linux/amd64` manifest.
- No model is injected for Track 2. The container must have a reliable inference route available during evaluation.
- Scoring is the mean of factual accuracy and requested-style match across every clip and style.
- Submission slots are rate-limited. A new image is submitted only after a complete A/B gate, never merely because it builds.

## 2. Evidence behind the redesign

The current leaders use simpler pipelines than v37:

| Candidate | Official score | Observed production pattern |
|---|---:|---|
| DescribeX / Ryzen Siblings | 0.9217 | Broad visual coverage, a 150-250 word factual description, then all four styles in one JSON response; captions are 2-4 sentences and roughly 40-120 words. |
| UniKLers | 0.9175 | MiniMax-M3 receives the native video, writes a detailed 5-7 sentence description, then produces all four styles together in one JSON response. |
| Mvconceptlab v36 | 0.9133 | Ten high-resolution frames, multiple observers, one shared writer. This is the proven rollback floor. |
| Mvconceptlab v37 | Pending | Six lower-resolution frames, a short fact list capped at five items, sequential draft/verify calls, and independent short style writers. |

The leading hypothesis is therefore not that more agents or more verification are needed. It is that v37 compressed the visual evidence and final captions too aggressively. The next candidate restores factual recall, broad temporal coverage, and joint style generation while keeping v37's reliable I/O and fallback protections.

## 3. Recommended candidate: v38 Leader-Parity Plus

### 3.1 Frame selection

- Treat public DescribeX evidence as two different sources, not one proven winning image. The officially linked repository extracts at fps=min(video_fps, 60/duration), caps near 60 frames, then uniformly subsamples 16 including first and last. The similarly branded OCI image, whose relationship to the scored submission remains unproven, selects eight frames at approximately 5%, 16.25%, 27.5%, 38.75%, 50%, 61.25%, 72.5%, and 83.75%.
- Test the OCI-exact geometry against an independent endpoint-aware eight-frame variant spanning 5%-95%; do not call the latter an exact DescribeX reproduction.
- Encode at a maximum edge of 768 pixels and JPEG quality 85.
- Always include near-start and near-end evidence. Do not discard the first or last 15% of a long clip.
- Do not use scene detection in the primary arm. Adaptive selection remains an A/B arm because uniform sampling is the only pattern directly associated with the current first-place score.

### 3.2 Canonical scene description

- Primary model: `accounts/fireworks/models/kimi-k2p6`.
- One multimodal call per clip with reasoning disabled, temperature 0.2-0.3, and a bounded output of about 600 tokens.
- Target 150-250 words of factual prose, covering:
  - setting and visible time/weather indicators;
  - main and secondary subjects;
  - concrete appearance details, colors, quantities, and distinguishing objects when clearly visible;
  - actions and chronological changes;
  - legible text or logos, otherwise explicitly uncertain;
  - camera movement and overall visual atmosphere.
- The prompt forbids speculation but does not cap the scene to five facts and does not ban all colors, counts, or types.
- There is no mandatory same-model verification pass. A failed description call goes to the configured fallback rather than treating an incomplete draft as verified truth.

### 3.3 Joint four-style generation

- Primary writer: `accounts/fireworks/models/gpt-oss-20b`.
- Fallback writer: a currently available low-latency text model such as DeepSeek V4 Flash, only after an actual API, empty-output, length, or malformed-JSON failure.
- One call generates every requested style together as strict JSON.
- Each caption is 2-4 complete sentences and normally 40-100 words, with an absolute cap of 120 words and 700 characters.
- All four outputs share the same factual description, but each uses a clearly different rhetorical structure.
- The writer may invent a comparison or punchline, but not a new subject, action, object, place, identity, or event.
- Style definitions remain literal to the challenge. `humorous_non_tech` rejects programming jargon; `humorous_tech` must contain a natural technology or programming analogy tied to a visible action.

### 3.4 Validation and repair

Validation is deliberately narrow:

- exact requested keys are present;
- every value is a non-empty English string;
- JSON, Unicode, and sentence termination are valid;
- word and character limits are respected;
- obvious corruption, duplicated fragments, leaked reasoning, or cross-task text is rejected;
- the non-tech caption contains no banned technical jargon;
- the tech caption contains a recognizable technical reference.

The validator does not attempt open-world factual entailment with a small noun list. If only one style fails, request one targeted rewrite from the same canonical description. Do not regenerate all four captions. If the complete joint response fails structurally, retry the joint call once, then fail over.

### 3.5 Reliability retained from v37

- Prewrite a valid results skeleton before inference.
- Atomically checkpoint each completed task.
- Bound HTTP retries and ignore long `Retry-After` values that threaten the global budget.
- Preserve a provider fallback path and deterministic emergency captions.
- Run two clips concurrently under a 75-second per-task deadline and one retry layer.
- Use the immutable v36 image only as whole-run rollback. The source reconstruction remains available as a C0 profile selected at task initialization but is never entered after v38 inference has spent budget and is never labeled exact v36.

## 4. Alternative arms

### Arm B: MiniMax native-video parity

- `accounts/fireworks/models/minimax-m3` receives the original public video URL.
- First call: 5-7 sentence factual description.
- Second call: four styles together in strict JSON.
- B0 faithfully keeps the observed UniKL generation parameters and retry/normalization behavior without v38 length constraints, forced response format, or repair.
- B1 differs from B0 only by the v38 validator and one targeted repair.

This arm directly reproduces the architecture associated with the current second-place score. It is the preferred fallback if frame-based Kimi misses temporal changes, audio-visible context, or scene transitions.

### Arm C: selective dual evidence

- Run Kimi frames and MiniMax native video in parallel only for clips classified as multi-scene, action-heavy, text-heavy, or temporally ambiguous.
- The joint writer receives both descriptions labeled as independent observations.
- It uses details present in either observation when they do not conflict, rather than intersecting the descriptions and deleting recall.
- This arm is accepted only if its accuracy gain exceeds its latency and contradiction cost on the stress corpus.

### Rejected primary approaches

- Mandatory same-model draft/verify: not validated by the leaders and can repeat the same mistake.
- Four independent style calls: increases drift, latency, and inconsistent fact coverage.
- Systematic reranking or critique: top pipelines do not require it, and more complex public competitors currently score lower.
- Bundling a large local VLM into the submitted image: model weights plus runtime risk exceed the value until a stable evaluation GPU is guaranteed.

## 5. MI300X notebook evaluation

The AMD notebook is a zero-Fireworks-credit teacher and benchmark environment, not an unproven dependency of the submitted container.

### 5.1 Models

Run one model at a time, pinned to exact Hub revisions:

1. `Qwen/Qwen3-VL-4B-Instruct` at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
2. `openbmb/MiniCPM-V-4.6` at revision `8169864629825dc1d755a5aa1cd8b5935dcbc83f`.
3. `OpenGVLab/InternVideo2_5_Chat_8B` at revision `87680fe232af8681b55322f68541679b91f99df1`.
4. `internlm/CapRL-Qwen3VL-4B` at revision `1db1c1dd241e2df95b59846a94cdee5300de9ef9` as a caption-detail teacher, not the production writer.

Start with Qwen3-VL-4B. MiniCPM-V measures the speed ceiling; InternVideo2.5 is used only if the first two miss temporal events.

### 5.2 Local factual benchmark

- Corpus: three public examples plus all direct-download stress clips already in the repository.
- Compare 8-frame uniform, 16-frame uniform, and native 1 FPS capped at 128 frames where supported.
- Every model returns the same factual JSON ledger: setting, subjects, actions, chronology, objects, OCR, colors/counts with confidence, and unsupported/uncertain fields.
- Measure warm latency, p95 latency, peak HBM, valid-output rate, visible-detail recall, false-detail count, temporal accuracy, and OCR accuracy.
- Create human gold ledgers with claim timestamps and evidence before viewing model output. Use blind double human review and deterministic claim matching against this gold. Do not use a model-produced ledger as truth or the same model as both generator and sole judge.
- A local teacher may improve prompts or create a reference ledger, but it enters the production route only if it remains reachable during official evaluation and passes the global runtime test.

## 6. A/B matrix

All arms run on identical videos, in identical order, with two repetitions after warm-up.

| Arm | Unique purpose |
|---|---|
| R0 | Immutable scored v36 index digest sha256:161efc8b098a6a46f395f01fb83ce7c41a9e71c61a51205b59934393bac5f19d |
| C0 | Controlled source reconstruction from commit 283ce7f; never labeled exact v36 |
| M0 | C0 plus only the public DescribeX model/fallback set |
| F1 | M0 plus only OCI-exact frame geometry |
| D1 | M0 plus only the 150-250 word rich description |
| L1 | M0 plus only 2-4 sentence/40-120 word caption instructions |
| J1 | M0 plus only one joint four-style JSON call |
| A_FULL | M0 plus F1+D1+L1+J1 |
| A_HARD | A_FULL plus only targeted repair |
| DESCRIBEX_OFFICIAL_REPO | Separate 16-frame, temperature-0.3, Kimi/GPT-OSS-120B official-repository profile |
| B0 | Faithful UniKL MiniMax native-video profile |
| B1 | B0 plus only validator and targeted repair |

Run order:

1. Offline/unit/contract tests for every arm.
2. Three explicit public clips for smoke only.
3. One interleaved run of each simple ablation on the 12 stress development clips.
4. Two repetitions only for the controlled baseline and finalists.
5. Freeze the finalist hashes.
6. Open a checksum-pinned 12-clip confirmation corpus that is new to the repository and was human-annotated before model output, then run the independent panel and human adjudication once.

## 7. Acceptance gates

A candidate is eligible for submission only when all conditions hold:

- 100% of tasks and requested styles are present in valid JSON.
- Complete official-like batch finishes in at most 535 seconds; target is at most 480 seconds.
- Nearest-rank task p95 is at most 90 seconds, maximum task time is below 125 seconds, and a single retry layer enforces per-task stage budgets.
- No severe invented subject, action, or event in the reviewed corpus.
- No style has generic fallback text when a model response was available.
- Formal captions retain at least the main subject, action, setting, and one useful secondary detail when visible.
- Creative styles remain fact-complete enough to describe the same scene rather than only the joke.
- Judges are filtered per pair against every model family actually used by baseline and candidate, fallbacks included, and prequalified on at least 20 controlled pairs.
- The primary statistical gate uses one aggregated delta per clip, an exact sign-flip test, and an empirical minimum detectable effect from baseline test/retest noise. Accuracy and style must each be non-inferior by style; a 50/50 mean is reported only as a proxy because official axis weights are unpublished.
- At least two qualified, model-family-independent auditors agree, with full-video human adjudication for sparse-frame uncertainty.
- The final public `linux/amd64` image passes an anonymous manifest inspection and an exact three-task Docker contract run.

If no arm clears these gates, leave the currently displayed 0.9133 leaderboard result untouched rather than consume another submission slot; record that digest attribution is inferred from timestamps, not exposed by Lablab.

## 8. Cost controls

Current standard Fireworks prices used for planning:

- Kimi K2.6: $0.95/M input tokens, $0.16/M cached input, $4.00/M output.
- MiniMax M3: $0.30/M input, $0.06/M cached input, $1.20/M output.
- GPT-OSS-20B: $0.07/M input, $0.035/M cached input, $0.30/M output.
- GPT-OSS-120B: $0.15/M input, $0.015/M cached input, $0.60/M output.

Image/video tokenization varies, so each call records provider usage fields and estimated USD. Cost is a secondary constraint for Track 2, but wasted experimentation is avoided:

- MI300X teacher benchmark first: zero Fireworks tokens.
- Paid public-clip smoke second, then one interleaved development pass for single-lever ablations.
- Two repetitions and remote panel only for frozen finalists.
- Default experimental API ceiling: $0.75 per complete decision cycle unless the user explicitly raises it.
- Cache by video digest, model, prompt revision, frame positions, and generation parameters so repeated judging never repeats vision inference.

## 9. Submission and score policy

- Do not resubmit while v37 is still unevaluated.
- Preserve every image tag, source commit, prompt hash, result file, runtime, and cost report.
- Change one major quality lever per scored submission whenever possible.
- If v37 scores below 0.9133, base v38 on the exact v36 behavior plus only the joint rich-description path.
- If v37 scores from 0.9133 through 0.9217, isolate joint generation and caption length before adding a second vision model.
- If v37 exceeds 0.9217, freeze quality behavior and change only operational defects.
- A new score below the baseline triggers evidence review before any further submission.

## 10. Security and licensing

- DescribeX and UniKL provide useful observable architecture, but their current repositories do not provide a dependable source-code licence for verbatim reuse. Reimplement the design independently.
- Use only dedicated, revocable provider credentials in a public Track 2 image. Never reuse personal or production credentials.
- Because Track 2 injects no API key, a safe long-term design is a restricted proxy token or dedicated contest key with spending limits. Rotate all embedded keys immediately after judging.
- Never print, inspect, or archive secret values in reports or test logs.

## 11. Deliverables after approval

1. An isolated v38 branch/worktree based on the selected baseline.
2. A leader-parity engine implemented behind an environment switch.
3. Unit, adversarial, contract, retry, length, JSON, and fallback tests.
4. A reproducible MI300X notebook benchmark with pinned models and measurements.
5. A causal local A/B report comparing R0/C0, each single-lever ablation, faithful/hardened MiniMax, and the frozen finalist.
6. A public linux/amd64 image only after all acceptance gates pass.
7. One Lablab resubmission, followed by heartbeat monitoring until the fresh score appears.

## 12. Evidence references

- Track 2 live leaderboard: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live?track=2
- DescribeX official project: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/ryzen-siblings/describex
- DescribeX public source: https://github.com/Anushiv7/DescribeX
- DescribeX similarly branded public OCI package inspected during research: https://github.com/users/espresso-bytes/packages/container/describex-agent
- UniKL official project: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/mfi/unikl-amd-video-captioning-agent
- UniKL public source: https://github.com/iiTzThoha/video-caption-agent
- Fireworks standard serverless pricing: https://docs.fireworks.ai/serverless/pricing
- Kimi K2.6 model page: https://fireworks.ai/models/fireworks/kimi-k2p6
- MiniMax M3 model page: https://fireworks.ai/models/fireworks/minimax-m3
- GPT-OSS-20B model page: https://fireworks.ai/models/fireworks/gpt-oss-20b
- Qwen3-VL official repository: https://github.com/QwenLM/Qwen3-VL
- InternVideo2.5 official repository: https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5
- VideoITG adaptive-frame research: https://github.com/NVlabs/VideoITG
- MetaCaptioner general visual-captioning research: https://github.com/OpenGVLab/MetaCaptioner

The similarly branded DescribeX OCI package is treated as a strong but non-official association with the scored submission. The official project-to-GitHub link is direct; the OCI-to-score relationship is not exposed directly by Lablab and must not be represented as certain.
