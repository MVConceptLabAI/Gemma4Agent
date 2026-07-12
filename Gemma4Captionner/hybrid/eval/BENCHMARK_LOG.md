# Benchmark Log — Track 2 Captioner

Persistent record of every caption config tested, scored by the **adversarial
vision audit** (Opus agents read the real frames and refute each claim). Two
axes that matter, measured on the 3 public demo clips (12 captions each unless
noted):

- **Precision** = mean per-claim accuracy (0-1); a contradicted claim is a hard error.
- **Detection** = total correct concrete details captured (higher = richer). `det/cap` = per caption.
- **Contra** = total contradictions (frames disprove the claim).

Never trust the old 8B `local_judge` — it scored wrong captions ~1.0. This log
is the source of truth. Append a row after every audit; keep the best config at top.

## Results (best precision+detection first)

| Date | Describe | Writer | Input | Precision | Detection (tot / per-cap) | Contra | Notes |
|---|---|---|---|---|---|---|---|
| 07-09 | **ENSEMBLE** (GPT-5.5 + Gemini-3.1-Pro + Opus-4.5) | Opus 4.5, no length cap | 10 stills @896 | 0.951 | **209 / 17.4** | 1 | ~3x the detail of any single model; reads the sign RIGHT (cross-model agreement: KOREA ILLIES + Starbucks + TAXPARK + 8 towers); 0 safety. WINNER for max-detail. |
| 07-09 | GPT-5.5 (e2e) | GPT-5.5 | 10 stills @896 | 0.972 | **120 / 10.0** | 1 | densest detail by far; ⚠️ 2 safety (stated race) |
| 07-09 | Gemini 3.1 Pro (e2e) | Gemini 3.1 Pro | 10 stills @896 | 0.943 | 87 / 7.3 | 0 | rich, 0 contradiction |
| 07-09 | Qwen3-VL-235B (frames) | **Claude Opus 4.5** | 10 stills @896 | **1.000** | 79 / 6.6 | 0 | perfect precision, grounded |
| 07-09 | Claude Opus 4.5 (e2e) | Claude Opus 4.5 | 10 stills @896 | 0.989 | 73 / 6.1 | 0 | very precise, natural |
| 07-09 | Qwen3-VL-235B (frames) | Claude Sonnet 4 | 10 stills @896 | 0.975 | 74 / 6.2 | 0 | round-1 winner (now beaten) |
| 07-09 | Gemini 2.5 Pro (e2e) | Gemini 2.5 Pro | 10 stills @896 | 0.967 | 60 / 5.0 | 0 | older; misquotes the sign |
| 07-09 | Claude Sonnet 4 (e2e) | Claude Sonnet 4 | 10 stills @896 | 0.967 | 62 / 5.2 | 1 | "laptop" error |
| 07-09 | Qwen3-VL-235B (frames) | Gemma-3-27B + deterministic | 10 stills @896 | 0.942 | 67 / 5.6 | 3 | robotic; sign misread ILDONG |

### Model access notes (verified)
- **Qwen3-VL via OpenRouter = image input only** (no video modality); Fireworks has
  no Qwen-VL deployed. To use Qwen-VL on real video, self-host it on the AMD MI300X
  (this is also the strongest "Use of AMD Platforms" story).
- **Gemini 3.1 Pro / 2.5 Pro accept real VIDEO.** On video input Gemini correctly
  reports the street sign as unreadable instead of hallucinating a company name —
  video beats sampled frames for temporal + honesty. (video e2e generator: add --video.)
- **GPT-5.5** detects ~2x the detail of any other model but must be told NOT to state
  race/skin (added to the ensemble/premium prompts).

## Multi-video generalization (beyond the 3 samples)

Official test set = the 15 clips in the AMD hackathon bucket
(`data/official_tasks.json`) — the exact distribution the judge samples from.
Plus 12 diverse Pexels stress clips. Audited on the real frames:

| Set | Config | Precision | Detection (tot / per-cap) | Contra | Safety |
|---|---|---|---|---|---|
| 12 diverse clips | ENSEMBLE (old writer) | 0.931 | 691 / 14.4 | 17 (0.35/cap) | 0 |
| 12 official clips | ENSEMBLE (old writer) | 0.924 | 745 / 15.5 | 20 (0.42/cap) | 0 |
| 11 official clips | **ENSEMBLE (strict writer) — SUBMISSION** | **0.942** | 652 / 14.8 | **13 (0.30/cap)** | 0 |

**Iteration result:** the strict writer (drop single-model specific claims unless
2+ models agree) lifted precision 0.924->0.942 and cut contradictions ~30% per
caption on the official jury distribution, keeping nearly all the detail. Adopted.

Iteration note: the ensemble's huge detail (14.4/caption) came with 17
contradictions on 48 captions — almost all from a SPECIFIC detail only one model
reported. Writer prompt tightened: single-model specific claims (exact color,
brand, count, sign text, left/right/fg-bg placement) are dropped unless a second
model agrees. Trades a little detail for precision; re-audited above.

## Ground truth notes (verified by looking at frames)

- v1 (Korean street): time-lapse/long-exposure motion blur, multi-lane + intersection,
  yellow ginkgo + some green trees, tall apartment high-rises, distant hills, green
  traffic signals, colored banners right sidewalk, a bus in traffic. Sign reads
  **"KOREA ILLIES ENGINEERING"** — UNreadable to every VLM (Qwen→ILDONG, Gemini→ILJIN);
  **do not quote it.**
- v2 (kitten): fluffy long-haired **orange/ginger** kitten, white chest, pink nose,
  walks toward camera, tail raised in last frame, dirt path + dry leaves, green
  **bushes/undergrowth** (NOT a garden/forest), tree trunk left, dappled sunlight.
  Eye color NOT reliable — do not state it.
- v3 (office): woman, dark curly hair high **bun**, small stud earrings, **silver cross
  pendant**, beige button-up over coral/orange top, **bright pink nails**, types on a
  **desktop monitor** (NOT a laptop) with a **black wired mouse** + coiled cable,
  white desk, green potted plant, circular ceiling lights, glass partitions.

## Method

`python scripts/premium_caption.py --model <id> --tasks data/sample_tasks.json --out out/x.json`
generates end-to-end (one VLM writes all 4). The pipeline (`python -m app.main`)
does Qwen-describe → writer. Audit with the precision-detection workflow, then
append the row here.
