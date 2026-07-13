# v40 W4 promotion evidence - 2026-07-12

## Decision

Promote the W4 per-style writer profile only after the final resource-capped
runtime is repeated with a funded OpenRouter account. Do not submit while the
account returns HTTP 402.

The last evaluated baseline is submission-v38:

- official score: `0.9067`
- rank at observation time: `5`
- resubmitted: `2026-07-12T09:11:00Z`
- evaluated: `2026-07-12T10:19:00Z`
- leader at observation time: Himawari, `0.9292`

## Controlled change

W4 preserves the v38 four-observer factual spine, writer model, temperature,
frames, token budget, validators, retries, concurrency, and fallback path. It
changes one generation decision: the four requested styles are written by four
parallel style-specific calls instead of one call returning all four captions.
The aggregate maximum writer output budget remains 3,000 tokens.

The release hardening additionally retries one malformed writer JSON response.
That does not alter successful inference.

## Paired multimodal evaluation

All panels were blind, position-balanced, and returned complete valid pairs.

### Public three clips, W4 versus exact v38

| Judge | Accuracy delta | Style delta | Final delta | Valid pairs |
|---|---:|---:|---:|---:|
| Mistral Medium 3.1 | +0.0500 | +0.0542 | +0.0521 | 12/12 |
| Llama 4 Maverick | +0.0183 | +0.0208 | +0.0196 | 12/12 |

### Sealed fresh five clips, W4 versus fresh exact v38

The confirmation corpus contains five previously unused clips covering animal
action, a person using technology, manual forging, heavy urban rain, and a
complex night food stall. All URLs, durations, hashes, and contamination checks
are recorded outside the repository in `tmp/fresh5_20260712/PROVENANCE.md`.

| Judge | Accuracy delta | Style delta | Final delta | Worst clip | Valid pairs |
|---|---:|---:|---:|---:|---:|
| Mistral Medium 3.1 | +0.0375 | +0.0575 | +0.0475 | +0.0250 | 20/20 |
| Llama 4 Maverick | +0.0210 | +0.0200 | +0.0205 | -0.0125 | 20/20 |

## Runtime and contract evidence

The W4 candidate completed a 12-clip, 48-caption run in `285.1` seconds:

- exit code `0`
- 12/12 task IDs present
- 48/48 requested captions non-empty
- valid JSON contract
- no static fallback caption
- maximum caption length `1,569` characters under the configured 1,600 limit
- one observer transport failure was tolerated by the remaining observers

This run used the W4 flag explicitly on the development image. The release
Dockerfile now pins `W4_STYLE_SPLIT=1`, and an independently built image reports:

- platform: `linux/amd64`
- `CAPTION_ENGINE=ensemble`
- `W4_STYLE_SPLIT=1`
- four observers
- `MAX_CONCURRENCY=2`
- `GLOBAL_BUDGET_S=540`
- uncompressed image size about 252 MB

A second 12-clip run with `--cpus=2 --memory=4g --memory-swap=4g` was correctly
stopped after OpenRouter began returning HTTP 402 because the account balance
was negative. Its outputs are invalid and are excluded from quality/runtime
evidence. Repeat this exact capped run after funding before tagging the release.

## Rejected competitor-derived hypotheses

- A clean-room Qwen3.7 direct-style geometry lost `0.0996` on average versus
  v38 across two judges, despite improving style versus the older direct arm.
- A formal-first anchor modeled only on a public top-three architecture lost
  `0.0250` versus W4 on its first public3 judge, with a worst clip of `-0.0375`.
  It failed the predeclared gate, so no second judge or fresh5 call was spent.
- The complete Himawari reproduction lost `0.1211` to v38 under the paired
  Mistral panel on eight clips. Its architecture was informative; copying the
  full pipeline was not supported by evidence.

## Release gates

- [x] Public3 positive on two independent judges.
- [x] Fresh5 positive on two independent judges.
- [x] 12-clip host runtime below ten minutes with a complete contract.
- [x] W4 enabled in Dockerfile and guarded in CI.
- [x] Malformed per-style writer JSON retried.
- [x] Offline preflight green.
- [ ] OpenRouter balance positive and a one-request paid-model probe returns 200.
- [ ] Exact release image passes 12 clips under 2 CPU / 4 GB in under ten minutes.
- [ ] Public GHCR digest and anonymous contract run verified.
- [ ] Lablab resubmission confirmation captured.

