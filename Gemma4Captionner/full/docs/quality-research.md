# Quality Research: Prompts, Skills, Plugins

Last updated: 2026-07-08.

## Objective

Improve the Track 2 captioner for leaderboard performance while preserving the
hackathon contract: Docker entrypoint, `/input/tasks.json`, `/output/results.json`,
four required English styles, valid JSON, runtime safety, and no leaked secrets.

## High-Value Findings

### 1. Keep the two-stage architecture

The strongest reusable pattern is still: visual understanding first, style
rewriting second. The `two-stage-video-captioner` repo reports that asking one
VLM to both see and obey restrictive filters causes leakage, while a second text
rewrite stage follows structured rules more reliably. This matches our existing
`DESCRIBE -> STYLE x4` pipeline, so we should deepen it rather than replace it.

Applied locally:

- `DESCRIBE_SYSTEM` now asks for richer visual details and temporal progression.
- `STYLE_PROVIDER_ORDER` can differ from `DESCRIBE_PROVIDER_ORDER`.
- Safety and taste rules are enforced after model output, not only in prompts.

### 2. Ask VLMs for fine-grained temporal and spatial evidence

NVIDIA's VLM prompting guide emphasizes structured JSON for image/video
understanding and warns that single-frame sampling can miss temporal context.
VCapsBench argues that video caption quality depends on fine-grained dimensions
such as action, camera movement, object position, shot type, and temporal
continuity.

Applied locally:

- Prompt asks for `visual_details`, `temporal_progression`, and uncertainties.
- Frame extraction remains hybrid scene-change plus uniform sampling.
- `quality_audit.py` flags generic captions that ignore visible evidence.

Next improvement:

- Add optional `camera_motion`, `shot_type`, and `spatial_relations` fields to
  the facts schema when model token budget allows.

### 3. Taste must be executable, not just subjective

Impeccable and Taste Skill converge on the same lesson: AI outputs drift toward
generic patterns unless anti-patterns are explicit and tested. Pilcrow applies
the same idea to prose with deterministic AI-tell rules.

Applied locally:

- `PRODUCT.md` now fixes the product register and objective: leaderboard first.
- `quality_audit.py` flags AI-tell phrases, generic filler, style bleed, and
  unsafe taste.
- `detail_audit.py` rejects local prompt variants that score well but lose
  concrete visual anchors.
- `grounding_audit.py` rejects known unsupported terms after manual frame
  inspection of the public sample clips.
- Prompting now separates reliable fine-grained observations from uncertain
  details, so exact counts, animal breeds, and ambiguous eye colors are not
  promoted into final captions.
- `models.py` rejects sensitive appearance jokes and low-taste phrases before
  results are written.

### 4. Use rubric judging before official submissions

Promptfoo's `llm-rubric`, DeepEval, and LLM-as-a-judge surveys all support
custom rubric-based evals. For this hackathon, the useful proxy is two-axis
scoring: video accuracy and style match.

Applied locally:

- `eval/local_judge.py` is now provider-agnostic: Fireworks, OpenRouter, or Groq.
- Fireworks-first judging is wired but treated as opportunistic: malformed JSON
  score objects are rejected, then the judge falls through to OpenRouter/Groq.
- Evidence-Locked Captioning v1 is implemented as an optional inference-time
  candidate/repair loop; it remains an A/B tool until it beats the canonical
  rich-detail output.
- OpenRouter proxy run on `out/demo_quality_results.json` scored:
  - mean accuracy: `1.000`
  - mean style match: `1.000`
  - mean final: `1.000`
  - captions scored: `12`
- Fireworks-first proxy run with fallback on the same canonical output scored
  mean final `0.975` locally, with visible Fireworks rate-limit and non-JSON
  failures; this is useful as an extra signal, not as the promoted score.

Next improvement:

- Run judge A/B comparisons across prompt variants and keep the best-scoring
  prompt bundle, not the newest prompt bundle.

## Recommended Skills / Plugins

Use now:

- `impeccable`: product UI polish, anti-slop detection, PRODUCT/DESIGN context.
- `frontend-design`: concrete HTML/CSS implementation.
- `verification-before-completion`: no success claim without fresh checks.
- `deep-research`: sourced research when changing strategy.

Useful external references, not installed into runtime:

- `Leonxlnx/taste-skill`: stronger anti-generic design heuristics and
  Codex-oriented `gpt-taste`.
- `SamGalanakis/pilcrow`: deterministic prose anti-tells; useful inspiration
  for `quality_audit.py`.
- `promptfoo` / `deepeval`: rubric eval frameworks if we later want a full
  prompt regression suite.
- `lmms-eval` / `VLMEvalKit`: heavyweight multimodal benchmark infrastructure;
  too large for this submission runtime, but good for offline model selection.

Avoid for the Docker runtime:

- Installing UI/design skills or eval frameworks inside the submitted image.
  They improve development, but the competition image should stay small and
  focused on caption generation.

## Source Links

- https://github.com/cseti007/two-stage-video-captioner
- https://github.com/leonxlnx/taste-skill
- https://github.com/pbakaus/impeccable
- https://github.com/SamGalanakis/pilcrow
- https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/
- https://arxiv.org/html/2505.23484v1
- https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric/
- https://github.com/confident-ai/deepeval
- https://github.com/EvolvingLMMs-Lab/lmms-eval
