# Gemma 4 Captioning V18

Grounded multi-style video captioning for AMD Developer Hackathon Track 2,
powered exclusively by Gemma 4 models.

V18 reads tasks from `/input/tasks.json`, analyses each video, and writes valid
results to `/output/results.json` before exiting. Every requested style is
normalized and validated so a single provider or model-formatting failure does
not corrupt the complete submission file.

## Public submission image

```bash
docker pull ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
docker run --rm \
  -v /absolute/path/to/input:/input \
  -v /absolute/path/to/output:/output \
  ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
```

Published properties:

- public OCI image
- `linux/amd64`
- approximately 242 MiB compressed
- command: `python -u -m app.main`
- global runtime budget: 540 seconds
- maximum parallel videos: 3

## V18 architecture

The Docker submission uses `CAPTION_ENGINE=gemma_demo`. Every inference request
stays within the Gemma 4 family:

- `google/gemma-4-31b-it` observes frames, consolidates evidence, writes
  captions, rewrites styles, and verifies grounding.
- `google/gemma-4-26b-a4b-it` receives a compact MP4 and adds only temporal or
  audible facts that agree with the frame evidence.

Processing sequence:

1. Download the MP4 with retries.
2. Extract 24 representative frames with FFmpeg, at up to 640 pixels per edge.
3. Analyse frames independently through nine bounded parallel Gemma 4 requests.
4. Compress the MP4 to one frame per second for direct video inspection. A clip
   up to 60 seconds uses one segment; a clip up to two minutes uses two segments
   covering its beginning and end.
5. Consolidate only supported subjects, actions, objects, setting, motion,
   lighting, sequence, and clearly confirmed audio.
6. Generate `formal`, `sarcastic`, `humorous_tech`, and
   `humorous_non_tech` captions.
7. Rewrite sarcasm and both humour styles with evidence-locked prompts.
8. Verify every literal claim and preserve supported visual detail.
9. Repair malformed text, duplicated fragments, unsupported absolutes, OCR
   false positives, generic jokes, and speed-inverting metaphors.
10. Retry once when Gemma returns pseudo-JSON instead of strict JSON, then
    normalize and validate the complete output contract.

V18 does not use GPT, Claude, Gemini, Qwen, Groq, or Fireworks models.

## Track 2 contract

Input file: `/input/tasks.json`

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/video.mp4",
    "styles": [
      "formal",
      "sarcastic",
      "humorous_tech",
      "humorous_non_tech"
    ]
  }
]
```

Output file: `/output/results.json`

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

The process exits with code 0 after successfully writing results. Invalid input
or a fatal startup failure returns a non-zero code. Individual inference errors
are isolated per task so the remaining clips and requested styles are still
written.

## Local inference

Install the pinned dependencies and provide an OpenRouter key:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
export CAPTION_ENGINE=gemma_demo
export INPUT_PATH=data/official_new12.json
export OUTPUT_PATH=out/results.json
python -u -m app.main
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENROUTER_API_KEY = 'sk-or-v1-...'
$env:CAPTION_ENGINE = 'gemma_demo'
$env:INPUT_PATH = 'data\official_new12.json'
$env:OUTPUT_PATH = 'out\results.json'
python -u -m app.main
```

Do not commit `.env` files or API keys. The hackathon does not inject runtime
credentials, so the publication workflow injects a temporary OpenRouter key
from the protected GitHub environment while building the submission image.

## Docker profile

The submitted Dockerfile pins these values:

| Variable | V18 value | Purpose |
|---|---:|---|
| `CAPTION_ENGINE` | `gemma_demo` | Select the Gemma-only pipeline |
| `DEMO_GEMMA_MODEL` | `google/gemma-4-31b-it` | Frame observer and caption writer |
| `DEMO_VIDEO_MODEL` | `google/gemma-4-26b-a4b-it` | Direct MP4 observer |
| `NUM_FRAMES` | `24` | Representative frames per clip |
| `FRAME_MAX_EDGE` | `640` | Maximum frame edge |
| `DEMO_OBSERVATION_CONCURRENCY` | `9` | Parallel frame requests |
| `MAX_CONCURRENCY` | `3` | Parallel videos |
| `VIDEO_CONTEXT_MAX_SECONDS` | `60` | Duration per direct-video segment |
| `VIDEO_CONTEXT_MAX_BYTES` | `12000000` | Maximum encoded segment size |
| `PER_TASK_TIMEOUT_S` | `220` | Per-video timeout |
| `GLOBAL_BUDGET_S` | `540` | Margin below the 10-minute limit |

## Validation evidence

### AMD-hosted 12-task set

Input: [`data/official_new12.json`](data/official_new12.json)

Measured on the V18 source profile:

- 12 tasks loaded from AMD-hosted clip URLs
- 48 requested captions written
- valid `/output/results.json`
- exit code 0
- total runtime: 340.7 seconds
- no task timeout

The run exposed one invalid pseudo-JSON writer response on task `6023186`.
The pipeline still completed with safe captions. V18 now retries malformed JSON
once with strict double-quoted-object instructions. A targeted rerun of
`6023186` completed in 69.9 seconds and produced grounded captions describing
people on a station platform beside a stopped yellow-and-white train.

### Diverse 30-second stress set

A separate 12-video set covering nature, urban traffic, animals, people,
sports, food, weather, and technology completed in 248.2 seconds with 48/48
captions and no missing styles. Two weak humour cases found by that run were
converted into permanent regression checks for generic non-tech fallbacks and
technology metaphors that invert visible speed.

## Offline checks

```bash
PYTHONPATH=. python scripts/test_demo_pipeline_quality.py
python -m compileall -q app scripts/test_demo_pipeline_quality.py
python scripts/contract_test.py
python scripts/preflight.py
```

The quality regression test covers:

- legitimate OCR such as `ERROR` and `BUKD3`
- malformed letter runs and duplicated words
- unsupported stillness and permanence
- generic humorous non-tech fallbacks
- slow technology metaphors applied to fast visible action
- malformed pseudo-JSON followed by a strict-JSON retry

## Publish

The GitHub workflow publishes on tags matching `gemma4-submission-*`:

```bash
git tag -f gemma4-submission-v18
git push origin refs/tags/gemma4-submission-v18 --force
```

The workflow builds a single `linux/amd64` manifest without provenance or SBOM
side manifests, then verifies the image with Docker Buildx.

## Repository layout

- `app/main.py` - Track 2 entry point and runtime budgeting
- `app/demo_pipeline.py` - Gemma-only evidence and caption pipeline
- `app/models.py` - task parsing, normalization, fallbacks, validation
- `data/official_new12.json` - AMD-hosted 12-task validation input
- `docs/` - browser demo and presentation material
- `scripts/test_demo_pipeline_quality.py` - V18 regression checks
- `Dockerfile` - public submission image profile
- `SUBMISSION.md` - hackathon submission copy
