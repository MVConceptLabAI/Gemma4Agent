# Hackathon submission kit — Track 2

## Title

`Gemma 4 SceneGate: Grounded Multi-Style Video Captioning`

## Short description

Gemma 4 SceneGate uses the same evidence-first Gemma 4 flow as the public demo: it observes sampled video frames independently, then produces grounded factual, sarcastic, humorous-tech and humorous-non-tech captions.

## Long description

Gemma 4 SceneGate is a Track 2 video-captioning agent designed for reliable, style-aware captions across varied videos. It samples 24 representative moments from each clip and asks Gemma 4 for concise, independent visual observations before consolidating only supported details. Gemma 4 then writes all four required styles: formal, sarcastic, humorous_tech and humorous_non_tech, improves both humorous variants, and performs a final grounded review.

The pipeline treats humour as a framing layer, not a source of new facts. Its humorous-tech caption uses one natural technology analogy tied to a visible moment, while its humorous-non-tech caption uses everyday humour without technical jargon. Validation ensures every requested style is present and the final output is valid JSON at `/output/results.json`.

## Application fields

| Field | Value |
| --- | --- |
| GitHub Repository | `https://github.com/MVConceptLabAI/Gemma4Agent/tree/gemma4-submission-v17/Gemma4Captionner/full` |
| Demo Application Platform | `Other` |
| Demo Application URL | `https://mvconceptlabai.github.io/Gemma4Agent/demo.html` |
| Docker Image | `ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v17` |

## Additional Information

```text
Track 2 submission. The container reads /input/tasks.json at startup and writes validated /output/results.json before exit, including a caption for every requested style. It supports formal, sarcastic, humorous_tech and humorous_non_tech output for each clip.

SceneGate samples 24 representative video moments, analyses each frame independently with Gemma 4 31B, then uses Gemma 4 for evidence consolidation, humorous-style polishing and the final grounded review. No other inference provider is used. The humorous-tech output uses one visible-scene-tied technology analogy; humorous-non-tech keeps to everyday humour without technical jargon or invented settings.

Public Docker reference verified as a linux/amd64 manifest:
ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v17
Digest: sha256:bf66e38e1f834cea9597e1994013b1ab2850070d2838342c4687919735f12468

Live browser demo: https://mvconceptlabai.github.io/Gemma4Agent/demo.html
```

## Before saving

1. The GitHub link and Docker tag both point to `gemma4-submission-v17`; keep these two references paired.
2. Paste the Docker **tag** exactly as shown, without `https://` and without the digest.
3. Use the GitHub Pages URL for the demo; a `raw.githubusercontent.com` URL displays source code and is not an executable demo.
4. Keep the repository and package public. The Docker manifest above was resolved anonymously as `linux/amd64`.
