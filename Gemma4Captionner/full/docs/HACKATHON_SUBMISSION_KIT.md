# Hackathon submission kit — Track 2

## Title

`Gemma 4 SceneGate: Grounded Multi-Style Video Captioning`

## Short description

Gemma 4 SceneGate grounds multi-style video captions in independent visual observations, then produces factual, sarcastic, humorous-tech and humorous-non-tech captions for every requested clip.

## Long description

Gemma 4 SceneGate is a Track 2 video-captioning agent designed for reliable, style-aware captions across varied videos. It samples representative moments from each clip, gathers independent visual observations, and consolidates only supported details before writing captions. The final writer produces all four required styles: formal, sarcastic, humorous_tech and humorous_non_tech.

The pipeline treats humour as a framing layer, not a source of new facts. Its humorous-tech caption uses one natural technology analogy tied to a visible moment, while its humorous-non-tech caption uses everyday humour without technical jargon. Validation ensures every requested style is present and the final output is valid JSON at `/output/results.json`.

## Application fields

| Field | Value |
| --- | --- |
| GitHub Repository | `https://github.com/MVConceptLabAI/Gemma4Agent/tree/gemma4-submission-v14/Gemma4Captionner/full` |
| Demo Application Platform | `Other` |
| Demo Application URL | `https://mvconceptlabai.github.io/Gemma4Agent/demo.html` |
| Docker Image | `ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v14` |

## Additional Information

```text
Track 2 submission. The container reads /input/tasks.json at startup and writes validated /output/results.json before exit, including a caption for every requested style. It supports formal, sarcastic, humorous_tech and humorous_non_tech output for each clip.

SceneGate samples representative video moments, combines independent visual observations, and uses Gemma 4 31B as the evidence-consolidation writer. Grounding rules prevent the style layer from adding unsupported people, actions, brands, speech or off-screen events. The humorous-tech output uses one visible-scene-tied technology analogy; humorous-non-tech keeps to everyday humour without technical jargon.

Public Docker reference verified as a linux/amd64 manifest:
ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v14
Digest: sha256:c109c4d255e94169c064e858043ce506d34273a37250eedd01b60165f0a579bd

Live browser demo: https://mvconceptlabai.github.io/Gemma4Agent/demo.html
```

## Before saving

1. The GitHub link and Docker tag both point to `gemma4-submission-v14`; keep these two references paired.
2. Paste the Docker **tag** exactly as shown, without `https://` and without the digest.
3. Use the GitHub Pages URL for the demo; a `raw.githubusercontent.com` URL displays source code and is not an executable demo.
4. Keep the repository and package public. The Docker manifest above was resolved anonymously as `linux/amd64`.
