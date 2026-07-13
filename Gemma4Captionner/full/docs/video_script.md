# Gemma 4 Captioning V18 demo script

## 0:00-0:20 - The problem

**Visual:** four caption styles beside one uploaded clip.

**Voice-over:** "One caption is not enough. The Track 2 judge checks both what
the video shows and whether formal, sarcastic, tech humour and everyday humour
actually sound different."

## 0:20-0:55 - Gemma 4 sees the evidence

**Visual:** timeline with 24 sampled frames, followed by two 60-second MP4
segments for a two-minute clip.

**Voice-over:** "V18 uses only Gemma 4. Gemma 4 31B observes 24 representative
frames in parallel. Gemma 4 26B also inspects the video directly in segments of
at most one minute, but can add only facts consistent with the visual evidence."

## 0:55-1:25 - Four distinct voices

**Visual:** facts consolidate into the four output cards.

**Voice-over:** "The same Gemma family consolidates facts, writes every required
style, strengthens sarcasm and both kinds of humour, then removes unsupported
claims. Repeated stock jokes are detected and rewritten."

## 1:25-1:45 - Submission proof

**Visual:** terminal showing the public pull and valid results JSON.

```bash
docker pull ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
docker run --rm -v "$PWD/in:/input:ro" -v "$PWD/out:/output" \
  ghcr.io/mvconceptlabai/gemma4-captioner:gemma4-submission-v18
```

**Voice-over:** "The Linux amd64 image reads tasks from `/input`, always writes
validated JSON to `/output`, and keeps the full batch below the ten-minute
limit. Gemma 4 sees, writes, jokes and verifies end to end."
