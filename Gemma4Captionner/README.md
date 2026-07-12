# Gemma 4 Captioner experiments

Two independent Track 2 Captioner variants, both derived from
`TheSkyGold/track2-captioner` at `submission-v40` and retaining its W4
per-style writer pipeline.

## Variants

- `full/` — Gemma 4 31B (`google/gemma-4-31b-it`) writes all four caption
  styles after the shared vision-observer factual spine.
- `hybrid/` — Opus 4.5 writes the formal caption; Gemma 4 31B writes the
  sarcastic, humorous-tech, and humorous-non-tech captions. The four writers
  receive the same cross-checked observations.

Each folder is a self-contained source snapshot. Run its W4 test with:

```bash
PYTHONPATH=. python scripts/test_w4_style_split.py
```

The hybrid routing guard is additionally available at:

```bash
PYTHONPATH=. python scripts/test_w4_hybrid_writers.py
```

These are A/B candidates, not verified leaderboard submissions. Test them with
a funded provider account and the target 2 CPU / 4 GB container limits before
publishing an image or submitting a score.
