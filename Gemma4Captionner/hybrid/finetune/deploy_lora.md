# Deploying your Gemma LoRA on Fireworks

Fireworks lets you upload a LoRA adapter and route inference to it via a
custom model id. The pipeline (`app/pipeline.py`) already reads `STYLE_LORA`
from env — set it to your deployed adapter id and the 4 style calls will
route there instead of the base Gemma.

## Prerequisites

- A trained LoRA adapter directory (e.g. `finetune/out/lora/`) produced by
  `train_gemma_lora.py`. It must contain `adapter_config.json` and
  `adapter_model.safetensors`.
- The Fireworks CLI:

```bash
pip install firectl
firectl signin
```

## Upload

```bash
firectl create model track2-gemma-styler \
    --base-model accounts/fireworks/models/gemma-3-27b-it \
    --model-type peft \
    --path finetune/out/lora
```

The command prints a model id like
`accounts/<your-account>/models/track2-gemma-styler`.

## Deploy on-demand (or serverless if available)

```bash
firectl deploy accounts/<your-account>/models/track2-gemma-styler
```

Wait for `status: DEPLOYED`.

## Wire the container to your LoRA

At submission time (or via the harness env if it allows env passthrough),
bake this into the image or pass it at run:

```bash
docker run --rm \
    -v $(pwd)/in:/input:ro -v $(pwd)/out:/output \
    -e FIREWORKS_API_KEY \
    -e STYLE_LORA=accounts/<your-account>/models/track2-gemma-styler \
    track2-captioner:submit
```

Now the 4 parallel style calls route to your fine-tuned model. Verify
with a smoke test — the captions should sound more consistent per-style
and better match your training few-shots.

## Cost-check

- Serverless Gemma 3 27B on Fireworks: about **$0.90 / 1M tokens** both
  in and out ([fireworks.ai/models/…/gemma-3-27b-it](https://fireworks.ai/models/fireworks/gemma-3-27b-it)).
- Judging harness runs ~12 clips × 4 styles × ~150 tokens output ≈ 7 k
  output tokens per submission. Cost per submission ≈ **< 1 ¢**.
- Fine-tune training (Unsloth, 27B, 3 epochs on 800 rows) is ~30-60 min
  on one MI300X — well within your $100 AMD Cloud credits.

## Tips

- Start with **4B** for iteration speed, promote to **27B** once your
  dataset is stable. LoRA on 4B trains in 5-10 minutes on a T4.
- Keep the LoRA rank low (r=16). Higher rank helps only if your dataset
  passes 2000+ rows.
- Re-train after any prompt change — the base model changes response
  patterns, and your LoRA is anchored to the training-time prompt format.
