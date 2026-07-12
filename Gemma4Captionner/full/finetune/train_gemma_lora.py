"""
LoRA fine-tune Gemma 3 as a Style Rewriter.

Runs on AMD Instinct MI300X (AMD Developer Cloud) OR a single NVIDIA GPU.
Uses Unsloth for the 22 GB VRAM fit that lets 27B fit in one card
(per unsloth.ai/blog/gemma3).

Prerequisites — install the training stack (NOT baked in the runtime image):

    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth"
    pip install --no-deps trl==0.11.4 peft==0.13.2 accelerate==1.0.1 bitsandbytes==0.44.1
    pip install datasets==3.1.0

For AMD MI300X (ROCm 6.x):
    pip install torch==2.5.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2

Usage:
    python finetune/train_gemma_lora.py \\
        --dataset finetune/dataset_v2.jsonl \\
        --model   unsloth/gemma-3-4b-it \\
        --output  finetune/out \\
        --epochs  3 \\
        --batch-size 2

For 27B on MI300X (192 GB HBM3): swap --model unsloth/gemma-3-27b-it and
bump --batch-size 4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TMPL = """<start_of_turn>system
You write video captions in the requested style.
<end_of_turn>
<start_of_turn>user
Style: {style}

Scene facts:
{facts}

Write the caption now.
<end_of_turn>
<start_of_turn>model
{caption}<end_of_turn>"""


def _load(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(
            {
                "text": PROMPT_TMPL.format(
                    style=obj["style"],
                    facts=json.dumps(obj["facts"], ensure_ascii=False),
                    caption=obj["caption"],
                )
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", default="unsloth/gemma-3-4b-it")
    p.add_argument("--output", type=Path, default=Path("finetune/out"))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true",
                   help="Load the dataset and validate schema but do not train")
    args = p.parse_args()

    print(f"Loading dataset {args.dataset}")
    rows = _load(args.dataset)
    print(f"-> {len(rows)} training rows")
    # Sanity: show one prompt.
    print("--- example row ---")
    print(rows[0]["text"][:600])
    print("--- end ---")

    if args.dry_run:
        print("Dry run - skipping training.")
        return

    # =============================================================
    # Real training path — imports live here so `--dry-run` works
    # in environments without torch/unsloth installed.
    # =============================================================
    from unsloth import FastLanguageModel  # type: ignore
    from datasets import Dataset  # type: ignore
    from trl import SFTTrainer  # type: ignore
    from transformers import TrainingArguments  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    ds = Dataset.from_list(rows)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=False,
            bf16=True,
            logging_steps=5,
            output_dir=str(args.output),
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()
    (args.output / "lora").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "lora")
    tokenizer.save_pretrained(args.output / "lora")
    print(f"LoRA saved -> {args.output / 'lora'}")


if __name__ == "__main__":
    main()
