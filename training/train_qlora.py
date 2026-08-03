"""Qwen2.5-1.5B-Instruct üzerinde QLoRA (4-bit NF4) fine-tuning.

    python prepare_dataset.py --n 1500
    python train_qlora.py --epochs 3

Tek 16 GB GPU'da (T4/A10/4080) rahat çalışır. Çıktı: output/triaj-qlora
"""
import argparse
import json
import pathlib
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "worker"))
from model import SYSTEM_PROMPT  # noqa: E402  tek kaynak: prompt worker ile aynı olmalı


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_dataset(rows: list[dict], tok, max_len: int) -> Dataset:
    """Sadece cevap token'larına loss uygula (prompt maskeli)."""
    samples = []
    for row in rows:
        user = (f"Kanal: {row['channel']}\nMüşteri segmenti: {row['customer_tier']}\n\n"
                f"Talep:\n{row['text']}")
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        answer = json.dumps(row["label"], ensure_ascii=False) + tok.eos_token

        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        samples.append({"input_ids": input_ids, "labels": labels,
                        "attention_mask": [1] * len(input_ids)})
    return Dataset.from_list(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="output/triaj-qlora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,      # QLoRA'nın çift kuantizasyonu
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_dataset(load_rows(args.train), tok, args.max_len)
    val_ds = build_dataset(load_rows(args.val), tok, args.max_len)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit",         # 4-bit eğitimde bellek dostu
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Adaptör kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
