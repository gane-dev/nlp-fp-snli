#!/usr/bin/env python
# Mix SNLI train with clean TextFooler JSONL (70/30 by default) and fine-tune ELECTRA-small.

import argparse, json, math, random, os
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)
import evaluate
import numpy as np

def load_clean_adv(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    # Expect keys: premise, hypothesis, label (0/1/2)
    clean = []
    for r in recs:
        if not r.get("premise") or not r.get("hypothesis"):
            continue
        lab = r.get("label")
        try:
            lab = int(lab)
        except:
            continue
        if lab not in (0,1,2):
            continue
        clean.append({"premise": r["premise"], "hypothesis": r["hypothesis"], "label": lab})
    return Dataset.from_list(clean)

def build_mixed_train(orig_train, adv_ds, adv_ratio=0.3, seed=42):
    """
    Build a mixed dataset where ~adv_ratio of examples are adversarial.
    Keeps total size ~= len(orig_train) to avoid inflating epochs.
    """
    rng = random.Random(seed)
    n_total = len(orig_train)
    n_adv = int(round(adv_ratio * n_total))
    n_orig = n_total - n_adv

    # sample / upsample adversarial examples
    adv_ds = adv_ds.shuffle(seed=seed)
    if len(adv_ds) >= n_adv:
        adv_part = adv_ds.select(range(n_adv))
    else:
        # upsample with replacement
        reps = math.ceil(n_adv / len(adv_ds))
        adv_list = adv_ds.to_list() * reps
        rng.shuffle(adv_list)
        adv_part = Dataset.from_list(adv_list[:n_adv])

    # sample original portion without replacement
    orig_part = orig_train.shuffle(seed=seed).select(range(n_orig))

    mixed = concatenate_datasets([orig_part, adv_part]).shuffle(seed=seed+1)
    return mixed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adv_jsonl", required=True, help="Path to clean TextFooler JSONL (premise,hypothesis,label).")
    ap.add_argument("--model_name", default="google/electra-small-discriminator")
    ap.add_argument("--out", default="checkpoints/robust-electra-snli")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--adv_ratio", type=float, default=0.30, help="Fraction of mixed train that is adversarial.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # 1) Load SNLI and filter labels
    snli = load_dataset("snli")
    train = snli["train"].filter(lambda x: x["label"] != -1)
    val   = snli["validation"].filter(lambda x: x["label"] != -1)
    test  = snli["test"].filter(lambda x: x["label"] != -1)

    # 2) Load clean adversarial JSONL
    adv_ds = load_clean_adv(args.adv_jsonl)
    if len(adv_ds) == 0:
        raise ValueError("No valid adversarial records found in JSONL.")

    # 3) Build mixed train (70/30 by default)
    mix_train = build_mixed_train(train, adv_ds, adv_ratio=args.adv_ratio, seed=args.seed)
    print(f"[INFO] Mixed train size: {len(mix_train)} | "
          f"orig≈{len(train) - int(round(args.adv_ratio * len(train)))} | adv≈{int(round(args.adv_ratio * len(train)))}")

    # 4) Tokenizer & preprocessing
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
        return tok(batch["premise"], batch["hypothesis"],
                   truncation=True, padding=False, max_length=args.max_len)

    mix_enc = mix_train.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])
    val_enc = val.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])
    test_enc= test.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])

    mix_enc.set_format("torch", columns=["input_ids","attention_mask","label"])
    val_enc.set_format("torch", columns=["input_ids","attention_mask","label"])
    test_enc.set_format("torch", columns=["input_ids","attention_mask","label"])

    # 5) Model & Trainer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    data_collator = DataCollatorWithPadding(tokenizer=tok)
    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=args.fp16,
        seed=args.seed,
        report_to=None,  # set to "tensorboard" or "wandb" if you use them
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mix_enc,
        eval_dataset=val_enc,
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # 6) Train & evaluate
    trainer.train()
    print("\n[DEV] ", trainer.evaluate(val_enc))
    print("[TEST]", trainer.evaluate(test_enc))

    # 7) Save best checkpoint
    best_dir = os.path.join(args.out, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    print(f"[INFO] Saved best model to: {best_dir}")

if __name__ == "__main__":
    main()
