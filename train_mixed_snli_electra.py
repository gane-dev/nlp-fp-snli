#!/usr/bin/env python
# Mix SNLI train with clean TextFooler JSONL (70/30 by default) and fine-tune ELECTRA-small.

import argparse, json, math, random, os
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)
import evaluate
import numpy as np
from transformers.trainer_utils import EvaluationStrategy, SaveStrategy


def load_clean_adv(jsonl_path):
    adv_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
    # Expect keys: premise, hypothesis, label (0/1/2)
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # expected: {"premise": "...", "hypothesis": "...", "label": 0/1/2}
            if r.get("premise") and r.get("hypothesis"):
                lab = int(r["label"])
                if lab in (0, 1, 2):
                    adv_list.append({
                        "premise": r["premise"],
                        "hypothesis": r["hypothesis"],
                        "label": lab})
    return adv_list

def get_snli_datasets(train):
    snli = load_dataset("snli")
    if train ==0:
        snli_train = snli["train"].filter(lambda x: x["label"] != -1)
    elif train == 1:
        snli_train = snli["validation"].filter(lambda x: x["label"] != -1)
    else:
        snli_train = snli["test"].filter(lambda x: x["label"] != -1)

    snli_list = [
        {
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "label": int(ex["label"])  # 0/1/2
        }
        for ex in snli_train
    ]
    return snli_list

def build_mixed_train(snli_list, adv_list, adv_ratio=0.3, seed=42):
    import random
    random.seed(seed)

    n_total = len(snli_list)
    n_adv = int(round(adv_ratio * n_total))
    n_orig = n_total - n_adv

    # sample adversarial (with upsampling if needed)
    if len(adv_list) >= n_adv:
        adv_sample = random.sample(adv_list, n_adv)
    else:
        # upsample with replacement
        repeats = (n_adv // len(adv_list)) + 1
        adv_big = (adv_list * repeats)[:n_adv]
        random.shuffle(adv_big)
        adv_sample = adv_big

    # sample original
    orig_sample = random.sample(snli_list, n_orig)

    mixed_list = orig_sample + adv_sample
    random.shuffle(mixed_list)
    print("Mixed total:", len(mixed_list))
    print("Adv fraction:", len(adv_sample) / len(mixed_list))
    return mixed_list

def get_mixed_dataset(mixed_list):
    from datasets import Dataset, Features, Value

    features = Features({
        "premise": Value("string"),
        "hypothesis": Value("string"),
        "label": Value("int64"),
    })

    mix_train = Dataset.from_list(mixed_list, features=features)
    print(mix_train.features)

    return Dataset.from_list(mixed_list)



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

    train = get_snli_datasets(0)
    val = build_mixed_train(get_snli_datasets(1),[],0.0)
    test= build_mixed_train(get_snli_datasets(2),[],0.0)

    mix_train = build_mixed_train(
        snli_list=train,
        adv_list=load_clean_adv(args.adv_jsonl),
        adv_ratio=args.adv_ratio,
        seed=args.seed
    )
    print(f"[INFO] Mixed train size: {len(mix_train)} | "
          f"orig≈{len(train) - int(round(args.adv_ratio * len(train)))} | adv≈{int(round(args.adv_ratio * len(train)))}")

    # 4) Tokenizer & preprocessing
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
        return tok(batch["premise"], batch["hypothesis"],
                   truncation=True, padding=False, max_length=args.max_len)

    mixed_ds = get_mixed_dataset(mix_train)
    val_ds = get_mixed_dataset(val)
    test_ds= get_mixed_dataset(test)
    mix_enc = mixed_ds.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])
    val_enc = val_ds.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])
    test_enc= test_ds.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])

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
        eval_strategy = EvaluationStrategy.EPOCH,
        save_strategy=SaveStrategy.EPOCH,
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
    best_dir = os.path.join(args.out, "adv_best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    print(f"[INFO] Saved best model to: {best_dir}")

if __name__ == "__main__":
    main()
