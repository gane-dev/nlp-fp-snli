from datasets import load_dataset, Dataset
import json






# train_snli_electra.py
# Fine-tunes ELECTRA-small on SNLI with Hugging Face Trainer.
# Requires: pip install transformers datasets evaluate accelerate

import argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed
)
import evaluate
from transformers.trainer_utils import SaveStrategy, EvaluationStrategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="google/electra-small-discriminator")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="checkpoints/electra-snli")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--report_to", type=str, default="none")  # "wandb","tensorboard","none"
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) Load SNLI and drop unlabeled rows (label == -1)

    snli = load_dataset("snli")
    orig_train = snli["train"].filter(lambda x: x["label"]!=-1)
    # load adv train jsonl into HF Dataset
    adv_list = [json.loads(l) for l in open("data/adversarial/train/snli_adv_train.jsonl")]
    # Map adv records: keys should be "premise","hypothesis","label" with label ints
    from datasets import Dataset
    adv_ds = Dataset.from_list([{"premise": r["premise"], "hypothesis": r["hypothesis"], "label": r["label"]} for r in adv_list])

    # sample sizes
    n_orig = len(orig_train)
    n_adv_needed = int(0.3 * n_orig)
    # if adv smaller, you can upsample adv with random.choice or reuse adv multiple times
    adv_ds = adv_ds.shuffle(seed=42)
    if len(adv_ds) < n_adv_needed:
        # upsample
        from datasets import Dataset
        repeats = (n_adv_needed // len(adv_ds)) + 1
        adv_list_big = (adv_ds.to_list() * repeats)[:n_adv_needed]
        adv_ds = Dataset.from_list(adv_list_big)
    else:
        adv_ds = adv_ds.select(range(n_adv_needed))

    # select same number of orig (or use all orig)
    orig_part = orig_train.select(range(n_orig - n_adv_needed))  # keep majority original
    mix_train = orig_part.concatenate(adv_ds).shuffle(seed=123)

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def preprocess(batch):
        return tok(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding=False,
            max_length=args.max_len,
        )
    mix = mix_train.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])
    mix.set_format("torch")
    val = snli["validation"].filter(lambda x: x["label"]!=-1).map(preprocess, batched=True, remove_columns=["premise","hypothesis"]).with_format("torch")
    #ds = ds.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # 3) Model (3 labels: entailment=0, neutral=1, contradiction=2 in SNLI)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    # 4) Metrics
    acc = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc.compute(predictions=preds, references=labels)

    # 5) Training args
    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        eval_strategy = EvaluationStrategy.EPOCH,
        save_strategy=SaveStrategy.EPOCH,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        fp16=args.fp16,
        logging_steps=50,
        seed=args.seed,
        report_to=None if args.report_to == "none" else args.report_to,
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mix,
        eval_dataset=val,
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # 7) Train + eval + test
    trainer.train()
    print("\nBest dev metrics:", trainer.evaluate(val))
    #print("\nTest metrics:", trainer.evaluate(ds["test"]))

    # 8) Save final
    trainer.save_model(args.out + "/adv_best")
    tok.save_pretrained(args.out + "/adv_best")

if __name__ == "__main__":
    main()
