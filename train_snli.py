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
    ds = load_dataset("snli")
    ds = ds.filter(lambda x: x["label"] != -1)

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

    ds = ds.map(preprocess, batched=True, remove_columns=["premise","hypothesis"])
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
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # 7) Train + eval + test
    trainer.train()
    print("\nBest dev metrics:", trainer.evaluate(ds["validation"]))
    print("\nTest metrics:", trainer.evaluate(ds["test"]))

    # 8) Save final
    trainer.save_model(args.out + "/best")
    tok.save_pretrained(args.out + "/best")

if __name__ == "__main__":
    main()
