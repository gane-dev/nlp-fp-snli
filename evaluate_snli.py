#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a (fine-tuned) ELECTRA model on an SNLI contrast set JSONL and compute robust accuracy.
Supports:
  1) Hugging Face saved model dirs (config + tokenizer + weights)
  2) Lightning .ckpt files (loads state_dict into HF model)

JSONL expected fields (typical):
  premise
  hypothesis_contrast (default)  OR hypothesis_orig (if --hyp_field=hypothesis_orig)
  label_contrast (default)       OR label_orig     (if --label_field=label_orig)
  phenomenon (optional) for per-slice scores

Label values accepted:
  strings: {"entailment","neutral","contradiction"}   OR
  ints:     {0,1,2} (assumed to map to E/N/C respectively)

Usage:
  python eval_contrast_snli.py \
    --model_path checkpoints/electra-snli/best \
    --contrast_path data/contrast_sets/snli_combined.jsonl \
    --out_csv reports/contrast_eval.csv

  # If you only have a Lightning .ckpt
  python eval_contrast_snli.py \
    --model_path lightning_ckpts/electra-snli-epoch=2-val_acc=0.90.ckpt \
    --base_model google/electra-small-discriminator \
    --contrast_path data/contrast_sets/snli_combined.jsonl
"""
import argparse, json, os, sys, math
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import pandas as pd

LABEL_STR_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID_TO_LABEL_STR = {v:k for k,v in LABEL_STR_TO_ID.items()}

def coerce_label(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x in LABEL_STR_TO_ID:
            return LABEL_STR_TO_ID[x]
        # tolerate common shorthand
        if x in {"ent", "e"}: return 0
        if x in {"neu", "n"}: return 1
        if x in {"con", "contradict", "c"}: return 2
        raise ValueError(f"Unrecognized label string: {x}")
    if isinstance(x, (int, np.integer)):
        if int(x) in {0,1,2}:
            return int(x)
    raise ValueError(f"Unrecognized label value: {x}")

class ContrastJsonDataset(Dataset):
    def __init__(self, records: List[Dict[str,Any]], tokenizer, hyp_field: str, label_field: str, max_len: int = 128):
        self.recs = records
        self.tok = tokenizer
        self.hyp_field = hyp_field
        self.label_field = label_field
        self.max_len = max_len

        # validate fields exist
        for i, r in enumerate(self.recs[:5]):
            if "premise" not in r:
                raise KeyError("Missing 'premise' in JSONL record.")
            if self.hyp_field not in r:
                raise KeyError(f"Missing '{self.hyp_field}' in JSONL record.")
            if self.label_field not in r:
                raise KeyError(f"Missing '{self.label_field}' in JSONL record.")

    def __len__(self): return len(self.recs)

    def __getitem__(self, idx):
        r = self.recs[idx]
        prem = r["premise"]
        hyp = r[self.hyp_field]
        label = coerce_label(r[self.label_field])

        enc = self.tok(
            prem, hyp, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        # keep for reporting
        item["premise_txt"] = prem
        item["hyp_txt"] = hyp
        item["label_int"] = label
        item["phenomenon"] = r.get("phenomenon", None)
        return item

def collate_fn(batch):
    # separate text/meta from tensors
    metas = {}
    for k in ["premise_txt","hyp_txt","label_int","phenomenon"]:
        metas[k] = [b[k] for b in batch]
        for b in batch:
            b.pop(k, None)
    # pad tensors with tokenizer's pad
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "meta": metas}

def try_load_model(model_path: str, base_model_name: Optional[str], num_labels: int = 3):
    """
    1) Try to load as a Hugging Face directory (config + tokenizer + weights).
    2) If model_path endswith .ckpt (Lightning), load state_dict and map to HF model initialized from base_model_name.
    """
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        return tok, mdl

    if model_path.endswith(".ckpt"):
        if base_model_name is None:
            raise ValueError("--base_model is required when loading a Lightning .ckpt")
        # load state dict
        ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # strip potential 'model.' prefix (from LightningModule)
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model."):]] = v
            else:
                new_state[k] = v
        tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
        missing, unexpected = mdl.load_state_dict(new_state, strict=False)
        if missing:
            print(f"[warn] Missing keys when loading ckpt: {missing[:5]}{' ...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading ckpt: {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")
        return tok, mdl

    # last resort: try to treat model_path as a HF model id on the hub
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    return tok, mdl

def load_jsonl(path: str) -> List[Dict[str,Any]]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            recs.append(json.loads(line))
    return recs

def evaluate(model, tokenizer, dataset: ContrastJsonDataset, batch_size: int = 64, device: Optional[str] = None, out_csv: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds, all_labels, all_probs = [], [], []
    meta_buf = {"premise_txt": [], "hyp_txt": [], "phenomenon": []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating on contrast set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            # confidence of predicted class
            conf = probs.max(dim=-1).values.cpu().tolist()
            all_probs.extend(conf)

            meta_buf["premise_txt"].extend(batch["meta"]["premise_txt"])
            meta_buf["hyp_txt"].extend(batch["meta"]["hyp_txt"])
            meta_buf["phenomenon"].extend(batch["meta"]["phenomenon"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float((all_preds == all_labels).mean())

    # per-phenomenon
    df = pd.DataFrame({
        "premise": meta_buf["premise_txt"],
        "hypothesis": meta_buf["hyp_txt"],
        "phenomenon": meta_buf["phenomenon"],
        "label": all_labels,
        "pred": all_preds,
        "correct": (all_preds == all_labels).astype(int),
        "confidence": all_probs,
        "label_str": [ID_TO_LABEL_STR.get(int(x), str(int(x))) for x in all_labels],
        "pred_str": [ID_TO_LABEL_STR.get(int(x), str(int(x))) for x in all_preds],
    })

    slice_tbl = None
    if df["phenomenon"].notna().any():
        slice_tbl = df.groupby("phenomenon")["correct"].mean().sort_values(ascending=False).to_frame()
        slice_tbl.rename(columns={"correct": "accuracy"}, inplace=True)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

    return accuracy, slice_tbl, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF dir OR Lightning .ckpt OR HF hub id")
    ap.add_argument("--base_model", default=None, help="HF base model id (required if model_path is .ckpt)")
    ap.add_argument("--contrast_path", required=True, help="Path to contrast JSONL")
    ap.add_argument("--hyp_field", default="hypothesis_contrast", help="JSON field for hypothesis")
    ap.add_argument("--label_field", default="label_contrast", help="JSON field for label")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_csv", default=None, help="Optional per-example CSV output")
    args = ap.parse_args()

    # load records
    recs = load_jsonl(args.contrast_path)

    # load model + tokenizer
    tok, model = try_load_model(args.model_path, args.base_model, num_labels=3)

    # dataset
    ds = ContrastJsonDataset(
        records=recs, tokenizer=tok,
        hyp_field=args.hyp_field, label_field=args.label_field,
        max_len=args.max_len
    )

    # eval
    acc, slice_tbl, df = evaluate(
        model, tok, ds, batch_size=args.batch, device=args.device, out_csv=args.out_csv
    )

    print("\n=== Robust Accuracy (overall) ===")
    print(f"{acc*100:.2f}%")

    if slice_tbl is not None:
        print("\n=== Per-phenomenon accuracy ===")
        print(slice_tbl.to_string(float_format=lambda x: f"{x*100:.2f}%"))
    else:
        print("\n(no 'phenomenon' field found — per-slice scores skipped)")

if __name__ == "__main__":
    main()
