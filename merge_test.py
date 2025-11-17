import json
import math
import random

from datasets import Dataset, concatenate_datasets


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

if __name__ == "__main__":
    # Example usage
    from datasets import load_dataset

    snli = load_dataset("snli")
    orig_train = snli["train"].filter(lambda x: x["label"] != -1)

    adv_ds = load_clean_adv("data/adversarial/train/snli_adv_train.jsonl")

    mixed_train = build_mixed_train(orig_train, adv_ds, adv_ratio=0.3, seed=42)

    print(f"Original train size: {len(orig_train)}")
    print(f"Adversarial train size: {len(adv_ds)}")
    print(f"Mixed train size: {len(mixed_train)}")