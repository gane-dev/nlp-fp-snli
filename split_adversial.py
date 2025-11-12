import json, random
from pathlib import Path
infile = "data/snli_textfooler_raw.jsonl"
all_recs = [json.loads(l) for l in open(infile)]
random.seed(42)
random.shuffle(all_recs)
cut = int(len(all_recs)*0.8)
train_adv = all_recs[:cut]
held_adv  = all_recs[cut:]

Path("data/adversarial/train").mkdir(parents=True, exist_ok=True)
with open("data/adversarial/train/snli_adv_train.jsonl","w") as fo:
    for r in train_adv: fo.write(json.dumps(r)+"\n")
with open("data/adversarial/train/snli_adv_held.jsonl","w") as fo:
    for r in held_adv: fo.write(json.dumps(r)+"\n")
