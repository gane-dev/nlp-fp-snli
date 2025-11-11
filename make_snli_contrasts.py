import json, random, re
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from datasets import load_dataset
# NLTK WordNet for antonyms/synonyms
import nltk
nltk.download("wordnet"); nltk.download("omw-1.4")
from nltk.corpus import wordnet as wn

random.seed(42)
np.random.seed(42)

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

# ----------------------------
# Helpers
# ----------------------------
AUX_PAT = r"\b(is|are|was|were|am|has|have|had|can|could|will|would|should|must|do|does|did)\b"

def insert_not(h: str) -> str:
    """
    Insert 'not' after first auxiliary verb. If none, prefix 'It is not true that '.
    """
    def repl(m):
        return m.group(0) + " not"
    if re.search(AUX_PAT, h):
        return re.sub(AUX_PAT, repl, h, count=1)
    # no auxiliary: lightweight fallback
    return "It is not true that " + h

def first_antonym(word: str) -> str or None:
    ants = set()
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            for a in l.antonyms():
                ants.add(a.name().replace("_", " "))
    return next(iter(ants)) if ants else None

def token_swap_antonym(h: str) -> str or None:
    """
    Try to swap exactly one content word with its antonym.
    """
    toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z\s]", h)
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for i in idxs:
        w = toks[i]
        if not re.match(r"^[A-Za-z]", w):  # skip punctuation
            continue
        lower = w.lower()
        ant = first_antonym(lower)
        if ant and ant.lower() != lower:
            # preserve capitalization
            ant_surface = ant.capitalize() if w[0].isupper() else ant
            toks[i] = ant_surface
            return "".join([t if re.match(r"[^A-Za-z]", t) else (" " + t) for t in toks]).strip()
    return None

QUANT_PAIRS = [
    ("some", "all"),
    ("a few", "all"),
    ("at least one", "all"),
    ("many", "all"),
    ("some", "none"),
    ("at least one", "none")
]

def quantifier_edit(h: str) -> List[Tuple[str, str]]:
    """
    Return a list of (h_new, target_change) suggested edits.
    We try pairs so you can label the contrast deterministically later.
    """
    h_low = h.lower()
    outs = []
    for src, tgt in QUANT_PAIRS:
        if re.search(rf"\b{re.escape(src)}\b", h_low):
            h_new = re.sub(rf"\b{re.escape(src)}\b", tgt, h, flags=re.IGNORECASE, count=1)
            outs.append((h_new, f"{src}->{tgt}"))
    return outs

PARA_MAP = {
    # very light paraphrases to keep entailment likely
    "man": "person",
    "woman": "person",
    "kid": "child",
    "kids": "children",
    "bike": "bicycle",
    "photo": "photograph",
    "plane": "airplane",
    "tv": "television",
    "car": "automobile",
    "pic": "picture",
    "dog": "canine",
    "cat": "feline",
}

def light_paraphrase(h: str) -> str:
    def repl(m):
        w = m.group(0)
        lw = w.lower()
        rep = PARA_MAP.get(lw, lw)
        return rep.capitalize() if w[0].isupper() else rep
    pat = r"\b(" + "|".join(map(re.escape, PARA_MAP.keys())) + r")\b"
    return re.sub(pat, repl, h, flags=re.IGNORECASE)

def jsonl_write(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Build contrast set
# ----------------------------
def build():
    ds = load_dataset("snli")
    # filter invalid labels
    valid_dev = [ex for ex in ds["validation"] if ex["label"] in (0,1,2)]
    random.shuffle(valid_dev)

    negation, antonymy, quantifier, paraphrase = [], [], [], []

    # We will try to construct:
    # - Negation (target: flip E->C when possible)
    # - Antonymy (flip E->C via antonym swap)
    # - Quantifier (nudge N->C by overgeneralization 'some'->'all', or E->C if applicable)
    # - Paraphrase (keep E->E with light synonyms)

    for ex in valid_dev:
        p = ex["premise"]
        h = ex["hypothesis"]
        y = LABEL_MAP[ex["label"]]

        # NEGATION: Prefer examples where original is entailment; make contrast contradiction
        if len(negation) < 75 and y == "entailment":
            h_neg = insert_not(h)
            if h_neg != h and len(h_neg) < 256:
                negation.append({
                    "premise": p,
                    "hypothesis_orig": h,
                    "label_orig": y,
                    "hypothesis_contrast": h_neg,
                    "label_contrast": "contradiction",
                    "phenomenon": "negation",
                    "source": "auto"
                })
                continue

        # ANTONYMY: Prefer entailment; swap single word to antonym to become contradiction
        if len(antonymy) < 75 and y == "entailment":
            h_ant = token_swap_antonym(h)
            if h_ant and h_ant != h and len(h_ant) < 256:
                antonymy.append({
                    "premise": p,
                    "hypothesis_orig": h,
                    "label_orig": y,
                    "hypothesis_contrast": h_ant,
                    "label_contrast": "contradiction",
                    "phenomenon": "antonymy",
                    "source": "auto-wordnet"
                })
                continue

        # QUANTIFIER: Look for 'some/at least one/many/a few' and push to 'all' or 'none'
        if len(quantifier) < 75:
            edits = quantifier_edit(h)
            if edits:
                # pick one
                h_new, tag = random.choice(edits)
                # heuristic label: changing quantifier usually invalidates entailment; use contradiction
                quantifier.append({
                    "premise": p,
                    "hypothesis_orig": h,
                    "label_orig": y,
                    "hypothesis_contrast": h_new,
                    "label_contrast": "contradiction",
                    "phenomenon": "quantifier",
                    "source": f"auto-{tag}"
                })
                continue

        # PARAPHRASE: keep entailment likely -> use light synonyms; label stays the same if originally E
        if len(paraphrase) < 75 and y == "entailment":
            h_para = light_paraphrase(h)
            if h_para != h:
                paraphrase.append({
                    "premise": p,
                    "hypothesis_orig": h,
                    "label_orig": y,
                    "hypothesis_contrast": h_para,
                    "label_contrast": "entailment",  # intent: meaning preserved
                    "phenomenon": "paraphrase",
                    "source": "auto-lite"
                })
                continue

        # stop if we have enough
        if len(negation) >= 75 and len(antonymy) >= 75 and len(quantifier) >= 75 and len(paraphrase) >= 75:
            break

    # If any bucket short, top up via looser conditions
    def top_up(bucket, name, maker_fn, target_n=75):
        attempts = 0
        for ex in valid_dev:
            if len(bucket) >= target_n: break
            attempts += 1
            rec = maker_fn(ex)
            if rec: bucket.append(rec)
            if attempts > 100000: break

    def make_neg(ex):
        p,h,y = ex["premise"], ex["hypothesis"], LABEL_MAP[ex["label"]]
        h2 = insert_not(h)
        if h2 != h:
            return {"premise":p,"hypothesis_orig":h,"label_orig":y,
                    "hypothesis_contrast":h2,"label_contrast":"contradiction",
                    "phenomenon":"negation","source":"auto-topup"}
        return None

    def make_ant(ex):
        p,h,y = ex["premise"], ex["hypothesis"], LABEL_MAP[ex["label"]]
        h2 = token_swap_antonym(h)
        if h2 and h2 != h:
            return {"premise":p,"hypothesis_orig":h,"label_orig":y,
                    "hypothesis_contrast":h2,"label_contrast":"contradiction",
                    "phenomenon":"antonymy","source":"auto-topup"}
        return None

    def make_quant(ex):
        p,h,y = ex["premise"], ex["hypothesis"], LABEL_MAP[ex["label"]]
        edits = quantifier_edit(h)
        if edits:
            h2, tag = random.choice(edits)
            return {"premise":p,"hypothesis_orig":h,"label_orig":y,
                    "hypothesis_contrast":h2,"label_contrast":"contradiction",
                    "phenomenon":"quantifier","source":f"auto-topup-{tag}"}
        return None

    def make_para(ex):
        p,h,y = ex["premise"], ex["hypothesis"], LABEL_MAP[ex["label"]]
        h2 = light_paraphrase(h)
        if h2 != h:
            # keep original label (often entailment) to test invariance
            return {"premise":p,"hypothesis_orig":h,"label_orig":y,
                    "hypothesis_contrast":h2,"label_contrast":y,
                    "phenomenon":"paraphrase","source":"auto-topup"}
        return None

    top_up(negation,   "negation",   make_neg)
    top_up(antonymy,   "antonymy",   make_ant)
    top_up(quantifier, "quantifier", make_quant)
    top_up(paraphrase, "paraphrase", make_para)

    all_recs = negation + antonymy + quantifier + paraphrase
    random.shuffle(all_recs)
    all_recs = all_recs[:300]  # ensure exactly ~300

    print({ "negation": len(negation), "antonymy": len(antonymy), "quantifier": len(quantifier), "paraphrase": len(paraphrase), "total": len(all_recs) })

    # Write
    out_path = "contrast_sets/snli_combined.jsonl"
    import os
    os.makedirs("contrast_sets", exist_ok=True)
    jsonl_write(out_path, all_recs)
    print(f"Wrote {len(all_recs)} records to {out_path}")

if __name__ == "__main__":
    build()
