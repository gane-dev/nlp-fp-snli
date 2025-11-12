from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

sbert = SentenceTransformer("all-MiniLM-L6-v2")
in_path = "snli_textfooler_raw.jsonl"
out_path = "snli_textfooler_filtered.jsonl"
threshold = 0.5

def label_to_int(l):
    if isinstance(l, str):
        s = l.lower()
        if "entail" in s: return 0
        if "neutral" in s: return 1
        if "contradict" in s: return 2
    try:
        return int(l)
    except:
        return None

with open(in_path) as fi, open(out_path, "w", encoding="utf-8") as fo:
    for line in tqdm(fi):
        r = json.loads(line)
        prem = r["premise"]
        orig_hyp = r.get("orig_hypothesis") or r.get("orig_hyp")
        adv = r["hypothesis_adv"]
        if not prem or not adv or not orig_hyp: 
            continue
        # compute similarity on the hypothesis alone, or on premise+hypothesis concatenation
        orig_text = orig_hyp
        adv_text = adv
        sim = float((sbert.encode([orig_text, adv_text], convert_to_numpy=True)))
        # above is wrong: we need cosine, so compute properly:
        emb_o = sbert.encode(orig_text, convert_to_numpy=True)
        emb_a = sbert.encode(adv_text, convert_to_numpy=True)
        import numpy as np
        cos = float(np.dot(emb_o, emb_a) / (np.linalg.norm(emb_o) * np.linalg.norm(emb_a)))
        if cos >= threshold:
            lab = label_to_int(r.get("label_raw", r.get("label")))
            if lab is None: continue
            out = {"premise": prem, "hypothesis": adv, "label": lab, "orig_hypothesis": orig_hyp, "sbert_cos": cos}
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
