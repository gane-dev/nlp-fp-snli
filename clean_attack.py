#!/usr/bin/env python3
# Convert TextAttack raw JSONL (with combined "Premise/Hypothesis" and [[markers]])
# into SNLI-style JSONL: {"premise": "...", "hypothesis": "...", "label": 0/1/2}

import argparse, json, re, sys
from pathlib import Path

PREM_MARK = r"\[\[\[\[Premise\]\]\]\]\s*:\s*"
HYP_MARK  = r"\[\[\[\[Hypothesis\]\]\]\]\s*:\s*"
SPLIT_MARK = r"<SPLIT>"

# Compile for speed
PREM_RE = re.compile(PREM_MARK, flags=re.IGNORECASE)
HYP_RE  = re.compile(HYP_MARK,  flags=re.IGNORECASE)
SPLIT_RE = re.compile(SPLIT_MARK, flags=re.IGNORECASE)

# [[token]] -> token
INLINE_BRACKETS_RE = re.compile(r"\[\[(.*?)\]\]")

def strip_inline_brackets(text: str) -> str:
    return INLINE_BRACKETS_RE.sub(r"\1", text)

def normalize_space(s: str) -> str:
    return " ".join(s.split())

def parse_combined_field(combined: str) -> tuple[str, str] | tuple[None, None]:
    """
    Handle formats like:
      '[[[[Premise]]]]: ... <SPLIT>[[[[Hypothesis]]]]: ...'
    Fallbacks:
      'premise [SEP] hypothesis'
    Returns (premise, hypothesis) or (None, None) if parse fails.
    """
    if not isinstance(combined, str) or not combined.strip():
        return None, None

    txt = combined.strip()

    # Case A: explicit Premise/Hypothesis markers with <SPLIT>
    if PREM_RE.search(txt) and HYP_RE.search(txt):
        # split once on <SPLIT>
        parts = SPLIT_RE.split(txt, maxsplit=1)
        if len(parts) == 2:
            left, right = parts
            prem = PREM_RE.sub("", left).strip()
            hyp  = HYP_RE.sub("", right).strip()
            return prem, hyp

    # Case B: use [SEP] delimiter
    if "[SEP]" in txt:
        prem, hyp = txt.split("[SEP]", 1)
        return prem.strip(), hyp.strip()

    # As a last resort: try to guess by 'Hypothesis' marker alone
    if HYP_RE.search(txt):
        bits = HYP_RE.split(txt, maxsplit=1)
        prem = PREM_RE.sub("", bits[0]).strip()
        hyp  = bits[1].strip()
        return prem, hyp

    return None, None

def label_to_int(y):
    if y is None:
        return None
    if isinstance(y, (int,)) or (isinstance(y, str) and y.isdigit()):
        i = int(y)
        if i in (0,1,2): return i
    if isinstance(y, str):
        s = y.strip().lower()
        if "entail" in s: return 0
        if "neutral" in s or s == "1": return 1
        if "contradict" in s or s == "2": return 2
    return None

def process_record(raw: dict) -> dict | None:
    """
    Expects keys like:
      - "hypothesis_adv": combined string with markers  (from TextAttack export)
      - optional: "premise", "orig_hypothesis" (may be null)
      - "label_raw" or "label"
    Produces:
      {"premise": "...", "hypothesis": "...", "label": 0/1/2}
    """
    combined = raw.get("hypothesis_adv")
    prem = raw.get("premise")
    orig_hyp = raw.get("orig_hypothesis")

    # If premise/hyp not present, try to recover from combined field
    if (prem is None or not str(prem).strip()) or (orig_hyp is None or not str(orig_hyp).strip()):
        p2, h2 = parse_combined_field(combined)
        if p2 and h2:
            prem_from_combined = strip_inline_brackets(p2)
            hyp_from_combined  = strip_inline_brackets(h2)
            prem = prem or prem_from_combined
            # For training we want the *adversarial* hypothesis as input,
            # not the original; but we still use parsed original if needed.
            # The adv hypothesis is the combined right side, so use that.
            adv_hyp = hyp_from_combined
        else:
            # No way to recover
            return None
    else:
        # We have prem + orig_hyp fields; adversarial is in combined
        p2, h2 = parse_combined_field(combined)
        adv_hyp = strip_inline_brackets(h2) if h2 else None
        prem = prem.strip() if isinstance(prem, str) else prem

    if not prem or not adv_hyp:
        return None

    prem = normalize_space(prem)
    adv_hyp = normalize_space(adv_hyp)

    # Map label
    lab = label_to_int(raw.get("label_raw", raw.get("label")))
    if lab is None:
        return None

    return {"premise": prem, "hypothesis": adv_hyp, "label": lab}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Path to raw TextAttack JSONL")
    ap.add_argument("--out_jsonl", required=True, help="Path to write cleaned SNLI-style JSONL")
    args = ap.parse_args()

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = 0, 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fi, open(args.out_jsonl, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip(): 
                continue
            raw = json.loads(line)
            rec = process_record(raw)
            if rec is None:
                dropped += 1
                continue
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. Kept {kept}, dropped {dropped}. Wrote {args.out_jsonl}")

if __name__ == "__main__":
    main()
