import csv, json, argparse
from pathlib import Path

def ta_csv_to_jsonl(csv_path, out_path, orig_field="original_text", adv_field="perturbed_text", label_field="ground_truth_output"):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # adapt keys based on your TextAttack CSV columns
            orig = r.get(orig_field) or r.get("original_text") or r.get("original")
            adv  = r.get(adv_field)  or r.get("perturbed_text") or r.get("perturbed")
            label = r.get(label_field) or r.get("ground_truth_output") or r.get("original_result")
            # if original stored as "premise [SEP] hypothesis" you must split
            # Here we assume TextAttack original_text is "premise [SEP] hypothesis"
            if orig and "[SEP]" in orig:
                prem, hyp = orig.split("[SEP]", 1)
                prem, hyp = prem.strip(), hyp.strip()
            else:
                # fallback: you may have separate columns - adjust accordingly
                prem, hyp = None, None
            rows.append({"premise": prem, "hypothesis_adv": adv, "label_raw": label, "orig_hypothesis": hyp})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import sys
    ta_csv_to_jsonl(sys.argv[1], sys.argv[2])
