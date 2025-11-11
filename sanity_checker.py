import json, collections
path = "contrast_sets/snli_combined.jsonl"
labs, phens = [], []
for line in open(path, "r", encoding="utf-8"):
    r = json.loads(line)
    for k in ["premise","hypothesis_orig","hypothesis_contrast","label_orig","label_contrast","phenomenon"]:
        assert k in r, f"Missing {k}"
    labs.append(r["label_contrast"])
    phens.append(r["phenomenon"])
print("Label distribution:", collections.Counter(labs))
print("Phenomenon distribution:", collections.Counter(phens))
