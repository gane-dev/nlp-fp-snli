import json, random
for r in random.sample(list(open("contrast_sets/snli_combined.jsonl")), 5):
    print(json.loads(r), "\n")
