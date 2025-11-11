python eval_contrast_snli.py \
  --model_path checkpoints/electra-snli/best \
  --contrast_path data/contrast_sets/snli_combined.jsonl \
  --out_csv reports/contrast_eval.csv
# Evaluate original pairs instead of contrast
python eval_contrast_snli.py \
  --model_path checkpoints/electra-snli/best \
  --contrast_path data/contrast_sets/snli_manual.jsonl \
  --hyp_field hypothesis_orig \
  --label_field label_orig