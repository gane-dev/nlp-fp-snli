python eval_contrast_snli.py --model_path checkpoints/electra-snli/best --contrast_path data/contrast_sets/snli_combined.jsonl --out_csv reports/contrast_eval.csv
# Evaluate original pairs instead of contrast
python eval_contrast_snli.py \
  --model_path checkpoints/electra-snli/best \
  --contrast_path data/contrast_sets/snli_manual.jsonl \
  --hyp_field hypothesis_orig \
  --label_field label_orig


textattack attack --model-from-huggingface checkpoints/electra-snli/best --dataset-from-huggingface snli --split validation --recipe TextFoolerJin2019 --num-examples 1000 --log-to-csv attacks/textfooler_snli_val.csv --random-seed 42


textattack attack --model checkpoints/electra-snli/best --dataset-from-huggingface snli --dataset-split validation --recipe textfooler --num-examples 1000 --log-to-csv attacks/text
fooler_snli_val.csv --random-seed 42


textattack attack --model-from-huggingface google/electra-base-discriminator --dataset-from-huggingface snli --dataset-split validation --recipe textfooler --num-examples 1000 --log-to-csv attacks/textfooler_snli_val_base.csv --random-seed 42

python ta_csv_to_jsonl.py textfooler_snli_val.csv snli_textfooler_raw.jsonl