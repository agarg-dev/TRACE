#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
train_set=classifier_train
base_model=Qwen3-8B
samples_per_class=3970
max_response_tokens=512
target_offset=4
folds=5
seed=42
output_dir=output/runs/detection/layer_probe_qwen

python src/analysis/select_layer.py \
    --dataset $dataset \
    --train-set $train_set \
    --base-model $base_model \
    --samples-per-class $samples_per_class \
    --max-response-tokens $max_response_tokens \
    --target-offset $target_offset \
    --folds $folds \
    --seed $seed \
    --out $output_dir
