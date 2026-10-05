#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
base_model=Qwen3-8B
train_set=classifier_train
activation_cache=output/activations/paper/$dataset/$train_set
checkpoint=output/runs/vq/qwen_wildguard/model_joint.pt
output_dir=output/runs/steering/qwen_wildguard_validation
method=additive_renorm_gate
strengths="0.25 0.5 0.75 1.0"
max_new_tokens=2048
batch_size=8

python src/steering/run_steering.py \
    --out $output_dir \
    --dataset $dataset \
    --train-set $train_set \
    --test-set outer_validation \
    --base-model $base_model \
    --activation-cache $activation_cache \
    --checkpoint $checkpoint \
    --methods $method \
    --lambdas $strengths \
    --max-new-tokens $max_new_tokens \
    --batch-size $batch_size
