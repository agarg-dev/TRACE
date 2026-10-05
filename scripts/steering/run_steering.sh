#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
base_model=Qwen3-8B
train_set=classifier_train
test_set=test
activation_cache=output/activations/paper/$dataset/$train_set
checkpoint=output/runs/vq/qwen_wildguard/model_joint.pt
output_dir=output/runs/steering/qwen_wildguard
audit_cache=output/runs/analysis/steering_cache
method=additive_renorm_gate
strength=0.75
max_new_tokens=2048
batch_size=8

python src/steering/run_steering.py \
    --out $output_dir \
    --dataset $dataset \
    --train-set $train_set \
    --test-set $test_set \
    --base-model $base_model \
    --activation-cache $activation_cache \
    --checkpoint $checkpoint \
    --methods $method \
    --lambdas $strength \
    --max-new-tokens $max_new_tokens \
    --batch-size $batch_size \
    --audit-cache-out $audit_cache
