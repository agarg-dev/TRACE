#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
base_model=Qwen3-8B
train_set=classifier_train
read_layer=20
target_layer=24
activation_cache=output/activations/paper/$dataset/$train_set
output_dir=output/runs/vq/qwen_wildguard
num_codes=800
nhead=16
epochs=100
batch_size=16
grad_accum=4
patience=0
max_grad_norm=1
learning_rate=3e-4
weight_decay=1e-4
warmup_fraction=0.05
learning_rate_floor=0.1
region_loss_weight=5
code_score_prior_strength=10
seed=42

python src/vq/train_vq.py \
    --out $output_dir \
    --dataset $dataset \
    --train-set $train_set \
    --activation-cache $activation_cache \
    --target-activation-cache $activation_cache \
    --base-model $base_model \
    --read-layer $read_layer \
    --target-layer $target_layer \
    --num-codes $num_codes \
    --nhead $nhead \
    --epochs $epochs \
    --batch-size $batch_size \
    --grad-accum $grad_accum \
    --patience $patience \
    --max-grad-norm $max_grad_norm \
    --lr $learning_rate \
    --weight-decay $weight_decay \
    --warmup-frac $warmup_fraction \
    --lr-floor $learning_rate_floor \
    --task-weight $region_loss_weight \
    --code-score-prior-strength $code_score_prior_strength \
    --seed $seed
