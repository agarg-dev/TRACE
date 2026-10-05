#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
base_model=Qwen3-8B
train_set=classifier_train
read_layer=20
target_layer=24
activation_cache=output/activations/paper/$dataset/$train_set
target_activation_cache=$activation_cache
output_dir=output/runs/vq/qwen_wildguard
num_codes=800
decoder_layers=4
nhead=16
feedforward_multiplier=1.5
dropout=0
epochs=100
batch_size=16
gradient_accumulation=4
patience=0
max_gradient_norm=1
learning_rate=3e-4
weight_decay=1e-4
warmup_fraction=0.05
learning_rate_floor=0.1
commitment_cost=0.1
perplexity_weight=0.01
region_loss_weight=5
code_score_prior_strength=10
seed=42

python src/vq/train_vq.py \
    --out $output_dir \
    --dataset $dataset \
    --train-set $train_set \
    --activation-cache $activation_cache \
    --target-activation-cache $target_activation_cache \
    --base-model $base_model \
    --read-layer $read_layer \
    --target-layer $target_layer \
    --num-codes $num_codes \
    --decoder-layers $decoder_layers \
    --nhead $nhead \
    --ff-mult $feedforward_multiplier \
    --dropout $dropout \
    --epochs $epochs \
    --batch-size $batch_size \
    --grad-accum $gradient_accumulation \
    --patience $patience \
    --max-grad-norm $max_gradient_norm \
    --lr $learning_rate \
    --weight-decay $weight_decay \
    --warmup-frac $warmup_fraction \
    --lr-floor $learning_rate_floor \
    --commitment-cost $commitment_cost \
    --perplexity-weight $perplexity_weight \
    --task-weight $region_loss_weight \
    --code-score-prior-strength $code_score_prior_strength \
    --seed $seed
