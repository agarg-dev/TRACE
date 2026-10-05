#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
train_set=classifier_train
activation_cache="output/activations/paper/$dataset/$train_set"
activation_layer=20
vq_run=output/runs/vq/qwen_wildguard
vq_checkpoint=model_joint.pt
out=output/runs/detection/qwen_wildguard_search
input_representation=hybrid
code_vector_source=activation_mean
raw_projection_dim=256
vq_projection_dim=256
objective=mean_f1
trials=100
startup_trials=12
epochs=30
batch_size=16
code_batch_size=16
patience=6
seed=42

python src/detection/search_sequence_classifier.py \
    --out $out \
    --dataset $dataset \
    --train-set $train_set \
    --activation-cache $activation_cache \
    --activation-layer $activation_layer \
    --vq-run $vq_run \
    --vq-checkpoint $vq_checkpoint \
    --input-representation $input_representation \
    --code-vector-source $code_vector_source \
    --raw-projection-dim $raw_projection_dim \
    --vq-projection-dim $vq_projection_dim \
    --objective $objective \
    --trials $trials \
    --startup-trials $startup_trials \
    --epochs $epochs \
    --batch-size $batch_size \
    --code-batch-size $code_batch_size \
    --patience $patience \
    --seed $seed
