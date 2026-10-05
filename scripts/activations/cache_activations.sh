#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
base_model=Qwen3-8B
train_set=classifier_train
test_set=test
read_layer=20
target_layer=24
cache_tag=paper

python src/activations/cache_activations.py \
    --dataset $dataset \
    --base-model $base_model \
    --data-set $train_set \
    --layers $read_layer $target_layer \
    --cache-tag $cache_tag

python src/activations/cache_activations.py \
    --dataset $dataset \
    --base-model $base_model \
    --data-set $test_set \
    --layers $read_layer \
    --cache-tag $cache_tag
