#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
seed=42

python src/data/build_detector_training_set.py \
    --dataset $dataset \
    --seed $seed
