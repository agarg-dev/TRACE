#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

config=configs/detection/qwen_wildguard.yaml
seed=42

python src/detection/train_sequence_classifier.py \
    --config $config \
    --seed $seed
