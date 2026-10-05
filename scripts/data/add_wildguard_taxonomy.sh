#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
source=data/$dataset/raw
dataset_dir=data/$dataset

python src/data/add_wildguard_taxonomy.py \
    --source $source \
    --dataset-dir $dataset_dir
