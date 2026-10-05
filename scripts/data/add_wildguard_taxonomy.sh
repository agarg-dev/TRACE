#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
source=data/$dataset/raw
dataset_dir=data/$dataset
taxonomy_revision=d29c47f41c8b51348b5c8e8c81c039b3132b66d1

python src/data/add_wildguard_taxonomy.py \
    --source $source \
    --dataset-dir $dataset_dir \
    --taxonomy-revision $taxonomy_revision
