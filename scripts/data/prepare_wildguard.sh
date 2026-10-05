#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

model=qwen3_8b
revision=main

python src/data/prepare_wildguard.py \
    --model $model \
    --revision $revision
