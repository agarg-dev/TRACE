#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

steering_run=output/runs/steering/qwen_wildguard
labels=unsafe
batch_size=8

python src/evaluation/harmbench_judge.py \
    --run $steering_run \
    --labels $labels \
    --batch-size $batch_size
