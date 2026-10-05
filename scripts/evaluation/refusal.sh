#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

steering_run=output/runs/steering/qwen_wildguard
output=$steering_run/refusal.json

python src/evaluation/refusal.py \
    --run $steering_run \
    --output $output
