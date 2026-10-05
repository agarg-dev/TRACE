#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

steering_run=output/runs/steering/qwen_wildguard
repetition_floor=0.5
max_nll_increase=1
max_edited_fraction=0.9
suppression_threshold=0.5

python src/steering/summarize_steering.py \
    --run $steering_run \
    --repetition-floor $repetition_floor \
    --max-nll-increase $max_nll_increase \
    --max-steered-fraction $max_edited_fraction \
    --supp $suppression_threshold
