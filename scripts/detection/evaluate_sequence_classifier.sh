#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

dataset=wildguard_qwen3_8b
classifier_run=output/runs/detection/qwen_wildguard
classifier_checkpoint=classifier.pt
test_cache=output/activations/paper/$dataset/test
test_set=test
audit_cache=output/runs/analysis/detection_cache
batch_size=16
code_batch_size=16

python src/detection/evaluate_sequence_classifier.py \
    --classifier-run $classifier_run \
    --classifier-checkpoint $classifier_checkpoint \
    --test-dataset $dataset \
    --test-set $test_set \
    --activation-cache $test_cache \
    --batch-size $batch_size \
    --code-batch-size $code_batch_size \
    --audit-cache-out $audit_cache
