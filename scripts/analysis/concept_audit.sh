#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

concepts=output/runs/analysis/concept_consensus/consensus.json
detection_cache=output/runs/analysis/detection_cache
steering_cache=output/runs/analysis/steering_cache
detection_output=output/runs/analysis/detection_audit
steering_output=output/runs/analysis/steering_audit
threshold=argmax
result_key=euclid_additive_renorm_gate_lam0.75

python src/analysis/concept_audit.py detection \
    --cache $detection_cache \
    --concepts $concepts \
    --threshold $threshold \
    --out $detection_output

python src/analysis/concept_audit.py steering \
    --cache $steering_cache \
    --concepts $concepts \
    --result-key $result_key \
    --out $steering_output
