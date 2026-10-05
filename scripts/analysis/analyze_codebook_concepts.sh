#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

run_42=output/runs/analysis/concepts_42
run_43=output/runs/analysis/concepts_43
run_44=output/runs/analysis/concepts_44
consensus=output/runs/analysis/concept_consensus

python src/analysis/analyze_codebook_concepts.py \
    --consensus-runs $run_42 $run_43 $run_44 \
    --output $consensus
