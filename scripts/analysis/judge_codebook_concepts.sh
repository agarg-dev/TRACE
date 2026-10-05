#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

checkpoint=output/runs/vq/qwen_wildguard/model_joint.pt
activation_cache=output/activations/paper/wildguard_qwen3_8b/classifier_train
model_path=models/Qwen3-8B
run_42=output/runs/analysis/concepts_42
run_43=output/runs/analysis/concepts_43
run_44=output/runs/analysis/concepts_44
consensus=output/runs/analysis/concept_consensus
judge_model=gemini-3.1-flash-lite
seed_1=42
seed_2=43
seed_3=44

python src/analysis/judge_codebook_concepts.py prepare \
    --checkpoint $checkpoint \
    --activation-cache $activation_cache \
    --model-path $model_path \
    --output $run_42 \
    --seed $seed_1 \
    --judge-model $judge_model

python src/analysis/judge_codebook_concepts.py judge \
    --output $run_42 \
    --judge-model $judge_model

python src/analysis/analyze_codebook_concepts.py --run $run_42

python src/analysis/judge_codebook_concepts.py prepare \
    --checkpoint $checkpoint \
    --activation-cache $activation_cache \
    --model-path $model_path \
    --output $run_43 \
    --seed $seed_2 \
    --judge-model $judge_model

python src/analysis/judge_codebook_concepts.py judge \
    --output $run_43 \
    --judge-model $judge_model

python src/analysis/analyze_codebook_concepts.py --run $run_43

python src/analysis/judge_codebook_concepts.py prepare \
    --checkpoint $checkpoint \
    --activation-cache $activation_cache \
    --model-path $model_path \
    --output $run_44 \
    --seed $seed_3 \
    --judge-model $judge_model

python src/analysis/judge_codebook_concepts.py judge \
    --output $run_44 \
    --judge-model $judge_model

python src/analysis/analyze_codebook_concepts.py --run $run_44

python src/analysis/judge_codebook_concepts.py prepare-agreement \
    --runs $run_42 $run_43 $run_44 \
    --output $consensus \
    --judge-model $judge_model

python src/analysis/judge_codebook_concepts.py judge-agreement \
    --output $consensus \
    --judge-model $judge_model
