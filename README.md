# TRACE

This repository contains the implementation of TRACE (Temporal Risk Assessment and Concept Editing). TRACE learns a discrete codebook from language-model activations, uses the learned concepts for streaming harmfulness detection, and steers activations along harmful-to-benign concept directions during generation.

This work was accepted at the [2026 Symposium on Model Accountability, Sustainability and Healthcare (SMASH)](https://smashcon.org/en/). The extended abstract is available [here](Ankur_Garg_SmashCon.pdf).

## Repository structure

```text
├── scripts/
│   ├── data/          # Dataset preparation
│   ├── activations/   # Activation extraction
│   ├── vq/            # Codebook training
│   ├── detection/     # Detector training and evaluation
│   ├── steering/      # TRACE steering
│   ├── evaluation/    # HarmBench and refusal evaluation
│   └── analysis/      # Layer selection and concept audit
├── configs/           # Final detector configurations
├── src/               # Python implementation
├── tests/             # Offline test suite
└── requirements.txt
```

## Installation

We use Python 3.12 and CUDA-enabled PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The reported experiments used PyTorch 2.12.0, Transformers 5.9.0, and FAISS 1.8.0. Small numerical differences may occur across hardware and FAISS versions.

## Data

Datasets and model weights are not included in this repository.

WildGuard and S-Eval prompt-response pairs are taken from [StreamGuardBench](https://huggingface.co/datasets/Alibaba-AAIG/StreamGuardBench). WildGuard training also uses prompt annotations from [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix). Request access to WildGuardMix and authenticate with Hugging Face before running the taxonomy step.

Set the model or dataset at the top of each script and run:

```bash
bash scripts/data/prepare_wildguard.sh
bash scripts/data/add_wildguard_taxonomy.sh
bash scripts/data/build_detector_training_set.sh
bash scripts/data/prepare_s_eval.sh
```

The supported model names are `qwen3_8b`, `llama_3_1_8b_instruct`, and `internlm3_8_instruct`. Prepared data are saved under `data/`.

## Models

Download the following Hugging Face models and place them under `models/`:

- `Qwen/Qwen3-8B` as `models/Qwen3-8B`
- `meta-llama/Llama-3.1-8B-Instruct` as `models/Llama-3.1-8B-Instruct`
- `internlm/internlm3-8b-instruct` as `models/InternLM3-8B-Instruct`
- `cais/HarmBench-Llama-2-13b-cls` as `models/HarmBench-Llama-2-13b-cls`

## Configuration

The shell scripts contain the experiment settings at the top of each file. They are set to Qwen/WildGuard by default. Use the following dataset names and layer pairs for the other experiments.

| Generator | Dataset | Training split | Read / target layer | Steering strength |
|---|---|---|---:|---:|
| Qwen3-8B | `wildguard_qwen3_8b` | `classifier_train` | 20 / 24 | 0.75 |
| Qwen3-8B | `s_eval_qwen3_8b` | `train` | 20 / 24 | 0.50 |
| Llama-3.1-8B-Instruct | `wildguard_llama_3_1_8b_instruct` | `classifier_train` | 14 / 18 | 0.50 |
| Llama-3.1-8B-Instruct | `s_eval_llama_3_1_8b_instruct` | `train` | 14 / 18 | 0.75 |
| InternLM3-8B-Instruct | `wildguard_internlm3_8b_instruct` | `classifier_train` | 34 / 38 | 0.50 |
| InternLM3-8B-Instruct | `s_eval_internlm3_8b_instruct` | `train` | 34 / 38 | 0.50 |

The exact detector settings used in the paper are in `configs/detection/`. Select the configuration and
seed at the top of `scripts/detection/train_sequence_classifier.sh`. Seeds 43 and 44 reuse the same
configuration as seed 42.

## Running TRACE

Run all commands from the repository root. Edit the variables at the top of each shell script before changing the generator or dataset.

```bash
# Cache training and test activations
bash scripts/activations/cache_activations.sh

# Train the codebook
bash scripts/vq/train_vq.sh

# Search detector hyperparameters (optional)
bash scripts/detection/search_sequence_classifier.sh

# Train the final detector with the selected YAML configuration
bash scripts/detection/train_sequence_classifier.sh

# Evaluate detection
bash scripts/detection/evaluate_sequence_classifier.sh

# Generate steered responses and evaluate them
bash scripts/steering/validate_steering.sh
bash scripts/steering/run_steering.sh
bash scripts/steering/summarize_steering.sh
bash scripts/evaluation/harmbench_judge.sh
bash scripts/evaluation/refusal.sh
```

Outputs are saved under `output/runs/`. Activation caches are saved under `output/activations/`.
The full activation caches can exceed 100 GB for one generator-dataset pair. The reported runs used a
40 GB GPU for activation caching and VQ training, a 10 GB GPU for detector training, and up to 200 GB of
host memory for VQ training.
The steering validation script evaluates strengths 0.25, 0.5, 0.75, and 1.0 on 200 harmful and 200 safe
responses from the held-out validation split. Point the summary and HarmBench scripts to the validation
run before setting the selected strength in `run_steering.sh`.

## Layer selection and concept audit

Layer probes can be run with:

```bash
bash scripts/analysis/select_layer.sh
```

The WildGuard probes use 3,970 Qwen, 3,084 Llama, and 3,169 InternLM responses per class. Qwen and Llama
use a 512-token cap, while InternLM uses 2,048 tokens. Layer selection uses seed 42. The Qwen and Llama
2,048-token checks use seeds 42--44. Repeating the probes on S-Eval training data checks whether the
selected layers remain predictive; it does not change the selected layer. The S-Eval checks use 3,499
Qwen, 3,079 Llama, and 2,611 InternLM responses per class.

The Qwen/WildGuard semantic audit uses Gemini to label sampled concept contexts. Set `GEMINI_API_KEY`, edit the paths in the analysis scripts, and run:

```bash
bash scripts/analysis/judge_codebook_concepts.sh
bash scripts/analysis/analyze_codebook_concepts.sh
bash scripts/analysis/concept_audit.sh
```

The paper uses three independently sampled labeling runs with seeds 42, 43, and 44, followed by semantic agreement across the three runs.

## Development

The repository is a flat source tree that is put on the path with `PYTHONPATH=src` by the shell
scripts. `pyproject.toml` declares the same layout for packaging and for pytest, so the offline test
suite runs with:

```bash
python -m pytest
```

The suite covers the codebook statistics, the vector quantizer and cross-layer VQ-VAE, the activation
cache loader, dataset splits, prompt/activation batching, the training diagnostics, and the refusal
rule. It needs only CPU PyTorch, NumPy, and pytest: FAISS, `datasets`, and `matplotlib` are imported
on demand and are not required to run it.

### Code-score methods

`--region-score-method` selects how a code's harmfulness is estimated from the training partition:

| Method | Meaning |
|---|---|
| `response_presence` (default) | each code counts once per response that fires it |
| `response_frequency` | each response contributes unit total mass, split across its codes |
| `token_occurrence` | every code occurrence inherits its response label |

Each estimator reports `signed_harmfulness`, the enrichment of a code's harmful rate over the
partition's base rate, normalized so that `+1` and `-1` are the extremes. Codes scoring above zero
form the harmful region and the rest form the benign region. When the partition contains only harmful
(or only safe) responses, every code scores `0` rather than `NaN`, and an unknown method name raises
instead of silently falling back to a different estimator.

## Citation

If you find this work useful or use it in your research, please cite:

```bibtex
@misc{garg2026trace,
  title  = {{TRACE}: Temporal Risk Assessment and Concept Editing for Safer Language Generation},
  author = {Garg, Ankur and Yu, Xuemin and Sajjad, Hassan and Ebrahimi Kahou, Samira},
  year   = {2026},
  note   = {Accepted at the Symposium on Model Accountability, Sustainability and Healthcare (SMASH 2026)},
  url    = {https://github.com/agarg-dev/TRACE}
}
```
