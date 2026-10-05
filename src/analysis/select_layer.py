#!/usr/bin/env python
"""Select a read layer with probes on final response-token activations.

Probes use balanced training responses and five-fold cross-validation.
"""

import argparse
import json
import time

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer

from data.dataset_splits import training_split
from model_inputs import (
    configure_transformers_compatibility, format_prompt_token_ids, initialize_internlm3_rotary_embeddings,
)
from project_config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET,
    DETECTION_RUNS_DIR,
    base_model_path,
    resolve_project_path,
)


def balanced_sample(rows, samples_per_class, seed):
    """Choose the same number of non-empty safe and harmful responses deterministically."""
    groups = {
        label: [row for row in rows if row["label"] == label and row["response"].strip()]
        for label in (0, 1)
    }
    for label, group in groups.items():
        if len(group) < samples_per_class:
            raise ValueError(
                f"requested {samples_per_class} examples with label {label}, but the split has {len(group)}"
            )

    random_state = np.random.RandomState(seed)
    sampled = []
    for label in (0, 1):
        indices = random_state.permutation(len(groups[label]))[:samples_per_class]
        sampled.extend(groups[label][index] for index in indices)
    order = random_state.permutation(len(sampled))
    return [sampled[index] for index in order]


def candidate_hidden_state_indices(num_hidden_layers, requested_layers=None, target_offset=4):
    """Return residual boundaries compatible with TRACE's fixed reconstruction offset."""
    if target_offset < 1 or target_offset >= num_hidden_layers:
        raise ValueError(f"target offset must be in [1, {num_hidden_layers - 1}]")
    last_candidate = num_hidden_layers - target_offset
    layers = (
        list(range(1, last_candidate + 1))
        if requested_layers is None
        else list(dict.fromkeys(requested_layers))
    )
    invalid = [layer for layer in layers if layer < 1 or layer > last_candidate]
    if invalid:
        raise ValueError(
            f"read layers must be in [1, {last_candidate}] so their layer + {target_offset} "
            f"reconstruction target exists; got {invalid}"
        )
    return layers


def extract_final_activations(
    rows,
    tokenizer,
    model,
    layers,
    max_response_tokens,
    id_key,
):
    """Extract ``[response, layer, hidden]`` activations at the final retained token."""
    hidden_size = int(model.config.hidden_size)
    # The model computes bfloat16 states. Float32 retains those values exactly for sklearn;
    # converting to NumPy float16 would introduce a second, unnecessary quantization.
    activations = np.zeros((len(rows), len(layers), hidden_size), dtype=np.float32)
    response_lengths = np.zeros(len(rows), dtype=np.int32)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    example_ids = np.asarray([int(row[id_key]) for row in rows], dtype=np.int64)
    device = next(model.parameters()).device

    for example_index, row in enumerate(rows):
        prompt_token_ids = format_prompt_token_ids(tokenizer, row["prompt"])
        response_token_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        response_token_ids = response_token_ids[:max_response_tokens]
        if not response_token_ids:
            raise ValueError(f"sampled response {row[id_key]} became empty after tokenization")

        final_position = len(prompt_token_ids) + len(response_token_ids) - 1
        input_ids = torch.tensor([prompt_token_ids + response_token_ids], device=device)
        with torch.inference_mode():
            hidden_states = model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states

        final_tensor = torch.stack([hidden_states[layer][0, final_position, :] for layer in layers], dim=0)
        activations[example_index] = final_tensor.float().cpu().numpy()
        response_lengths[example_index] = len(response_token_ids)

        if (example_index + 1) % 25 == 0 or example_index + 1 == len(rows):
            print(
                f"  extracted {example_index + 1:>4}/{len(rows)} responses "
                f"(last retained length {len(response_token_ids):>4})",
                flush=True,
            )

    return activations, labels, example_ids, response_lengths


def _binary_metrics(labels, scores):
    predictions = np.asarray(scores) >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def evaluate_layers(activations, labels, layers, folds, seed):
    """Run response-grouped cross-validation with the exact same probe at every layer."""
    if folds < 2 or min(np.bincount(labels, minlength=2)) < folds:
        raise ValueError("each class must contain at least one example per cross-validation fold")
    split_indices = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(labels, labels))
    results = []

    for layer_offset, layer in enumerate(layers):
        final_activations = activations[:, layer_offset].astype(np.float32)
        fold_results = []
        for fold, (train_indices, validation_indices) in enumerate(split_indices):
            # This intentionally matches ITI's simple unscaled sklearn logistic probe.
            probe = LogisticRegression(random_state=seed, max_iter=1000)
            probe.fit(final_activations[train_indices], labels[train_indices])
            response_scores = probe.predict_proba(final_activations[validation_indices])[:, 1]
            response_metrics = _binary_metrics(labels[validation_indices], response_scores)
            fold_results.append({"fold": fold, "response": response_metrics})

        summary = {
            metric: {
                "mean": float(np.mean([fold_result["response"][metric] for fold_result in fold_results])),
                "std": float(np.std([fold_result["response"][metric] for fold_result in fold_results])),
            }
            for metric in fold_results[0]["response"]
        }
        results.append({"layer": layer, "response": summary, "folds": fold_results})
        print(
            f"  layer {layer:>2}: accuracy {summary['accuracy']['mean']:.3f} "
            f"| F1 {summary['f1']['mean']:.3f}",
            flush=True,
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-set", default="classifier_train")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--samples-per-class", type=int, default=3970)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    parser.add_argument("--target-offset", type=int, default=4)
    parser.add_argument("--layers", type=int, nargs="+", help="default: every layer compatible with target offset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="run directory under output/runs/detection by default")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("layer selection requires a GPU compute node")

    training_data = training_split(args.train_set, args.dataset)
    with training_data.source.open() as source_file:
        rows = [json.loads(line) for line in source_file]
    sampled_rows = balanced_sample(
        rows, args.samples_per_class, args.seed
    )
    model_path = base_model_path(args.base_model)
    configure_transformers_compatibility()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    initialize_internlm3_rotary_embeddings(model)
    layers = candidate_hidden_state_indices(model.config.num_hidden_layers, args.layers, args.target_offset)
    run_dir = (
        resolve_project_path(args.out)
        if args.out
        else DETECTION_RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_layer_selection_{args.dataset}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nTRACE LAYER SELECTION", flush=True)
    print(f"  model          : {args.base_model} ({model.config.num_hidden_layers} transformer blocks)", flush=True)
    print(
        f"  calibration    : {args.dataset}/{training_data.name}; official test data is not used",
        flush=True,
    )
    print(f"  balanced sample: {args.samples_per_class} safe + {args.samples_per_class} harmful", flush=True)
    candidate_description = (
        f"hidden_states[{layers[0]}] ... hidden_states[{layers[-1]}]"
        if layers == list(range(layers[0], layers[-1] + 1))
        else ", ".join(f"hidden_states[{layer}]" for layer in layers)
    )
    print(f"  candidates     : {candidate_description}", flush=True)
    print(f"  VQ target      : hidden_states[L + {args.target_offset}]", flush=True)
    print(f"  response cap   : {args.max_response_tokens} tokens", flush=True)
    print(f"  validation     : {args.folds}-fold response-grouped stratified cross-validation", flush=True)

    started_at = time.time()
    activations, labels, example_ids, response_lengths = extract_final_activations(
        sampled_rows, tokenizer, model, layers, args.max_response_tokens, training_data.id_key
    )
    del model
    torch.cuda.empty_cache()
    layer_results = evaluate_layers(activations, labels, layers, args.folds, args.seed)
    ranked_layers = sorted(
        layer_results, key=lambda result: (-result["response"]["accuracy"]["mean"], result["layer"])
    )
    selected = ranked_layers[0]
    result = {
        "method": "final_token_layerwise_linear_probing",
        "config": {
            "dataset": args.dataset,
            "train_set": training_data.name,
            "source": str(training_data.source),
            "base_model": args.base_model,
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "sample_size": len(sampled_rows),
            "max_response_tokens": args.max_response_tokens,
            "target_offset": args.target_offset,
            "folds": args.folds,
            "candidate_hidden_state_indices": layers,
            "probe": "sklearn.linear_model.LogisticRegression(random_state=seed, max_iter=1000)",
            "probe_training_position": "final retained response token after the response-token cap",
            "classification_threshold": 0.5,
            "selection_objective": "mean held-out response accuracy over five stratified folds",
            "sklearn_version": sklearn.__version__,
        },
        "layer_convention": {
            "hidden_state_index": "hidden_states[L] is the output of transformer block L-1 and input to block L",
            "steering_hook_index": "L-1",
            "candidate_rule": (
                "exclude embeddings and require hidden_states[L + target_offset] to exist for TRACE's "
                "four-block reconstruction horizon"
            ),
        },
        "sample_summary": {
            "safe": int((labels == 0).sum()),
            "harmful": int((labels == 1).sum()),
            "minimum_response_tokens": int(response_lengths.min()),
            "median_response_tokens": float(np.median(response_lengths)),
            "maximum_response_tokens": int(response_lengths.max()),
            "minimum_example_id": int(example_ids.min()),
            "maximum_example_id": int(example_ids.max()),
        },
        "selection": {
            "hidden_state_index": selected["layer"],
            "steering_hook_index": selected["layer"] - 1,
            "reconstruction_target_hidden_state_index": selected["layer"] + args.target_offset,
            "response_accuracy_mean": selected["response"]["accuracy"]["mean"],
            "response_accuracy_std": selected["response"]["accuracy"]["std"],
            "response_f1_mean": selected["response"]["f1"]["mean"],
            "margin_over_runner_up": (
                selected["response"]["accuracy"]["mean"]
                - ranked_layers[1]["response"]["accuracy"]["mean"]
                if len(ranked_layers) > 1 else None
            ),
        },
        "layers": layer_results,
        "elapsed_seconds": time.time() - started_at,
    }
    output_path = run_dir / "layer_selection.json"
    with output_path.open("w") as output_file:
        json.dump(result, output_file, indent=2)

    print("\nSELECTED SHARED LAYER", flush=True)
    print(f"  hidden state   : hidden_states[{selected['layer']}]")
    print(f"  steering hook  : transformer block {selected['layer'] - 1} output")
    print(f"  VQ target      : hidden_states[{selected['layer'] + args.target_offset}]")
    print(f"  accuracy       : {selected['response']['accuracy']['mean']:.3f}")
    print(f"  response F1    : {selected['response']['f1']['mean']:.3f}")
    print(f"  report         : {output_path}", flush=True)


if __name__ == "__main__":
    main()
