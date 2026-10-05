#!/usr/bin/env python
"""Evaluate a frozen VQ, raw-activation, or hybrid temporal classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.concept_audit import write_detection_cache
from activations.activation_cache import (
    load_activation_sequences,
    resolve_activation_cache,
)
from data.dataset_splits import evaluation_split
from project_config import DEFAULT_DATASET, resolve_project_path
from vq.model import load_vq_checkpoint

from detection.sequence_classifier import (
    CodeSequenceClassifier,
    HybridSequenceClassifier,
    RawActivationSequenceClassifier,
    assign_code_sequences,
    assign_hybrid_sequences,
    score_activation_sequences,
    score_code_sequences,
    score_hybrid_sequences,
    summarize_scores,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-run", required=True)
    parser.add_argument("--classifier-checkpoint", default="classifier.pt")
    parser.add_argument("--test-dataset", default=None,
                        help="dataset to evaluate (default: classifier training dataset)")
    parser.add_argument("--test-set", default="test")
    parser.add_argument("--activation-cache", default=None,
                        help="test activation cache")
    parser.add_argument("--vq-run", default=None,
                        help="override the VQ run recorded in the classifier checkpoint")
    parser.add_argument("--vq-checkpoint", default=None,
                        help="override the VQ checkpoint filename recorded during training")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--code-batch-size", type=int, default=16)
    parser.add_argument(
        "--audit-cache-out",
        help="optionally save token IDs, code IDs, and all detector scores for later analysis",
    )
    return parser.parse_args()


def streaming_detection_positions(scored, thresholds):
    """Record the first threshold crossing for every response without storing full trajectories."""
    rows = []
    for sequence, token_scores in zip(scored["sequences"], scored["token_scores"]):
        total_tokens = int(len(token_scores))
        crossings = {}
        for name, threshold in thresholds.items():
            triggered = np.flatnonzero(token_scores >= threshold)
            crossing_token = int(triggered[0] + 1) if len(triggered) else None
            crossings[name] = {
                "threshold": float(threshold),
                "first_crossing_token": crossing_token,
                "crossing_fraction": crossing_token / total_tokens if crossing_token is not None else None,
            }
        rows.append({
            "response_id": int(sequence["idx"]),
            "label": int(sequence["label"]),
            "total_tokens": total_tokens,
            "crossings": crossings,
        })
    return rows


def summarize_detection_positions(rows, threshold_name):
    """Summarize detection position over harmful responses that cross a frozen threshold."""
    harmful_rows = [row for row in rows if row["label"] == 1]
    triggered = [
        row["crossings"][threshold_name]
        for row in harmful_rows
        if row["crossings"][threshold_name]["first_crossing_token"] is not None
    ]
    tokens = [row["first_crossing_token"] for row in triggered]
    fractions = [row["crossing_fraction"] for row in triggered]
    return {
        "n_harmful": len(harmful_rows),
        "n_triggered": len(triggered),
        "detection_coverage": len(triggered) / len(harmful_rows) if harmful_rows else None,
        "mean_first_crossing_token": float(np.mean(tokens)) if tokens else None,
        "median_first_crossing_token": float(np.median(tokens)) if tokens else None,
        "mean_crossing_fraction": float(np.mean(fractions)) if fractions else None,
        "median_crossing_fraction": float(np.median(fractions)) if fractions else None,
    }


def main():
    args = parse_args()
    classifier_run = resolve_project_path(args.classifier_run)
    classifier_checkpoint_path = classifier_run / args.classifier_checkpoint
    classifier_checkpoint = torch.load(classifier_checkpoint_path, map_location="cpu")
    config = classifier_checkpoint["config"]
    harmfulness_score_weight = config.get(
        "harmfulness_score_weight", config.get("code_score_skip_scale", 0.0)
    )
    input_representation = config.get("input_representation", "vq")
    args.test_dataset = args.test_dataset or config.get("dataset", DEFAULT_DATASET)

    evaluation_data = evaluation_split(args.test_set, args.test_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nTEMPORAL ACTIVATION CLASSIFIER EVALUATION", flush=True)
    print(f"  classifier : {classifier_checkpoint_path}", flush=True)
    print(f"  input      : {input_representation}", flush=True)
    print(f"  test data  : {args.test_dataset}/{evaluation_data.name}", flush=True)
    print(f"  device     : {device}", flush=True)

    activation_layer = int(config.get("activation_layer", 24))
    activation_cache = resolve_activation_cache(
        args.test_dataset,
        evaluation_data,
        explicit_path=args.activation_cache,
    )
    print(f"  activations: layer {activation_layer} from {activation_cache}", flush=True)
    activation_sequences = load_activation_sequences(activation_cache, activation_layer)
    vq_checkpoint_path = code_features = None
    if input_representation in {"vq", "hybrid"}:
        recorded_vq_path = Path(config["vq_checkpoint_path"])
        if args.vq_run:
            vq_run = resolve_project_path(args.vq_run)
            vq_checkpoint_path = vq_run / (args.vq_checkpoint or recorded_vq_path.name)
        elif args.vq_checkpoint:
            vq_checkpoint_path = recorded_vq_path.parent / args.vq_checkpoint
        else:
            vq_checkpoint_path = resolve_project_path(recorded_vq_path)
        print(f"  VQ         : {vq_checkpoint_path}", flush=True)

        vq_model, vq_checkpoint = load_vq_checkpoint(vq_checkpoint_path, device)
        vq_layer = int(vq_checkpoint["config"].get("read_layer", 24))
        if vq_layer != activation_layer:
            raise ValueError(
                f"classifier expects layer {activation_layer}, but the VQ checkpoint reads layer {vq_layer}"
            )
        codebook = vq_model.quantizer.codebook.detach().float().cpu().clone()
        code_features = classifier_checkpoint["code_features"]
        code_vector_source = config.get("code_vector_source", "codebook")
        if code_vector_source == "codebook":
            code_vectors = codebook
        elif code_vector_source == "activation_mean":
            code_vectors = classifier_checkpoint["code_vectors"]
        else:
            raise ValueError(f"unknown code-vector source {code_vector_source!r}")
        print(f"  code vectors: {code_vector_source}", flush=True)
        sequences = (
            assign_hybrid_sequences(
                vq_model, activation_sequences, device, args.code_batch_size,
                include_token_ids=bool(args.audit_cache_out),
            )
            if input_representation == "hybrid"
            else assign_code_sequences(
                vq_model, activation_sequences, device, args.code_batch_size,
                include_token_ids=bool(args.audit_cache_out),
            )
        )
        del activation_sequences, vq_model
        if input_representation == "hybrid":
            model = HybridSequenceClassifier(
                config["activation_dim"], code_vectors, code_features,
                config["raw_projection_dim"], config["vq_projection_dim"], config["hidden_dim"],
                config["num_layers"], config["dropout"], config["hazard_bias_initialization"],
                harmfulness_score_weight,
            ).to(device)
            score_sequences = score_hybrid_sequences
        else:
            model = CodeSequenceClassifier(
                code_vectors, code_features, config["projection_dim"], config["hidden_dim"],
                config["num_layers"], config["dropout"], config["hazard_bias_initialization"],
                harmfulness_score_weight,
            ).to(device)
            score_sequences = score_code_sequences
    elif input_representation == "raw":
        if args.vq_run or args.vq_checkpoint:
            raise ValueError("VQ overrides do not apply to a raw-activation classifier")
        sequences = activation_sequences
        model = RawActivationSequenceClassifier(
            config["activation_dim"], config["projection_dim"], config["hidden_dim"],
            config["num_layers"], config["dropout"], config["hazard_bias_initialization"],
        ).to(device)
        score_sequences = score_activation_sequences
    else:
        raise ValueError(f"unknown classifier input representation {input_representation!r}")

    if args.audit_cache_out and input_representation == "raw":
        raise ValueError("a concept audit cache requires a VQ or hybrid detector")

    if device == "cuda":
        torch.cuda.empty_cache()
    model.load_state_dict(classifier_checkpoint["model"], strict=True)
    model.eval()
    thresholds = classifier_checkpoint["thresholds"]
    scored = score_sequences(
        model, sequences, args.batch_size, device,
        include_token_details=bool(args.audit_cache_out),
    )
    streaming_thresholds = dict(thresholds["streaming"])
    audit_cache_path = None
    if args.audit_cache_out:
        audit_cache_path = resolve_project_path(args.audit_cache_out)
        write_detection_cache(
            audit_cache_path,
            scored,
            code_features,
            {
                "classifier_checkpoint": str(classifier_checkpoint_path),
                "vq_checkpoint": str(vq_checkpoint_path),
                "input_representation": input_representation,
                "code_vector_source": config.get("code_vector_source", "codebook"),
                "code_feature_names": config.get("code_feature_names", []),
                "dataset": args.test_dataset,
                "test_set": evaluation_data.name,
                "activation_cache": str(activation_cache),
                "response_thresholds": thresholds["response"],
                "streaming_thresholds": streaming_thresholds,
            },
        )
        print(f"  audit cache: {audit_cache_path}", flush=True)
    detection_positions = streaming_detection_positions(scored, streaming_thresholds)
    response_report = summarize_scores(scored["labels"], scored["response_scores"], thresholds["response"])
    streaming_report = summarize_scores(scored["labels"], scored["streaming_scores"], streaming_thresholds)
    report = {
        "classifier_checkpoint": str(classifier_checkpoint_path),
        "input_representation": input_representation,
        "training_dataset": config["dataset"],
        "training_set": config["train_set"],
        "test_dataset": args.test_dataset,
        "test_set": evaluation_data.name,
        "n_responses": int(len(scored["labels"])),
        "n_safe": int((scored["labels"] == 0).sum()),
        "n_harmful": int((scored["labels"] == 1).sum()),
        "threshold_source": "classifier validation split",
        "response": response_report,
        "streaming": streaming_report,
        "streaming_detection_position": {
            name: summarize_detection_positions(detection_positions, name)
            for name in streaming_thresholds
        },
        "streaming_detection_positions_file": "streaming_detection_positions.jsonl",
    }
    if vq_checkpoint_path is not None:
        report["vq_checkpoint"] = str(vq_checkpoint_path)
        report["code_vector_source"] = config.get("code_vector_source", "codebook")
    if audit_cache_path is not None:
        report["audit_cache"] = str(audit_cache_path)
    output_name = f"evaluation_{args.test_dataset}_{evaluation_data.name}".replace("/", "_")
    output_dir = classifier_run / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "detection_report.json"
    json.dump(report, open(report_path, "w"), indent=2)
    positions_path = output_dir / report["streaming_detection_positions_file"]
    with positions_path.open("w") as output_file:
        for row in detection_positions:
            output_file.write(json.dumps(row) + "\n")

    print(f"  responses  : {report['n_harmful']} harmful / {report['n_safe']} safe", flush=True)
    report_sections = [
        ("response", response_report),
        ("cumulative streaming", streaming_report),
    ]
    for name, section in report_sections:
        print(f"\n  {name.upper()}", flush=True)
        print(f"    AUC {section['auc']:.3f}  AP {section['average_precision']:.3f}", flush=True)
        for operating_name, metrics in section["operating_points"].items():
            print(
                f"    {operating_name:<8} threshold {metrics['threshold']:.4f}  "
                f"F1 {metrics['f1']:.3f}  precision {metrics['precision']:.3f}  "
                f"recall {metrics['recall']:.3f}  FPR {metrics['fpr']:.3f}",
                flush=True,
            )
    print(f"\n  report     : {report_path}", flush=True)


if __name__ == "__main__":
    main()
