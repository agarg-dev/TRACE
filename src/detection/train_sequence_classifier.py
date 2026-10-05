#!/usr/bin/env python
"""Train a causal GRU detector on activations, VQ concepts, or both."""

import argparse
import json
import time
import numpy as np
import torch
import yaml

from activations.activation_cache import (
    load_activation_sequences,
    read_activation_cache_info,
    resolve_activation_cache,
)
from data.dataset_splits import DETECTOR_TRAIN_SET_BY_DATASET, train_validation_split, training_split
from project_config import DEFAULT_DATASET, DETECTION_RUNS_DIR, READ_LAYER, resolve_project_path
from vq.model import load_vq_checkpoint

from detection.sequence_classifier import (
    CodeSequenceClassifier,
    HybridSequenceClassifier,
    RawActivationSequenceClassifier,
    activation_mean_code_vectors,
    assign_code_sequences,
    assign_hybrid_sequences,
    checkpoint_code_feature_statistics,
    evaluate_loss,
    initial_hazard_bias,
    maximum_f1,
    pad_activation_batch,
    pad_code_batch,
    pad_hybrid_batch,
    rescore_no_task_checkpoint_by_response_presence,
    score_activation_sequences,
    score_code_sequences,
    score_hybrid_sequences,
    select_thresholds,
    sequence_batches,
    streaming_classification_loss,
    summarize_scores,
    threshold_report,
)


def read_config_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    config_path = parser.parse_known_args()[0].config
    if not config_path:
        return {}

    path = resolve_project_path(config_path)
    return yaml.safe_load(path.read_text()) or {}


def parse_args():
    config = read_config_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML file containing detector settings")
    parser.add_argument("--input-representation", choices=["vq", "raw", "hybrid"], default="vq",
                        help="VQ vectors, raw cached activations, or both representations")
    parser.add_argument("--vq-run")
    parser.add_argument("--vq-checkpoint", default="model_joint.pt")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-set", default=None)
    parser.add_argument("--activation-cache", default=None,
                        help="activation cache to use (VQ mode defaults to its recorded training cache)")
    parser.add_argument("--activation-layer", type=int, default=None,
                        help="cached layer to read (VQ mode defaults to the checkpoint's read layer)")
    parser.add_argument("--out", help="classifier run directory")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--raw-projection-dim", type=int, default=256,
                        help="raw branch width in hybrid mode")
    parser.add_argument("--vq-projection-dim", type=int, default=256,
                        help="VQ branch width in hybrid mode")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--code-batch-size", type=int, default=16,
                        help="responses per batch while assigning frozen VQ codes")
    parser.add_argument("--code-vector-source", choices=["codebook", "activation_mean"], default="codebook",
                        help="frozen vector looked up for each assigned VQ code")
    parser.add_argument("--code-score-prior-strength", type=float, default=10.0,
                        help="required prior strength of the stored VQ harmfulness score")
    parser.add_argument("--harmfulness-score-weight", type=float, default=None,
                        help="direct signed code-score contribution to each token's hazard logit")
    parser.add_argument(
        "--rescore-response-presence", action="store_true",
        help="for a no-task VQ checkpoint, recompute presence scores from its exact VQ-training partition",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.5,
                        help="multiply the learning rate by this factor when validation loss plateaus")
    parser.add_argument("--lr-patience", type=int, default=2,
                        help="validation-loss plateau patience before reducing the learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="minimum learning rate used by the plateau scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--streaming-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--selection-metric", choices=["mean_ap", "mean_f1"], default="mean_ap",
        help="checkpoint selection: mean AP or mean best-threshold F1 across validation heads",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="default: use the VQ checkpoint's split seed")
    parser.set_defaults(**config)
    return parser.parse_args()


def main():
    args = parse_args()
    uses_vq = args.input_representation in {"vq", "hybrid"}
    if uses_vq and not args.vq_run:
        raise ValueError("--vq-run is required for VQ and hybrid detectors")
    if args.rescore_response_presence and not uses_vq:
        raise ValueError("response-presence rescoring requires a VQ-based detector")
    if args.code_vector_source != "codebook" and not uses_vq:
        raise ValueError("activation-mean code vectors require VQ or hybrid input")
    if args.harmfulness_score_weight is None:
        args.harmfulness_score_weight = 1.0 if uses_vq else 0.0
    if args.input_representation == "raw" and args.harmfulness_score_weight:
        args.harmfulness_score_weight = 0.0
    args.train_set = args.train_set or DETECTOR_TRAIN_SET_BY_DATASET[args.dataset]
    training_data = training_split(args.train_set, args.dataset)
    args.train_set = training_data.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = time.time()

    print("\nTEMPORAL ACTIVATION CLASSIFIER", flush=True)
    print(f"  input         : {args.input_representation}", flush=True)
    print(f"  training data : {args.dataset}/{training_data.name}", flush=True)
    print(f"  device        : {device}", flush=True)
    print(f"  selection     : {args.selection_metric} on validation responses", flush=True)

    vq_run = vq_checkpoint_path = codebook = code_features = code_statistics = checkpoint_regions = None
    code_vectors = code_vector_counts = None
    if uses_vq:
        vq_run = resolve_project_path(args.vq_run)
        vq_checkpoint_path = vq_run / args.vq_checkpoint
        print(f"  VQ checkpoint : {vq_checkpoint_path}", flush=True)
        vq_model, vq_checkpoint = load_vq_checkpoint(vq_checkpoint_path, device)
        vq_config = vq_checkpoint["config"]
        checkpoint_layer = int(vq_config.get("read_layer", READ_LAYER))
        if args.activation_layer is not None and args.activation_layer != checkpoint_layer:
            raise ValueError(
                f"--activation-layer {args.activation_layer} does not match the VQ checkpoint's "
                f"read layer {checkpoint_layer}"
            )
        args.activation_layer = checkpoint_layer
        split_seed = int(vq_config.get("seed", 42) if args.seed is None else args.seed)
        checkpoint_regions = vq_checkpoint.get("regions")
        recorded_cache = vq_config.get("training_cache")
        if recorded_cache:
            print(f"  VQ training data: {recorded_cache}", flush=True)
        else:
            print("  checkpoint does not record its training cache", flush=True)
        if args.activation_cache is None and recorded_cache:
            checkpoint_dataset = vq_config.get("dataset")
            checkpoint_train_set = vq_config.get("train_set")
            if checkpoint_dataset == args.dataset and checkpoint_train_set == args.train_set:
                args.activation_cache = recorded_cache
        del vq_checkpoint
    else:
        args.activation_layer = READ_LAYER if args.activation_layer is None else args.activation_layer
        split_seed = int(42 if args.seed is None else args.seed)

    activation_cache = resolve_activation_cache(
        args.dataset,
        training_data,
        explicit_path=args.activation_cache,
    )
    print(f"  activations   : layer {args.activation_layer} from {activation_cache}", flush=True)
    activation_sequences = load_activation_sequences(activation_cache, args.activation_layer)
    if uses_vq:
        retain_response_activations = (
            args.input_representation == "hybrid" or args.code_vector_source == "activation_mean"
        )
        sequences = (
            assign_hybrid_sequences(vq_model, activation_sequences, device, args.code_batch_size)
            if retain_response_activations
            else assign_code_sequences(vq_model, activation_sequences, device, args.code_batch_size)
        )
        codebook = vq_model.quantizer.codebook.detach().float().cpu().clone()
        num_codes = codebook.shape[0]
        activation_dim = int(activation_sequences[0]["x"].shape[1])
        del activation_sequences, vq_model
        if args.input_representation == "hybrid":
            pad_batch, score_sequences = pad_hybrid_batch, score_hybrid_sequences
        else:
            pad_batch, score_sequences = pad_code_batch, score_code_sequences
        if args.rescore_response_presence:
            if vq_config.get("dataset") != args.dataset or vq_config.get("train_set") != args.train_set:
                raise ValueError("response-presence rescoring requires the VQ checkpoint's training dataset")
            checkpoint_regions = rescore_no_task_checkpoint_by_response_presence(
                sequences, num_codes, vq_config, args.code_score_prior_strength
            )
            print(
                f"  rescored      : response presence over "
                f"{checkpoint_regions['n_score_responses']} VQ-training responses",
                flush=True,
            )
    else:
        sequences = activation_sequences
        activation_dim = int(sequences[0]["x"].shape[1])
        pad_batch = pad_activation_batch
        score_sequences = score_activation_sequences

    args.seed = split_seed
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)
    if device == "cuda":
        torch.cuda.empty_cache()

    train, validation = train_validation_split(sequences, split_seed)
    train_labels = np.array([sequence["label"] for sequence in train])
    n_safe, n_harmful = int((train_labels == 0).sum()), int((train_labels == 1).sum())
    if not n_safe or not n_harmful:
        raise ValueError("the classifier training split must contain both safe and harmful responses")
    class_weights = torch.tensor([1.0, n_safe / n_harmful], device=device)
    if uses_vq:
        code_vectors = codebook
        if args.code_vector_source == "activation_mean":
            code_vectors, code_vector_counts = activation_mean_code_vectors(train, codebook)
            if args.input_representation == "vq":
                train = [{name: value for name, value in sequence.items() if name != "x"} for sequence in train]
                validation = [
                    {name: value for name, value in sequence.items() if name != "x"}
                    for sequence in validation
                ]
        code_features, code_statistics = checkpoint_code_feature_statistics(
            checkpoint_regions, train, num_codes, args.code_score_prior_strength
        )
    hazard_bias = initial_hazard_bias(train)
    print(f"  split         : {len(train)} train / {len(validation)} validation", flush=True)
    print(f"  train labels  : {n_safe} safe / {n_harmful} harmful", flush=True)
    if uses_vq:
        print(f"  codebook      : {num_codes} × {codebook.shape[1]}", flush=True)
        if args.code_vector_source == "activation_mean":
            print(f"  code vectors  : mean layer-{args.activation_layer} training activation per assigned code "
                  f"({int((code_vector_counts == 0).sum())} unused codes retain codebook vectors)",
                  flush=True)
        else:
            print("  code vectors  : learned VQ codebook", flush=True)
        print(
            f"  code score    : {code_statistics['score_method'].replace('_', '-')} from "
            f"{code_statistics['score_source']} (prior strength {code_statistics['prior_strength']:g})",
            flush=True,
        )
        print("  code support  : response presence in the classifier training split (labels unused)", flush=True)
        print(f"  hazard prior  : code score × {args.harmfulness_score_weight:g} + GRU correction", flush=True)
        if args.input_representation == "hybrid":
            print(f"  raw branch    : {activation_dim} → {args.raw_projection_dim}", flush=True)
            print(f"  VQ branch     : {codebook.shape[1]} → {args.vq_projection_dim}", flush=True)
            vector_prefix = "activation_mean_" if args.code_vector_source == "activation_mean" else ""
            default_run_name = f"{vector_prefix}hybrid_gru_hazard_K{num_codes}"
        else:
            default_run_name = f"code_gru_hazard_K{num_codes}"
    else:
        print(f"  activations   : raw layer {args.activation_layer}, dimension {activation_dim}", flush=True)
        print("  hazard prior  : GRU prediction only (no VQ code-score features)", flush=True)
        default_run_name = "raw_activation_gru_hazard"

    run_dir = (
        resolve_project_path(args.out) if args.out
        else DETECTION_RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{default_run_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.input_representation == "vq":
        model = CodeSequenceClassifier(
            code_vectors, code_features, args.projection_dim, args.hidden_dim, args.num_layers,
            args.dropout, hazard_bias, args.harmfulness_score_weight,
        ).to(device)
    elif args.input_representation == "hybrid":
        model = HybridSequenceClassifier(
            activation_dim, code_vectors, code_features, args.raw_projection_dim, args.vq_projection_dim,
            args.hidden_dim, args.num_layers, args.dropout, hazard_bias, args.harmfulness_score_weight,
        ).to(device)
    else:
        model = RawActivationSequenceClassifier(
            activation_dim, args.projection_dim, args.hidden_dim, args.num_layers,
            args.dropout, hazard_bias,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_reduction_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )

    history, best_state = [], None
    best_selection_score, best_response_ap, best_streaming_ap = -1.0, -1.0, -1.0
    best_response_f1, best_streaming_f1 = -1.0, -1.0
    best_epoch, epochs_without_improvement = -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        totals = {"loss": 0.0, "response": 0.0, "streaming": 0.0}
        batch_count = 0
        epoch_started_at = time.time()
        for batch in sequence_batches(train, args.batch_size, shuffle=True, seed=split_seed + epoch):
            inputs, lengths, labels = pad_batch(batch, device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                response_logits, hazard_logits, valid_mask = model(inputs, lengths)
                losses = streaming_classification_loss(
                    response_logits, hazard_logits, valid_mask, lengths, labels, class_weights,
                    args.response_weight, args.streaming_weight,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].item())
            batch_count += 1

        train_losses = {name: value / max(1, batch_count) for name, value in totals.items()}
        validation_losses = evaluate_loss(
            model, validation, args.batch_size, args.response_weight, args.streaming_weight,
            class_weights, device, pad_batch,
        )
        validation_scores = score_sequences(model, validation, args.batch_size, device)
        validation_response_ap = float(summarize_scores(
            validation_scores["labels"], validation_scores["response_scores"], {"argmax": 0.5}
        )["average_precision"])
        validation_streaming_ap = float(summarize_scores(
            validation_scores["labels"], validation_scores["streaming_scores"], {"argmax": 0.5}
        )["average_precision"])
        validation_response_f1 = maximum_f1(validation_scores["labels"], validation_scores["response_scores"])
        validation_streaming_f1 = maximum_f1(validation_scores["labels"], validation_scores["streaming_scores"])
        if args.selection_metric == "mean_f1":
            selection_score = (validation_response_f1 + validation_streaming_f1) / 2
        else:
            selection_score = (validation_response_ap + validation_streaming_ap) / 2
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "seconds": round(time.time() - epoch_started_at, 1),
            "train": train_losses,
            "validation": validation_losses,
            "validation_response_ap": validation_response_ap,
            "validation_streaming_ap": validation_streaming_ap,
            "validation_response_f1": validation_response_f1,
            "validation_streaming_f1": validation_streaming_f1,
            "selection_score": selection_score,
        }
        history.append(row)
        print(
            f"  epoch {epoch:>2}  train {train_losses['loss']:.4f}  "
            f"val {validation_losses['loss']:.4f}  response AP {validation_response_ap:.4f}  "
            f"stream AP {validation_streaming_ap:.4f}  "
            f"selection {args.selection_metric} {selection_score:.4f}  "
            f"lr {learning_rate:.1e}  {row['seconds']:.1f}s",
            flush=True,
        )
        scheduler.step(validation_losses["loss"])
        updated_learning_rate = float(optimizer.param_groups[0]["lr"])
        if updated_learning_rate < learning_rate:
            print(f"           validation loss plateau: reducing lr to {updated_learning_rate:.1e}", flush=True)

        if selection_score > best_selection_score + 1e-6:
            best_selection_score = selection_score
            best_response_ap = validation_response_ap
            best_streaming_ap = validation_streaming_ap
            best_response_f1 = validation_response_f1
            best_streaming_f1 = validation_streaming_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"  early stopping after epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state, strict=True)

    validation_scores = score_sequences(model, validation, args.batch_size, device)
    thresholds = {
        "response": select_thresholds(validation_scores["labels"], validation_scores["response_scores"]),
        "streaming": select_thresholds(validation_scores["labels"], validation_scores["streaming_scores"]),
    }
    validation_report = threshold_report(validation_scores, thresholds)
    saved_arguments = vars(args).copy()
    config = {
        **saved_arguments,
        **{key: value for key, value in read_activation_cache_info(activation_cache).items()
           if key == "base_model"},
        "training_cache": str(activation_cache),
        "hazard_bias_initialization": hazard_bias,
        "split_seed": split_seed,
        "split_strategy": "stratified_90_10",
        "n_training_responses": len(train),
        "n_validation_responses": len(validation),
    }
    if uses_vq:
        config.update({
            "vq_run": str(vq_run),
            "vq_checkpoint_path": str(vq_checkpoint_path),
            "num_codes": int(num_codes),
            "codebook_dim": int(codebook.shape[1]),
            "code_feature_names": code_statistics["feature_names"],
            "code_score_source": code_statistics["score_source"],
            "code_score_method": code_statistics["score_method"],
            "code_vector_source": args.code_vector_source,
        })
        if args.code_vector_source == "activation_mean":
            config.update({
                "code_vector_activation_layer": args.activation_layer,
                "code_vector_training_partition": "classifier_training_split",
                "n_unused_code_vectors": int((code_vector_counts == 0).sum()),
            })
        if args.input_representation == "hybrid":
            config.pop("projection_dim", None)
            config.update({
                "activation_dim": activation_dim,
                "hybrid_input_dim": args.raw_projection_dim + args.vq_projection_dim + code_features.shape[1],
            })
        else:
            config.pop("raw_projection_dim", None)
            config.pop("vq_projection_dim", None)
    else:
        config.update({"activation_dim": activation_dim})
        for irrelevant_name in (
            "vq_run", "vq_checkpoint", "code_batch_size", "code_score_prior_strength",
            "raw_projection_dim", "vq_projection_dim", "code_vector_source",
        ):
            config.pop(irrelevant_name, None)

    # AP and F1 describe the retained epoch, whichever metric selected it.
    classifier_checkpoint = {
        "model": best_state,
        "config": config,
        "thresholds": thresholds,
        "best_epoch": best_epoch,
        "best_validation_response_ap": best_response_ap,
        "best_validation_streaming_ap": best_streaming_ap,
        "best_validation_mean_ap": (best_response_ap + best_streaming_ap) / 2,
        "best_validation_response_f1": best_response_f1,
        "best_validation_streaming_f1": best_streaming_f1,
        "best_validation_mean_f1": (best_response_f1 + best_streaming_f1) / 2,
        "best_validation_selection_score": best_selection_score,
    }
    if uses_vq:
        classifier_checkpoint.update({
            "code_features": code_features,
            "code_statistics": code_statistics,
        })
        if args.code_vector_source == "activation_mean":
            classifier_checkpoint.update({
                "code_vectors": code_vectors,
                "code_vector_token_counts": code_vector_counts,
            })
    torch.save(classifier_checkpoint, run_dir / "classifier.pt")
    json.dump(validation_report, open(run_dir / "validation_report.json", "w"), indent=2)
    training_summary = {
        "config": config,
        "best_epoch": best_epoch,
        "best_validation_response_ap": best_response_ap,
        "best_validation_streaming_ap": best_streaming_ap,
        "best_validation_mean_ap": (best_response_ap + best_streaming_ap) / 2,
        "best_validation_response_f1": best_response_f1,
        "best_validation_streaming_f1": best_streaming_f1,
        "best_validation_mean_f1": (best_response_f1 + best_streaming_f1) / 2,
        "best_validation_selection_score": best_selection_score,
        "thresholds": thresholds,
        "total_minutes": round((time.time() - started_at) / 60, 1),
        "history": history,
    }
    if code_statistics is not None:
        training_summary["code_statistics"] = {
            name: value for name, value in code_statistics.items()
            if name not in {"response_counts", "signed_harmfulness"}
        }
    json.dump(training_summary, open(run_dir / "training_summary.json", "w"), indent=2)
    print(f"\n  best epoch    : {best_epoch}", flush=True)
    print(f"  validation AP : response {best_response_ap:.4f}, streaming {best_streaming_ap:.4f}", flush=True)
    print(f"  selection     : {args.selection_metric} {best_selection_score:.4f}", flush=True)
    print(f"  classifier    : {run_dir / 'classifier.pt'}", flush=True)


if __name__ == "__main__":
    main()
