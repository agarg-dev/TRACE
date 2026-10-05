#!/usr/bin/env python
"""Tune the causal detector on validation data and save the selected checkpoint."""

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import optuna
import torch

from activations.activation_cache import (
    load_activation_sequences,
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
    score_activation_sequences,
    score_code_sequences,
    score_hybrid_sequences,
    select_thresholds,
    sequence_batches,
    streaming_classification_loss,
    summarize_scores,
    threshold_report,
)


WIDTHS = [64, 96, 128, 192, 256]
WEIGHT_DECAYS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
OBJECTIVE_LABELS = {
    "mean_ap": "mean validation response/streaming AP",
    "mean_f1": "mean validation-selected response/streaming F1",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-representation", choices=["vq", "raw", "hybrid"], default="vq")
    parser.add_argument("--vq-run", required=True)
    parser.add_argument("--vq-checkpoint", default="model_joint.pt")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-set", default=None)
    parser.add_argument("--activation-cache", default=None)
    parser.add_argument("--activation-layer", type=int, default=None)
    parser.add_argument("--raw-projection-dim", type=int, default=256,
                        help="fixed raw branch width in hybrid mode")
    parser.add_argument("--vq-projection-dim", type=int, default=256,
                        help="fixed VQ branch width in hybrid mode")
    parser.add_argument("--out", default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--startup-trials", type=int, default=12)
    parser.add_argument("--objective", choices=OBJECTIVE_LABELS, default="mean_ap")
    parser.add_argument(
        "--finalize-existing-study", action="store_true",
        help="retrain and evaluate the current best completed trial without launching more trials",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--code-batch-size", type=int, default=16)
    parser.add_argument("--code-vector-source", choices=["codebook", "activation_mean"], default="codebook",
                        help="frozen vector looked up for each assigned VQ code")
    parser.add_argument("--lr-reduction-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def suggest_hyperparameters(trial, input_representation, raw_projection_dim=256, vq_projection_dim=256):
    parameters = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", WIDTHS),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", WEIGHT_DECAYS),
        "streaming_weight": trial.suggest_float("streaming_weight", 0.5, 2.0, log=True),
    }
    if input_representation == "hybrid":
        parameters.update({
            "raw_projection_dim": raw_projection_dim,
            "vq_projection_dim": vq_projection_dim,
        })
    else:
        parameters["projection_dim"] = trial.suggest_categorical("projection_dim", WIDTHS)
    if input_representation in {"vq", "hybrid"}:
        parameters.update({
            "harmfulness_score_weight": trial.suggest_float("harmfulness_score_weight", 0.0, 2.0, step=0.25),
        })
    else:
        parameters.update({"harmfulness_score_weight": 0.0})
    return parameters


def make_model(prepared, parameters, device):
    if prepared.input_representation == "vq":
        return CodeSequenceClassifier(
            prepared.codebook, prepared.code_features, parameters["projection_dim"], parameters["hidden_dim"],
            1, parameters["dropout"], prepared.hazard_bias, parameters["harmfulness_score_weight"],
        ).to(device)

    if prepared.input_representation == "hybrid":
        return HybridSequenceClassifier(
            prepared.activation_dim, prepared.codebook, prepared.code_features,
            parameters["raw_projection_dim"], parameters["vq_projection_dim"], parameters["hidden_dim"],
            1, parameters["dropout"], prepared.hazard_bias, parameters["harmfulness_score_weight"],
        ).to(device)

    return RawActivationSequenceClassifier(
        prepared.activation_dim, parameters["projection_dim"], parameters["hidden_dim"],
        1, parameters["dropout"], prepared.hazard_bias,
    ).to(device)


def train_configuration(prepared, parameters, args, device):
    """Train one deterministic configuration and retain its best validation-objective checkpoint."""
    torch.manual_seed(prepared.split_seed)
    np.random.seed(prepared.split_seed)
    if device == "cuda":
        torch.cuda.empty_cache()

    model = make_model(prepared, parameters, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_reduction_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )
    history, best_state = [], None
    best_score = best_response_ap = best_streaming_ap = -1.0
    best_response_f1 = best_streaming_f1 = -1.0
    best_epoch, epochs_without_improvement = -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        total_loss, batch_count = 0.0, 0
        for batch in sequence_batches(
            prepared.train, args.batch_size, shuffle=True, seed=prepared.split_seed + epoch
        ):
            inputs, lengths, labels = prepared.pad_batch(batch, device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                response_logits, hazard_logits, valid_mask = model(inputs, lengths)
                losses = streaming_classification_loss(
                    response_logits, hazard_logits, valid_mask, lengths, labels, prepared.class_weights,
                    args.response_weight, parameters["streaming_weight"],
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(losses["loss"].item())
            batch_count += 1

        validation_losses = evaluate_loss(
            model, prepared.validation, args.batch_size, args.response_weight,
            parameters["streaming_weight"], prepared.class_weights, device, prepared.pad_batch,
        )
        scored = prepared.score_sequences(model, prepared.validation, args.batch_size, device)
        response_summary = summarize_scores(scored["labels"], scored["response_scores"], {"argmax": 0.5})
        streaming_summary = summarize_scores(scored["labels"], scored["streaming_scores"], {"argmax": 0.5})
        response_ap = float(response_summary["average_precision"])
        streaming_ap = float(streaming_summary["average_precision"])
        response_f1 = maximum_f1(scored["labels"], scored["response_scores"])
        streaming_f1 = maximum_f1(scored["labels"], scored["streaming_scores"])
        if args.objective == "mean_f1":
            selection_score = (response_f1 + streaming_f1) / 2
        else:
            selection_score = (response_ap + streaming_ap) / 2
        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": total_loss / max(1, batch_count),
            "validation_loss": validation_losses["loss"],
            "validation_response_ap": response_ap,
            "validation_streaming_ap": streaming_ap,
            "validation_response_f1": response_f1,
            "validation_streaming_f1": streaming_f1,
            "selection_score": selection_score,
        })
        scheduler.step(validation_losses["loss"])

        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_response_ap = response_ap
            best_streaming_ap = streaming_ap
            best_response_f1 = response_f1
            best_streaming_f1 = streaming_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state, strict=True)
    validation_scores = prepared.score_sequences(model, prepared.validation, args.batch_size, device)
    response_summary = summarize_scores(
        validation_scores["labels"], validation_scores["response_scores"], {"argmax": 0.5}
    )
    streaming_summary = summarize_scores(
        validation_scores["labels"], validation_scores["streaming_scores"], {"argmax": 0.5}
    )
    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_response_ap": best_response_ap,
        "best_streaming_ap": best_streaming_ap,
        "best_response_f1": best_response_f1,
        "best_streaming_f1": best_streaming_f1,
        "validation_response_f1_at_0.5": response_summary["operating_points"]["argmax"]["f1"],
        "validation_streaming_f1_at_0.5": streaming_summary["operating_points"]["argmax"]["f1"],
        "validation_scores": validation_scores,
        "code_features": prepared.code_features,
        "code_statistics": prepared.code_statistics,
        "history": history,
    }


def prepare_data(args, device):
    args.train_set = args.train_set or DETECTOR_TRAIN_SET_BY_DATASET[args.dataset]
    training_data = training_split(args.train_set, args.dataset)
    args.train_set = training_data.name
    vq_run = vq_checkpoint_path = codebook = checkpoint_regions = None
    vq_config = {}

    uses_vq = args.input_representation in {"vq", "hybrid"}
    if args.code_vector_source != "codebook" and not uses_vq:
        raise ValueError("activation-mean code vectors require VQ or hybrid input")
    if uses_vq:
        vq_run = resolve_project_path(args.vq_run)
        vq_checkpoint_path = vq_run / args.vq_checkpoint
        vq_model, vq_checkpoint = load_vq_checkpoint(vq_checkpoint_path, device)
        vq_config = vq_checkpoint["config"]
        checkpoint_layer = int(vq_config.get("read_layer", READ_LAYER))
        if args.activation_layer is not None and args.activation_layer != checkpoint_layer:
            raise ValueError(
                f"--activation-layer {args.activation_layer} does not match checkpoint layer {checkpoint_layer}"
            )
        args.activation_layer = checkpoint_layer
        split_seed = int(vq_config.get("seed", 42) if args.seed is None else args.seed)
        checkpoint_regions = vq_checkpoint.get("regions")
        if (
            args.activation_cache is None
            and vq_config.get("dataset") == args.dataset
            and vq_config.get("train_set") == args.train_set
        ):
            args.activation_cache = vq_config.get("training_cache")
        del vq_checkpoint
    else:
        vq_model = None
        args.activation_layer = READ_LAYER if args.activation_layer is None else args.activation_layer
        split_seed = int(42 if args.seed is None else args.seed)

    activation_cache = resolve_activation_cache(
        args.dataset,
        training_data,
        explicit_path=args.activation_cache,
    )
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
        vq_codebook = vq_model.quantizer.codebook.detach().float().cpu().clone()
        activation_dim = int(activation_sequences[0]["x"].shape[1])
        del activation_sequences, vq_model
        if args.input_representation == "hybrid":
            pad_batch, score_sequences = pad_hybrid_batch, score_hybrid_sequences
        else:
            pad_batch, score_sequences = pad_code_batch, score_code_sequences
        num_codes = int(vq_codebook.shape[0])
    else:
        sequences = activation_sequences
        pad_batch, score_sequences = pad_activation_batch, score_activation_sequences
        activation_dim, num_codes, vq_codebook = int(sequences[0]["x"].shape[1]), None, None

    train, validation = train_validation_split(sequences, split_seed)
    labels = np.array([sequence["label"] for sequence in train])
    n_safe, n_harmful = int((labels == 0).sum()), int((labels == 1).sum())
    if not n_safe or not n_harmful:
        raise ValueError("the classifier training split must contain both classes")
    codebook, code_vector_counts = vq_codebook, None
    code_features = code_statistics = None
    if uses_vq:
        if args.code_vector_source == "activation_mean":
            codebook, code_vector_counts = activation_mean_code_vectors(train, vq_codebook)
            if args.input_representation == "vq":
                train = [{name: value for name, value in sequence.items() if name != "x"} for sequence in train]
                validation = [
                    {name: value for name, value in sequence.items() if name != "x"}
                    for sequence in validation
                ]
        code_features, code_statistics = checkpoint_code_feature_statistics(
            checkpoint_regions, train, num_codes, expected_prior_strength=10.0
        )
    return SimpleNamespace(
        input_representation=args.input_representation,
        training_data=training_data,
        activation_cache=activation_cache,
        activation_dim=activation_dim,
        activation_layer=args.activation_layer,
        vq_run=vq_run,
        vq_checkpoint_path=vq_checkpoint_path,
        vq_config=vq_config,
        vq_codebook=vq_codebook,
        codebook=codebook,
        code_vector_counts=code_vector_counts,
        num_codes=num_codes,
        code_features=code_features,
        code_statistics=code_statistics,
        train=train,
        validation=validation,
        split_seed=split_seed,
        n_safe=n_safe,
        n_harmful=n_harmful,
        class_weights=torch.tensor([1.0, n_safe / n_harmful], device=device),
        hazard_bias=initial_hazard_bias(train),
        pad_batch=pad_batch,
        score_sequences=score_sequences,
    )


def serialize_trials(study):
    records = []
    for trial in study.trials:
        records.append({
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        })
    return records


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = time.time()
    prepared = prepare_data(args, device)
    args.seed = prepared.split_seed

    objective_suffix = "" if args.objective == "mean_ap" else f"_{args.objective}"
    default_name = (
        f"optuna_{args.trials}_{args.input_representation}_gru_"
        f"layer{args.activation_layer}{objective_suffix}"
    )
    run_dir = (
        resolve_project_path(args.out) if args.out
        else DETECTION_RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{default_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    study_path = run_dir / "study.db"
    storage = f"sqlite:///{study_path.resolve()}"
    sampler_seed = prepared.split_seed
    if args.finalize_existing_study:
        if not study_path.is_file():
            raise FileNotFoundError(f"cannot finalize missing Optuna study: {study_path}")
        study = optuna.load_study(study_name=default_name, storage=storage)
    else:
        if study_path.is_file():
            existing_study = optuna.load_study(study_name=default_name, storage=storage)
            sampler_seed += len(existing_study.trials)
        study = optuna.create_study(
            study_name=default_name, storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=sampler_seed, n_startup_trials=args.startup_trials, multivariate=True
            ),
            load_if_exists=True,
        )
    reference_parameters = {
        "hidden_dim": 128,
        "dropout": 0.1,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "streaming_weight": 1.0,
    }
    if args.input_representation != "hybrid":
        reference_parameters["projection_dim"] = 128
    if args.input_representation in {"vq", "hybrid"}:
        reference_parameters.update({"harmfulness_score_weight": 1.0})
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if args.finalize_existing_study and not completed_trials:
        raise ValueError("cannot finalize an Optuna study without a completed trial")
    if not args.finalize_existing_study and not study.trials:
        study.enqueue_trial(reference_parameters)

    print("\nOPTUNA TEMPORAL CLASSIFIER SEARCH", flush=True)
    print(f"  input       : {args.input_representation}", flush=True)
    print(f"  activations : layer {args.activation_layer} from {prepared.activation_cache}", flush=True)
    print(f"  split       : {len(prepared.train)} train / {len(prepared.validation)} validation", flush=True)
    print(f"  trials      : {args.trials}", flush=True)
    print(f"  startup     : {args.startup_trials} trials", flush=True)
    print(f"  sampler seed: {sampler_seed}", flush=True)
    print(f"  objective   : {OBJECTIVE_LABELS[args.objective]}", flush=True)
    if args.finalize_existing_study:
        print(f"  mode        : finalize best of {len(completed_trials)} completed trials", flush=True)

    def objective(trial):
        parameters = suggest_hyperparameters(
            trial, args.input_representation, args.raw_projection_dim, args.vq_projection_dim
        )
        result = train_configuration(prepared, parameters, args, device)
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("response_ap", result["best_response_ap"])
        trial.set_user_attr("streaming_ap", result["best_streaming_ap"])
        trial.set_user_attr("response_f1", result["best_response_f1"])
        trial.set_user_attr("streaming_f1", result["best_streaming_f1"])
        trial.set_user_attr("response_f1_at_0.5", result["validation_response_f1_at_0.5"])
        trial.set_user_attr("streaming_f1_at_0.5", result["validation_streaming_f1_at_0.5"])
        metric_name = "mean F1" if args.objective == "mean_f1" else "mean AP"
        print(
            f"  trial {trial.number:>2}: {metric_name} {result['best_score']:.4f}  "
            f"response {result['best_response_f1' if args.objective == 'mean_f1' else 'best_response_ap']:.4f}  "
            f"streaming {result['best_streaming_f1' if args.objective == 'mean_f1' else 'best_streaming_ap']:.4f}  "
            f"epoch {result['best_epoch']}", flush=True,
        )
        value = result["best_score"]
        del result
        if device == "cuda":
            torch.cuda.empty_cache()
        return value

    if not args.finalize_existing_study:
        finished_trials = sum(trial.state.is_finished() for trial in study.trials)
        remaining_trials = max(0, args.trials - finished_trials)
        if remaining_trials:
            study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    best_parameters = suggest_hyperparameters(
        optuna.trial.FixedTrial(study.best_trial.params), args.input_representation,
        args.raw_projection_dim, args.vq_projection_dim,
    )
    print(f"\n  retraining winning trial {study.best_trial.number}: {best_parameters}", flush=True)
    winner = train_configuration(prepared, best_parameters, args, device)
    thresholds = {
        "response": select_thresholds(
            winner["validation_scores"]["labels"], winner["validation_scores"]["response_scores"]
        ),
        "streaming": select_thresholds(
            winner["validation_scores"]["labels"], winner["validation_scores"]["streaming_scores"]
        ),
    }
    validation_report = threshold_report(winner["validation_scores"], thresholds)
    config = {
        "input_representation": args.input_representation,
        "dataset": args.dataset,
        "train_set": args.train_set,
        "activation_cache": str(prepared.activation_cache),
        "activation_layer": args.activation_layer,
        "training_cache": str(prepared.activation_cache),
        "hidden_dim": best_parameters["hidden_dim"],
        "num_layers": 1,
        "dropout": best_parameters["dropout"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": best_parameters["lr"],
        "lr_reduction_factor": args.lr_reduction_factor,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
        "weight_decay": best_parameters["weight_decay"],
        "response_weight": args.response_weight,
        "streaming_weight": best_parameters["streaming_weight"],
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "seed": prepared.split_seed,
        "split_seed": prepared.split_seed,
        "split_strategy": "stratified_90_10",
        "n_training_responses": len(prepared.train),
        "n_validation_responses": len(prepared.validation),
        "hazard_bias_initialization": prepared.hazard_bias,
        "search_method": "Optuna TPE",
        "search_sampler_seed": sampler_seed,
        "search_trials": args.trials,
        "search_startup_trials": args.startup_trials,
        "search_objective": args.objective,
        "search_completed_trials": len(completed_trials),
        "finalized_existing_study": args.finalize_existing_study,
        "selection_metric": OBJECTIVE_LABELS[args.objective],
    }
    if args.input_representation == "hybrid":
        config.update({
            "raw_projection_dim": best_parameters["raw_projection_dim"],
            "vq_projection_dim": best_parameters["vq_projection_dim"],
            "hybrid_input_dim": (
                best_parameters["raw_projection_dim"]
                + best_parameters["vq_projection_dim"]
                + prepared.code_features.shape[1]
            ),
        })
    else:
        config["projection_dim"] = best_parameters["projection_dim"]
    classifier_checkpoint = {
        "model": winner["best_state"],
        "config": config,
        "thresholds": thresholds,
        "best_epoch": winner["best_epoch"],
        "best_validation_response_ap": winner["best_response_ap"],
        "best_validation_streaming_ap": winner["best_streaming_ap"],
        "best_validation_response_f1": winner["best_response_f1"],
        "best_validation_streaming_f1": winner["best_streaming_f1"],
        "best_validation_mean_ap": (winner["best_response_ap"] + winner["best_streaming_ap"]) / 2,
        "best_validation_mean_f1": (winner["best_response_f1"] + winner["best_streaming_f1"]) / 2,
        "best_validation_selection_score": winner["best_score"],
    }
    if args.input_representation in {"vq", "hybrid"}:
        config.update({
            "vq_run": str(prepared.vq_run),
            "vq_checkpoint": args.vq_checkpoint,
            "vq_checkpoint_path": str(prepared.vq_checkpoint_path),
            "num_codes": prepared.num_codes,
            "codebook_dim": int(prepared.codebook.shape[1]),
            "code_batch_size": args.code_batch_size,
            "code_score_prior_strength": winner["code_statistics"]["prior_strength"],
            "harmfulness_score_weight": best_parameters["harmfulness_score_weight"],
            "code_feature_names": winner["code_statistics"]["feature_names"],
            "code_score_source": winner["code_statistics"]["score_source"],
            "code_score_method": winner["code_statistics"]["score_method"],
            "code_vector_source": args.code_vector_source,
        })
        if args.code_vector_source == "activation_mean":
            config.update({
                "code_vector_activation_layer": args.activation_layer,
                "code_vector_training_partition": "classifier_training_split",
                "n_unused_code_vectors": int((prepared.code_vector_counts == 0).sum()),
            })
        classifier_checkpoint.update({
            "code_features": winner["code_features"],
            "code_statistics": winner["code_statistics"],
        })
        if args.code_vector_source == "activation_mean":
            classifier_checkpoint.update({
                "code_vectors": prepared.codebook,
                "code_vector_token_counts": prepared.code_vector_counts,
            })
    if args.input_representation in {"raw", "hybrid"}:
        config["activation_dim"] = prepared.activation_dim

    torch.save(classifier_checkpoint, run_dir / "classifier.pt")
    json.dump(validation_report, open(run_dir / "validation_report.json", "w"), indent=2)
    search_summary = {
        "input_representation": args.input_representation,
        "code_vector_source": args.code_vector_source if args.input_representation in {"vq", "hybrid"} else None,
        "objective": args.objective,
        "n_trials": len(study.trials),
        "n_completed_trials": len(completed_trials),
        "n_startup_trials": args.startup_trials,
        "sampler_seed": sampler_seed,
        "finalized_existing_study": args.finalize_existing_study,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_parameters": best_parameters,
        "best_epoch": winner["best_epoch"],
        "best_validation_response_ap": winner["best_response_ap"],
        "best_validation_streaming_ap": winner["best_streaming_ap"],
        "best_validation_response_f1": winner["best_response_f1"],
        "best_validation_streaming_f1": winner["best_streaming_f1"],
        "best_validation_mean_ap": (winner["best_response_ap"] + winner["best_streaming_ap"]) / 2,
        "best_validation_mean_f1": (winner["best_response_f1"] + winner["best_streaming_f1"]) / 2,
        "best_validation_response_f1_at_0.5": winner["validation_response_f1_at_0.5"],
        "best_validation_streaming_f1_at_0.5": winner["validation_streaming_f1_at_0.5"],
        "total_minutes": round((time.time() - started_at) / 60, 1),
        "trials": serialize_trials(study),
        "winning_history": winner["history"],
    }
    json.dump(search_summary, open(run_dir / "search_summary.json", "w"), indent=2)
    print(f"  classifier  : {run_dir / 'classifier.pt'}", flush=True)
    print(f"  summary     : {run_dir / 'search_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
