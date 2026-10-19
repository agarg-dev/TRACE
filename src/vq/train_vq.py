#!/usr/bin/env python
"""Train the cross-layer VQ model on cached response activations."""

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_with_min_lr_schedule_with_warmup

from activations.cache import (
    load_cross_layer_sequences,
)
from data.dataset_splits import (
    DEFAULT_DATASET,
    train_validation_split,
    training_split,
)
from model_inputs import iter_activation_batches, pad_activation_sequences
from project_config import DEFAULT_BASE_MODEL, MODEL_DIR, READ_LAYER, TARGET_LAYER, VQ_RUNS_DIR
from vq.codebook import (
    CODE_SCORE_METHODS,
    assign_codes_for_sequences,
    code_harmfulness_statistics,
    code_score_region_source,
    initialize_codebook_from_sequences,
    split_assignments_by_response,
)
from vq.diagnostics import (
    append_diagnostic_record,
    gradient_norms_by_component,
    summarize_initial_codebook,
    summarize_region_changes,
    summarize_training_batch,
)
from vq.model import CrossLayerVQVAE


def accumulation_window_size(batch_number, num_batches, grad_accum):
    """Count microbatches in this update, including an incomplete final window."""
    window_start = batch_number - batch_number % grad_accum
    return min(grad_accum, num_batches - window_start)


def activation_rms_scale(sequences, key):
    """Return one scalar RMS estimated only from the supplied response partition."""
    square_sum = 0.0
    value_count = 0
    for sequence in sequences:
        activations = sequence[key].detach().float()
        square_sum += activations.square().sum(dtype=torch.float64).item()
        value_count += activations.numel()
    return math.sqrt(square_sum / value_count)


def relative_reconstruction_loss(reconstructed, target, loss_type="relative_mse"):
    """Mean per-token relative reconstruction loss over non-padding tokens."""
    reconstructed = reconstructed.float()
    valid_tokens = target.norm(dim=2) > 1e-6
    relative_error = (reconstructed - target).pow(2).sum(dim=2) / (target.pow(2).sum(dim=2) + 1e-8)
    if loss_type == "relative_mse":
        token_losses = relative_error
    elif loss_type == "robust_relative_huber":
        # This equals relative MSE while the error norm is no larger than the target norm, then grows
        # linearly so a rare extreme token cannot dominate the accumulated optimizer step.
        token_losses = torch.where(
            relative_error <= 1.0,
            relative_error,
            2.0 * torch.sqrt(relative_error.clamp_min(1e-12)) - 1.0,
        )
    else:
        raise ValueError(f"unknown reconstruction loss {loss_type!r}")
    return token_losses[valid_tokens].mean()


def balanced_region_class_weights(response_labels, device):
    """Return inverse-frequency weights whose mean is one over a binary response partition."""
    response_labels = torch.tensor(list(response_labels), dtype=torch.long, device=device)
    class_counts = torch.bincount(response_labels, minlength=2)
    return response_labels.numel() / (2.0 * class_counts.float())


def region_separation_loss(
    encoded, labels, codebook, benign_codes, harmful_codes, temperature,
    reduction="token_mean", class_weights=None,
):
    """Penalize soft code-assignment mass landing in the response label's opposite region."""
    codebook = codebook.detach().float()
    encoded = encoded.float()
    squared_distance = (
        encoded.pow(2).sum(-1, keepdim=True) - 2 * encoded @ codebook.t() + codebook.pow(2).sum(-1)
    )
    squared_distance = squared_distance / encoded.shape[-1]
    soft_assignments = F.softmax(-squared_distance / temperature, dim=-1)
    wrong_region_mass = torch.where(
        labels == 0,
        soft_assignments.index_select(-1, harmful_codes).sum(-1),
        soft_assignments.index_select(-1, benign_codes).sum(-1),
    )
    valid_tokens = labels >= 0
    if reduction == "token_mean":
        return wrong_region_mass[valid_tokens].mean()
    if reduction != "response_class_mean":
        raise ValueError(f"unknown region-loss reduction {reduction!r}")
    response_losses = (wrong_region_mass * valid_tokens).sum(1) / valid_tokens.sum(1)
    response_labels = labels[:, 0]
    return (response_losses * class_weights.index_select(0, response_labels)).mean()


def normalized_joint_validation_loss(metrics, reference_metrics, include_task=True):
    """Average validation losses after scaling each by one fixed reference value."""
    reference_reconstruction = reference_metrics["recon"]
    if reference_reconstruction <= 0:
        raise ValueError("the reconstruction reference loss must be positive")
    normalized_losses = [metrics["recon"] / reference_reconstruction]
    if include_task:
        reference_task = reference_metrics["task"]
        if reference_task <= 0:
            raise ValueError("the task reference loss must be positive")
        normalized_losses.append(metrics["task"] / reference_task)
    return sum(normalized_losses) / len(normalized_losses)


def compute_code_region_statistics(
    response_assignments, response_labels, num_codes, score_method, prior_strength
):
    """Compute the configured harmfulness statistic for every code."""
    return code_harmfulness_statistics(
        score_method, response_assignments, response_labels, num_codes, prior_strength
    )


def derive_code_regions(model, sequences, num_codes, device, score_method, prior_strength):
    """Recompute harmful and benign regions in the trained encoder's representation space."""
    assignments = assign_codes_for_sequences(
        model, sequences, device, assignment_space="encoded_activation"
    )
    response_assignments = split_assignments_by_response(sequences, assignments)
    statistics = compute_code_region_statistics(
        response_assignments,
        (sequence["label"] for sequence in sequences),
        num_codes,
        score_method,
        prior_strength,
    )
    harmful_codes = np.flatnonzero(statistics["signed_harmfulness"] > 0).tolist()
    harmful_code_set = set(harmful_codes)
    benign_codes = [code for code in range(num_codes) if code not in harmful_code_set]
    return (
        torch.tensor(benign_codes, device=device) if benign_codes else None,
        torch.tensor(harmful_codes, device=device) if harmful_codes else None,
        statistics,
    )


def evaluate_vq_model(
    model, sequences, batch_size, device, benign_codes=None, harmful_codes=None, temperature=1.0,
    region_loss_reduction="token_mean",
):
    """Return reconstruction, explained variance, norm ratio, and optional region loss."""
    model.eval()
    token_count = 0
    squared_error_sum = target_square_sum = relative_error_sum = norm_ratio_sum = 0.0
    target_sum = None
    region_loss_sum = region_loss_weight = 0.0
    region_loss_enabled = benign_codes is not None and harmful_codes is not None
    class_weights = (
        balanced_region_class_weights((sequence["label"] for sequence in sequences), device)
        if region_loss_enabled and region_loss_reduction == "response_class_mean" else None
    )
    with torch.no_grad():
        for batch in iter_activation_batches(sequences, batch_size, False, 0):
            target = pad_activation_sequences(batch, "y", device)
            output = model(pad_activation_sequences(batch, "x", device), device=device)
            reconstructed = output["reconstructed"].float()
            valid_tokens = target.norm(dim=2) > 1e-6
            target_tokens = target[valid_tokens]
            reconstructed_tokens = reconstructed[valid_tokens]
            squared_error = (reconstructed_tokens - target_tokens).pow(2)

            squared_error_sum += squared_error.sum().item()
            target_square_sum += target_tokens.pow(2).sum().item()
            target_sum = target_tokens.sum(0) if target_sum is None else target_sum + target_tokens.sum(0)
            relative_error_sum += (
                squared_error.sum(1) / (target_tokens.pow(2).sum(1) + 1e-8)
            ).sum().item()
            norm_ratio_sum += (
                reconstructed_tokens.norm(dim=1) / (target_tokens.norm(dim=1) + 1e-8)
            ).sum().item()
            token_count += target_tokens.shape[0]

            if region_loss_enabled:
                labels = torch.full((len(batch), output["z_e"].shape[1]), -1, dtype=torch.long, device=device)
                for row, sequence in enumerate(batch):
                    labels[row, :sequence["x"].shape[0]] = sequence["label"]
                batch_region_loss = region_separation_loss(
                    output["z_e"], labels, output["codebook_snapshot"],
                    benign_codes, harmful_codes, temperature, region_loss_reduction, class_weights,
                ).item()
                batch_weight = len(batch) if region_loss_reduction == "response_class_mean" else 1.0
                region_loss_sum += batch_region_loss * batch_weight
                region_loss_weight += batch_weight

    total_variance = target_square_sum - target_sum.pow(2).sum().item() / token_count
    return {
        "recon": relative_error_sum / token_count,
        "ev": 1.0 - squared_error_sum / (total_variance + 1e-8),
        "l2_ratio": norm_ratio_sum / token_count,
        "task": region_loss_sum / region_loss_weight if region_loss_enabled else 0.0,
    }


def print_training_config(args, device, activation_dim, num_train, num_validation, num_test):
    print("\nCONFIG", flush=True)
    print(
        f"  {device} | layer {args.read_layer} -> {args.target_layer} | dim {activation_dim} | "
        f"train {num_train}, val {num_validation}, test {num_test}", flush=True,
    )
    print(
        f"  codes {args.num_codes} | decoder {args.decoder_layers}x{args.nhead} | "
        f"batch {args.batch_size}x{args.grad_accum} | epochs {args.epochs} | lr {args.lr:g}",
        flush=True,
    )
    print(
        f"  reconstruction {args.reconstruction_loss} | task weight {args.task_weight:g} | "
        f"normalization {args.activation_normalization} | output {args.out}", flush=True,
    )


def print_epoch_row(epoch, num_codes, row, usage, num_dead_codes, seconds):
    print(
        f"epoch {epoch:3d} | train {row['train_loss']:.3f} "
        f"(recon {row['recon']:.3f}, task {row['task']:.3f}) | "
        f"val {row['val_recon']:.4f}, task {row['val_task']:.3f}, joint {row['val_joint']:.3f} | "
        f"codes {usage['active_codes']}/{num_codes}, dead {num_dead_codes}, ppl {row['perplexity']:.1f} | "
        f"grad {row['gradient_norm_mean']:.2f}/{row['gradient_norm_max']:.2f}, "
        f"clip {100 * row['gradient_clip_fraction']:.1f}% | lr {row['lr']:.1e}, {seconds:.1f}s",
        flush=True,
    )


def print_training_summary(
    best_reconstruction, best_reconstruction_epoch,
    best_validation_task, best_validation_task_epoch,
    best_joint, best_joint_epoch, joint_reference_metrics,
    checkpoint_metrics, metric_partition, total_minutes, run_dir,
    gpu_peak_gb=0.0, cpu_peak_gb=0.0,
):
    print("\nDONE", flush=True)
    print(
        f"  model.pt        best val_recon          {best_reconstruction:.4f} "
        f"@ epoch {best_reconstruction_epoch}", flush=True,
    )
    print(
        f"  model_task.pt   best val_task           {best_validation_task:.4f} "
        f"@ epoch {best_validation_task_epoch}", flush=True,
    )
    print(
        f"  model_joint.pt  best normalized joint   {best_joint:.4f} @ epoch {best_joint_epoch} "
        f"(reference recon {joint_reference_metrics['recon']:.4f}, "
        f"task {joint_reference_metrics['task']:.4f})", flush=True,
    )
    for checkpoint_name, metrics in checkpoint_metrics.items():
        if metrics is None:
            continue
        print(
            f"  {checkpoint_name:<13}{metric_partition} recon {metrics['recon']:.4f}   "
            f"EV {metrics['ev']:.3f}   L2 {metrics['l2_ratio']:.3f}   "
            f"task {metrics['task']:.3f}", flush=True,
        )
    print(f"  peak memory: GPU {gpu_peak_gb:.1f} GB (reserved)  |  CPU {cpu_peak_gb:.1f} GB", flush=True)
    print(f"  total time {total_minutes:.1f} min   ->   {run_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="run directory (default: timestamped directory under output/runs/vq)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-set", default="train")
    parser.add_argument("--activation-cache", required=True,
                        help="cache containing the read-layer activations and aligned metadata")
    parser.add_argument("--target-activation-cache",
                        help="cache containing aligned target-layer activations (default: read cache)")
    parser.add_argument("--read-layer", type=int, default=READ_LAYER,
                        help="hidden-state index used as the VQ input")
    parser.add_argument("--target-layer", type=int, default=TARGET_LAYER,
                        help="hidden-state index reconstructed by the decoder")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--num-codes", type=int, default=800, help="codebook size")
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8, help="decoder attention heads")
    parser.add_argument("--ff-mult", type=float, default=1.5,
                        help="decoder FFN width = ff_mult × dim")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="decoder dropout (0 = off; underfitting prefers off)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="micro-batch (what must fit in memory)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="micro-batches per optimizer step; effective batch = batch_size × grad_accum")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="clip the global gradient norm before each optimizer step (0 = disabled)")
    parser.add_argument("--stability-diagnostics", action="store_true",
                        help="record region churn and detailed batches when gradients spike")
    parser.add_argument("--gradient-anomaly-threshold", type=float, default=5.0,
                        help="pre-clipping gradient norm that triggers a detailed diagnostic record")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="stop after this zero-indexed epoch while retaining the full LR schedule")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.1,
                        help="fraction of total steps for LR warmup")
    parser.add_argument("--lr-floor", type=float, default=0.1,
                        help="cosine decays to lr_floor*peak (not 0)")
    parser.add_argument("--commitment-cost", type=float,
                        default=0.1, help="VQ commitment weight (beta)")
    parser.add_argument("--perplexity-weight", type=float,
                        default=0.01, help="codebook-utilization loss weight")
    parser.add_argument("--reconstruction-loss", choices=["relative_mse", "robust_relative_huber"],
                        default="relative_mse", help="per-token reconstruction objective")
    parser.add_argument("--activation-normalization", choices=["none", "fixed_rms"], default="none",
                        help="optionally divide read/target activations by train-partition scalar RMS")
    parser.add_argument("--use-sampling", type=int,
                        choices=[0, 1], default=1,
                        help="1 = stochastic top-k assignment during training; 0 = greedy nearest")
    parser.add_argument("--temperature", type=float, default=1.0, help="VQ top-k sampler softmax temperature")
    parser.add_argument("--top-k", type=int, default=10,
                        help="VQ sampler: sample among the nearest top_k codes")
    parser.add_argument("--task-weight", type=float, default=1.0,
                        help="lambda on the H/B task loss (0 = off)")
    parser.add_argument("--task-tau", type=float, default=1.0,
                        help="softmax temperature in the H/B task loss (per-dim distance)")
    parser.add_argument(
        "--region-loss-reduction", choices=["token_mean", "response_class_mean"],
        default="token_mean",
        help="average H/B task loss over tokens, or equally over responses and response classes",
    )
    parser.add_argument("--code-score-prior-strength", type=float, default=10.0,
                        help="response-equivalent prior for smoothed harmful-code region scores")
    parser.add_argument("--region-score-method",
                        choices=list(CODE_SCORE_METHODS),
                        default="response_presence",
                        help="how code occurrences are counted when defining harmful/benign regions")
    parser.add_argument("--region-recompute-every",
                        type=int, default=1,
                        help="re-derive H/B task-loss regions every N epochs (0 = keep initial regions)")
    parser.add_argument("--code-repr",
                        choices=["token", "word", "phrase"], default="phrase",
                        help="render init examples as a token, whole word, or context phrase")
    parser.add_argument("--context-len", type=int, default=4,
                        help="±tokens of context for phrase examples")
    parser.add_argument("--patience", type=int, default=15, help="early-stop patience on val reconstruction")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_optimizer(model, lr, weight_decay):
    """AdamW with weight decay on encoder/decoder weights, none on biases/norms (and the EMA codebook)."""
    encoder_params, decoder_params, nodecay_params = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            nodecay_params.append(parameter)
        elif "_ContinuousEmbedding" in name:
            encoder_params.append(parameter)
        else:
            decoder_params.append(parameter)
    parameter_groups = [
        {"params": encoder_params, "weight_decay": weight_decay},
        {"params": decoder_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(parameter_groups, lr=lr)


def main():
    args = parse_args()

    # Load the requested read and target activation caches.
    training_data = training_split(args.train_set or "train", args.dataset)
    args.train_set = training_data.name
    training_cache = Path(args.activation_cache)
    target_activation_cache = Path(args.target_activation_cache or args.activation_cache)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = (
        Path(args.out) if args.out
        else VQ_RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_vqvae_spherical_K{args.num_codes}"
    )
    args.out = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = run_dir / "stability_diagnostics.jsonl"
    if args.stability_diagnostics and diagnostic_path.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostics: {diagnostic_path}")
    started_at = time.time()

    # Keep detector validation responses outside the VQ train, validation, and test partitions.
    sequences = load_cross_layer_sequences(
        training_cache, target_activation_cache, args.read_layer, args.target_layer
    )
    activation_dim = sequences[0]["x"].shape[1]
    detector_training, _ = train_validation_split(sequences, args.seed)
    train, holdout = train_validation_split(detector_training, args.seed)
    validation, test = train_validation_split(holdout, args.seed, validation_fraction=0.5)
    if args.activation_normalization == "fixed_rms":
        args.read_activation_scale = activation_rms_scale(train, "x")
        args.target_activation_scale = activation_rms_scale(train, "y")
    else:
        args.read_activation_scale = 1.0
        args.target_activation_scale = 1.0
    print_training_config(args, device, activation_dim, len(train), len(validation), len(test))

    # Initialize the encoder, codebook, and decoder before defining concept regions.
    print("\nINIT (spherical k-means)", flush=True)
    initialization_started_at = time.time()
    model = CrossLayerVQVAE(
        args.num_codes, activation_dim, decoder_layers=args.decoder_layers,
        nhead=args.nhead, ff_mult=args.ff_mult,
        dropout=args.dropout, commitment_cost=args.commitment_cost, perplexity_weight=args.perplexity_weight,
        use_sampling=bool(args.use_sampling), temperature=args.temperature, top_k=args.top_k,
        activation_normalization=args.activation_normalization,
        read_activation_scale=args.read_activation_scale,
        target_activation_scale=args.target_activation_scale,
    ).to(device)
    quantizer = model.quantizer
    init_info = initialize_codebook_from_sequences(model, train, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR / args.base_model), trust_remote_code=True
    )
    checkpoint_config = {
        **vars(args),
        "training_cache": str(training_cache),
        "target_activation_cache": str(target_activation_cache),
        "read_layer": args.read_layer,
        "target_layer": args.target_layer,
        "init": "spherical",
        "activation_loading": "preloaded_cpu",
        "activation_normalization_source": (
            "vq_train_partition" if args.activation_normalization == "fixed_rms" else None
        ),
        "activation_encoder_projection_init": "pytorch_default",
        "codebook_initialization_space": "raw_activation",
        "initial_region_assignment_space": "raw_activation",
        "recomputed_region_assignment_space": "encoded_activation",
        "region_score_method": args.region_score_method,
        "ema_assignment_order": "current_batch_uses_pre_update_snapshot_next_batch_uses_ema_update",
        "split_scheme": "nested_90_5_5",
        "split_partition_counts": {
            "vq_train": len(train),
            "vq_validation": len(validation),
            "vq_test": len(test),
        },
        "outer_detector_validation_excluded": True,
    }
    initialization_checkpoint = {
        "codebook": quantizer.codebook.data.cpu(),
        "init_info": init_info,
        "config": checkpoint_config,
    }
    torch.save(initialization_checkpoint, run_dir / "codebook_init.pt")
    stats, stats_summary, enrichment_regions = summarize_initial_codebook(
        model, train, init_info, tokenizer, device, args.region_score_method,
        args.code_score_prior_strength,
        args.code_repr, args.context_len
    )
    (run_dir / "init_stats.json").write_text(
        json.dumps(stats, indent=2, default=float, ensure_ascii=False) + "\n"
    )
    print(stats_summary, flush=True)
    init_time = time.time() - initialization_started_at
    print(f"  init time {init_time:.1f}s", flush=True)

    # Split the initialized codebook into benign and harmful regions for the task loss.
    regions = enrichment_regions
    current_region_statistics = enrichment_regions
    current_region_source = regions["source"]
    current_region_assignment_space = "raw_activation"
    benign_idx = torch.tensor(regions["benign_codes"], device=device) if regions["benign_codes"] else None
    harmful_idx = torch.tensor(regions["harmful_codes"], device=device) if regions["harmful_codes"] else None
    task_enabled = args.task_weight > 0 and benign_idx is not None and harmful_idx is not None
    if args.task_weight > 0 and not task_enabled:
        print("  WARNING: a code region is empty; task loss is disabled", flush=True)
    elif task_enabled:
        print(
            f"  task-loss regions: {len(regions['benign_codes'])} benign / "
            f"{len(regions['harmful_codes'])} harmful (source={regions['source']})",
            flush=True,
        )
    training_region_class_weights = (
        balanced_region_class_weights((sequence["label"] for sequence in train), device)
        if task_enabled and args.region_loss_reduction == "response_class_mean" else None
    )

    previous_harmful_codes = set(regions["harmful_codes"])
    current_region_diagnostic = summarize_region_changes(
        current_region_statistics, previous_harmful_codes
    )
    if args.stability_diagnostics:
        append_diagnostic_record(
            diagnostic_path, {"type": "regions", "epoch": 0, **current_region_diagnostic}
        )

    joint_reference_metrics = evaluate_vq_model(
        model, validation, args.batch_size, device, benign_idx, harmful_idx, args.task_tau,
        args.region_loss_reduction,
    )
    normalized_joint_validation_loss(
        joint_reference_metrics, joint_reference_metrics, include_task=task_enabled
    )
    print(
        f"  joint-selection reference: val_recon {joint_reference_metrics['recon']:.4f} | "
        f"val_task {joint_reference_metrics['task']:.4f}", flush=True,
    )

    # Region scores remain detector and steering features when the region loss is ablated. Refreshing
    # them does not affect optimization at task_weight=0, but avoids saving initialization-time scores.
    recompute_regions = (
        benign_idx is not None and harmful_idx is not None and args.region_recompute_every > 0
    )
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    micro_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    opt_steps_per_epoch = (micro_per_epoch + args.grad_accum - 1) // args.grad_accum
    total_steps = args.epochs * opt_steps_per_epoch
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, int(args.warmup_frac * total_steps), total_steps, min_lr_rate=args.lr_floor
    )

    # Train the VQ model and retain checkpoints for each validation criterion.
    print("\nTRAINING", flush=True)
    best_reconstruction, best_reconstruction_epoch = float("inf"), -1
    best_vtask, best_vtask_epoch = float("inf"), -1
    best_joint, best_joint_epoch = float("inf"), -1
    epochs_no_improve, history = 0, []

    for epoch in range(args.epochs):
        epoch_started_at = time.time()
        model.train()
        quantizer.reset_usage_stats()
        sums = {
            "loss": 0.0,
            "recon": 0.0,
            "task": 0.0,
            "commit": 0.0,
            "ppl_loss": 0.0,
            "perplexity": 0.0,
            "resets": 0,
            "gradient_norm": 0.0,
            "gradient_norm_max": 0.0,
            "clipped_steps": 0,
        }
        optimizer_steps = 0
        batch_count = 0
        output = None
        accumulated_diagnostic_batches = []
        optimizer.zero_grad()

        for batch_number, batch in enumerate(
            iter_activation_batches(train, args.batch_size, True, args.seed + epoch)
        ):
            read_activations = pad_activation_sequences(batch, "x", device)
            target = pad_activation_sequences(batch, "y", device)
            labels = torch.full((len(batch), read_activations.shape[1]), -1, dtype=torch.long, device=device)
            for row, sequence in enumerate(batch):
                labels[row, :sequence["x"].shape[0]] = sequence["label"]

            codebook_before = quantizer.codebook.detach().clone() if args.stability_diagnostics else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                output = model(read_activations, device=device)

            reconstruction_loss = relative_reconstruction_loss(
                output["reconstructed"], target, args.reconstruction_loss
            )
            region_loss = region_separation_loss(
                output["z_e"], labels, output["codebook_snapshot"], benign_idx, harmful_idx, args.task_tau,
                args.region_loss_reduction, training_region_class_weights,
            ) if task_enabled else read_activations.new_zeros(())
            loss = reconstruction_loss + output["loss"] + args.task_weight * region_loss

            if args.stability_diagnostics:
                batch_diagnostic = summarize_training_batch(
                    batch, read_activations, target, labels, output, quantizer, codebook_before
                )
                batch_diagnostic["losses"].update({
                    "optimized_reconstruction": reconstruction_loss.detach().item(),
                    "task": region_loss.detach().item(), "total": loss.detach().item()
                })
                accumulated_diagnostic_batches.append(batch_diagnostic)

            accumulation_size = accumulation_window_size(batch_number, micro_per_epoch, args.grad_accum)
            (loss / accumulation_size).backward()

            if (batch_number + 1) % args.grad_accum == 0 or (batch_number + 1) == micro_per_epoch:
                component_gradient_norms = (
                    gradient_norms_by_component(model) if args.stability_diagnostics else {}
                )
                if args.max_grad_norm > 0:
                    clipped_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    gradient_norm = float(clipped_norm)
                    sums["gradient_norm"] += gradient_norm
                    sums["gradient_norm_max"] = max(sums["gradient_norm_max"], gradient_norm)
                    sums["clipped_steps"] += int(gradient_norm > args.max_grad_norm)
                elif args.stability_diagnostics:
                    squared_norms = sum(
                        value * value for value in component_gradient_norms.values()
                    )
                    gradient_norm = math.sqrt(squared_norms)
                if args.stability_diagnostics and (
                    not math.isfinite(gradient_norm) or gradient_norm >= args.gradient_anomaly_threshold
                ):
                    append_diagnostic_record(diagnostic_path, {
                        "type": "gradient_anomaly", "epoch": epoch,
                        "optimizer_step_in_epoch": optimizer_steps,
                        "global_step": epoch * opt_steps_per_epoch + optimizer_steps,
                        "pre_clip_gradient_norm": gradient_norm,
                        "component_gradient_norms": component_gradient_norms,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "batches": accumulated_diagnostic_batches,
                    })
                optimizer_steps += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulated_diagnostic_batches = []

            sums["loss"] += loss.item()
            sums["recon"] += reconstruction_loss.item()
            sums["task"] += region_loss.item()
            sums["commit"] += output["commit_loss"].item()
            sums["ppl_loss"] += output["perplexity_loss"].item()
            sums["perplexity"] += output["perplexity"].item()
            sums["resets"] += output["n_reset_codes"]
            batch_count += 1
        if device == "cuda":
            torch.cuda.empty_cache()

        # Refresh regions after training so every saved checkpoint carries statistics from its own state.
        if recompute_regions and (epoch + 1) % args.region_recompute_every == 0:
            benign_region, harmful_region, region_statistics = derive_code_regions(
                model, train, args.num_codes, device, args.region_score_method,
                args.code_score_prior_strength,
            )
            if benign_region is not None and harmful_region is not None:
                benign_idx, harmful_idx = benign_region, harmful_region
                current_region_statistics = region_statistics
                current_region_source = code_score_region_source(args.region_score_method)
                current_region_assignment_space = "encoded_activation"
                new_harmful_codes = set(harmful_idx.detach().cpu().tolist())
                current_region_diagnostic = summarize_region_changes(
                    region_statistics, new_harmful_codes, previous_harmful_codes
                )
                previous_harmful_codes = new_harmful_codes
                if args.stability_diagnostics:
                    append_diagnostic_record(diagnostic_path, {
                        "type": "regions", "epoch": epoch, **current_region_diagnostic
                    })

        # Measure validation losses and codebook use after the epoch update.
        encoder = model.activation_encoder
        if hasattr(encoder, "alpha"):
            alpha = encoder.alpha if encoder.is_fixed else torch.sigmoid(encoder.alpha)
            encoder_mix = float((alpha * 0.5).detach().item())
        else:
            encoder_mix = 0.0
        usage = model.get_codebook_usage()
        codebook = quantizer.codebook.data
        codebook_norms = codebook.norm(dim=1)
        max_norm_code = int(codebook_norms.argmax())
        benign_norm = codebook[benign_idx].norm(dim=1).mean().item() if benign_idx is not None else 0.0
        harmful_norm = codebook[harmful_idx].norm(dim=1).mean().item() if harmful_idx is not None else 0.0
        val_metrics = evaluate_vq_model(
            model, validation, args.batch_size, device, benign_idx, harmful_idx, args.task_tau,
            args.region_loss_reduction,
        )
        joint_validation_loss = normalized_joint_validation_loss(
            val_metrics, joint_reference_metrics, include_task=task_enabled
        )
        gradient_norm_mean = 0.0
        gradient_clip_fraction = 0.0
        if args.max_grad_norm > 0:
            gradient_norm_mean = sums["gradient_norm"] / optimizer_steps
            gradient_clip_fraction = sums["clipped_steps"] / optimizer_steps

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "sec": round(time.time() - epoch_started_at, 1),
            "train_loss": sums["loss"] / batch_count,
            "recon": sums["recon"] / batch_count,
            "task": sums["task"] / batch_count,
            "commit": sums["commit"] / batch_count,
            "perplexity_loss": sums["ppl_loss"] / batch_count,
            "perplexity": sums["perplexity"] / batch_count,
            "gradient_norm_mean": gradient_norm_mean,
            "gradient_norm_max": sums["gradient_norm_max"],
            "gradient_clip_fraction": gradient_clip_fraction,
            "encoder_mix": encoder_mix,
            "val_recon": val_metrics["recon"],
            "val_ev": val_metrics["ev"],
            "val_l2_ratio": val_metrics["l2_ratio"],
            "val_task": val_metrics["task"],
            "val_joint": joint_validation_loss,
            "active_codes": usage["active_codes"],
            "dead_codes": output["n_dead_codes"],
            "code_resets": sums["resets"],
            "b_norm": benign_norm,
            "h_norm": harmful_norm,
            "max_code_norm": float(codebook_norms[max_norm_code]),
            "max_code_norm_id": max_norm_code,
            "region_changes": current_region_diagnostic["n_changed_codes"],
            "near_boundary_codes": current_region_diagnostic["n_scores_within_0.01_of_boundary"],
        }
        history.append(row)
        print_epoch_row(epoch, args.num_codes, row, usage, output["n_dead_codes"], row["sec"])
        if args.stability_diagnostics:
            append_diagnostic_record(diagnostic_path, {"type": "epoch", **row})

        # Save the best reconstruction, region-loss, and joint checkpoints independently.
        checkpoint_data = {
            "model": model.state_dict(),
            "codebook": codebook.cpu(),
            "init_info": init_info,
            "config": checkpoint_config,
            "regions": {
                "benign_codes": benign_idx.detach().cpu().tolist() if benign_idx is not None else [],
                "harmful_codes": harmful_idx.detach().cpu().tolist() if harmful_idx is not None else [],
                "source": current_region_source,
                "assignment_space": current_region_assignment_space,
                "score_method": args.region_score_method,
                "prior_strength": args.code_score_prior_strength,
                "signed_harmfulness": np.asarray(
                    current_region_statistics["signed_harmfulness"], dtype=np.float32
                ).tolist(),
            },
        }
        reconstruction_improved = val_metrics["recon"] < best_reconstruction - 1e-5
        if reconstruction_improved:
            best_reconstruction, best_reconstruction_epoch, epochs_no_improve = (
                val_metrics["recon"], epoch, 0
            )
            torch.save(checkpoint_data, run_dir / "model.pt")
        else:
            epochs_no_improve += 1
        if val_metrics["task"] < best_vtask - 1e-5:
            best_vtask, best_vtask_epoch = val_metrics["task"], epoch
            torch.save(checkpoint_data, run_dir / "model_task.pt")
        if joint_validation_loss < best_joint - 1e-5:
            best_joint, best_joint_epoch = joint_validation_loss, epoch
            torch.save(checkpoint_data, run_dir / "model_joint.pt")
        if args.patience and epochs_no_improve >= args.patience:
            print(f"  early stop @ epoch {epoch}", flush=True)
            break
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            print(f"  diagnostic stop after epoch {epoch}", flush=True)
            break

    # Evaluate each retained checkpoint on the held-out VQ test partition.
    evaluation_sequences = test if test else validation
    metric_partition = "test" if test else "validation"

    def evaluate_checkpoint(checkpoint_name):
        checkpoint_path = run_dir / checkpoint_name
        if not checkpoint_path.exists():
            return None
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_regions = checkpoint.get("regions", {})
        checkpoint_benign = checkpoint_regions.get("benign_codes", [])
        checkpoint_harmful = checkpoint_regions.get("harmful_codes", [])
        checkpoint_benign = torch.tensor(checkpoint_benign, device=device) if checkpoint_benign else None
        checkpoint_harmful = torch.tensor(checkpoint_harmful, device=device) if checkpoint_harmful else None
        return evaluate_vq_model(
            model, evaluation_sequences, args.batch_size, device,
            checkpoint_benign, checkpoint_harmful, args.task_tau, args.region_loss_reduction,
        )

    model_checkpoint_metrics = evaluate_checkpoint("model.pt")
    task_model_checkpoint_metrics = evaluate_checkpoint("model_task.pt")
    joint_model_checkpoint_metrics = evaluate_checkpoint("model_joint.pt")
    total_minutes = (time.time() - started_at) / 60
    gpu_peak_gb = torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else 0.0
    cpu_peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    checkpoint_metrics = {
        "model.pt": model_checkpoint_metrics,
        "model_task.pt": task_model_checkpoint_metrics,
        "model_joint.pt": joint_model_checkpoint_metrics,
    }
    summary = {
        "best_reconstruction": best_reconstruction,
        "best_reconstruction_epoch": best_reconstruction_epoch,
        "best_task": best_vtask,
        "best_task_epoch": best_vtask_epoch,
        "best_joint": best_joint,
        "best_joint_epoch": best_joint_epoch,
        "joint_selection": {
            "criterion": "mean(val_recon/reference_recon, val_task/reference_task)",
            "reference_source": "initialized model before training",
            "reference_metrics": joint_reference_metrics,
            "task_included": task_enabled,
        },
        "test_recon": model_checkpoint_metrics["recon"] if test else None,
        "test_ev": model_checkpoint_metrics["ev"] if test else None,
        "test_l2_ratio": model_checkpoint_metrics["l2_ratio"] if test else None,
        "test_metrics": checkpoint_metrics if test else {name: None for name in checkpoint_metrics},
        "checkpoint_validation_metrics": checkpoint_metrics if not test else None,
        "checkpoint_metric_partition": metric_partition,
        "init_time_s": round(init_time, 1),
        "total_time_min": round(total_minutes, 1),
        "gpu_peak_gb": round(gpu_peak_gb, 1),
        "cpu_peak_gb": round(cpu_peak_gb, 1),
        "init_info": init_info,
        "config": checkpoint_config,
        "history": history,
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    print_training_summary(
        best_reconstruction, best_reconstruction_epoch, best_vtask, best_vtask_epoch,
        best_joint, best_joint_epoch, joint_reference_metrics, checkpoint_metrics,
        metric_partition, total_minutes, run_dir, gpu_peak_gb, cpu_peak_gb,
    )


if __name__ == "__main__":
    main()
