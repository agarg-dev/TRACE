#!/usr/bin/env python
"""Train the cross-layer VQ model from cached response activations.

The nested split keeps detector validation responses out of VQ training.
Training writes the selected checkpoints and a summary to the run directory.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_with_min_lr_schedule_with_warmup

from activations.activation_cache import (
    load_cross_layer_sequences,
    resolve_vq_activation_caches,
)
from data.dataset_splits import DETECTOR_TRAIN_SET_BY_DATASET, train_validation_split, training_split
from model_inputs import iter_activation_batches, pad_activation_sequences
from project_config import DEFAULT_BASE_MODEL, DEFAULT_DATASET, READ_LAYER, TARGET_LAYER, VQ_RUNS_DIR
from vq.codebook import (
    assign_codes_for_sequences,
    initialize_spherical_codebook,
    smoothed_response_code_statistics,
    split_assignments_by_response,
)
from vq.training_summary import (
    summarize_initial_codebook,
    summarize_region_changes,
)
from vq.model import CrossLayerVQVAE


def accumulation_window_size(batch_number, num_batches, grad_accum):
    """Count microbatches in this update, including an incomplete final window."""
    window_start = batch_number - batch_number % grad_accum
    return min(grad_accum, num_batches - window_start)


def relative_reconstruction_loss(reconstructed, target):
    """Mean per-token relative reconstruction loss over non-padding tokens."""
    reconstructed = reconstructed.float()
    valid_tokens = target.norm(dim=2) > 1e-6
    relative_error = (reconstructed - target).pow(2).sum(dim=2) / (target.pow(2).sum(dim=2) + 1e-8)
    return relative_error[valid_tokens].mean()


def region_separation_loss(
    encoded, labels, codebook, benign_codes, harmful_codes, temperature,
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
    return wrong_region_mass[valid_tokens].mean()


def normalized_joint_validation_loss(metrics, reference_metrics, include_task=True):
    """Average validation losses after scaling each by one fixed reference value."""
    normalized_losses = [metrics["recon"] / reference_metrics["recon"]]
    if include_task:
        normalized_losses.append(metrics["task"] / reference_metrics["task"])
    return sum(normalized_losses) / len(normalized_losses)


def derive_code_regions(model, sequences, num_codes, device, prior_strength):
    """Recompute harmful and benign regions in the trained encoder's representation space."""
    assignments = assign_codes_for_sequences(
        model, sequences, device, assignment_space="encoded_activation"
    )
    response_assignments = split_assignments_by_response(sequences, assignments)
    statistics = smoothed_response_code_statistics(
        response_assignments,
        (sequence["label"] for sequence in sequences),
        num_codes,
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
):
    """Return reconstruction, explained variance, norm ratio, and optional region loss."""
    model.eval()
    token_count = 0
    squared_error_sum = target_square_sum = relative_error_sum = norm_ratio_sum = 0.0
    target_sum = None
    region_loss_sum = region_loss_weight = 0.0
    region_loss_enabled = benign_codes is not None and harmful_codes is not None
    with torch.no_grad():
        for batch in iter_activation_batches(sequences, batch_size, False, 0):
            target = pad_activation_sequences(batch, "y", device)
            output = model(pad_activation_sequences(batch, "x", device))
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
                    benign_codes, harmful_codes, temperature,
                ).item()
                region_loss_sum += batch_region_loss
                region_loss_weight += 1

    total_variance = target_square_sum - target_sum.pow(2).sum().item() / token_count
    return {
        "recon": relative_error_sum / token_count,
        "ev": 1.0 - squared_error_sum / (total_variance + 1e-8),
        "l2_ratio": norm_ratio_sum / token_count,
        "task": region_loss_sum / region_loss_weight if region_loss_enabled else 0.0,
    }


def print_training_config(args, device, activation_dim, num_train, num_validation, num_test):
    print(
        f"device={device}  dim={activation_dim}  layers={args.read_layer}->{args.target_layer}  "
        f"responses={num_train}/{num_validation}/{num_test}", flush=True,
    )
    print(
        f"codes={args.K}  decoder={args.decoder_layers}x{args.nhead}  epochs={args.epochs}  "
        f"batch={args.batch_size}x{args.grad_accum}  lr={args.lr:g}  seed={args.seed}", flush=True,
    )
    print(
        f"commit={args.commitment_cost:g}  perplexity={args.perplexity_weight:g}  "
        f"task={args.task_weight:g}  output={args.out}", flush=True,
    )


def print_epoch(epoch, row, num_codes):
    print(
        f"epoch {epoch:3d} | train {row['train_loss']:.3f} "
        f"(recon {row['recon']:.3f}, task {row['task']:.3f}) | "
        f"val {row['val_recon']:.4f}, task {row['val_task']:.3f}, joint {row['val_joint']:.3f} | "
        f"codes {row['active_codes']}/{num_codes}, dead {row['dead_codes']}, ppl {row['perplexity']:.1f} | "
        f"lr {row['lr']:.1e}, {row['sec']:.1f}s",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="run directory (default: timestamped directory under output/runs/vq)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-set", default=None)
    parser.add_argument("--activation-cache",
                        help="cache containing the read-layer activations and response metadata")
    parser.add_argument("--target-activation-cache",
                        help="cache containing target-layer activations (default: read cache)")
    parser.add_argument("--read-layer", type=int, default=READ_LAYER,
                        help="hidden-state index used as the VQ input")
    parser.add_argument("--target-layer", type=int, default=TARGET_LAYER,
                        help="hidden-state index reconstructed by the decoder")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--num-codes", "--K", dest="K", type=int, default=800, help="codebook size")
    parser.add_argument("--decoder-layers", "--decoder_layers", dest="decoder_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8, help="decoder attention heads")
    parser.add_argument("--ff-mult", "--ff_mult", dest="ff_mult", type=float, default=1.5,
                        help="decoder FFN width = ff_mult × dim")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="decoder dropout (0 = off; underfitting prefers off)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16,
                        help="micro-batch (what must fit in memory)")
    parser.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=1,
                        help="micro-batches per optimizer step; effective batch = batch_size × grad_accum")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="clip the global gradient norm before each optimizer step (0 = disabled)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", "--warmup_frac", dest="warmup_frac", type=float, default=0.1,
                        help="fraction of total steps for LR warmup")
    parser.add_argument("--lr-floor", "--lr_floor", dest="lr_floor", type=float, default=0.1,
                        help="cosine decays to lr_floor*peak (not 0)")
    parser.add_argument("--commitment-cost", "--commitment_cost", dest="commitment_cost", type=float,
                        default=0.1, help="VQ commitment weight (beta)")
    parser.add_argument("--perplexity-weight", "--perplexity_weight", dest="perplexity_weight", type=float,
                        default=0.01, help="codebook-utilization loss weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="VQ top-k sampler softmax temperature")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=10,
                        help="VQ sampler: sample among the nearest top_k codes")
    parser.add_argument("--task-weight", "--task_weight", dest="task_weight", type=float, default=1.0,
                        help="lambda on the H/B task loss (0 = off)")
    parser.add_argument("--task-tau", "--task_tau", dest="task_tau", type=float, default=1.0,
                        help="softmax temperature in the H/B task loss (per-dim distance)")
    parser.add_argument("--code-score-prior-strength", type=float, default=10.0,
                        help="response-equivalent prior for smoothed harmful-code region scores")
    parser.add_argument("--region-recompute-every", "--region_recompute_every", dest="region_recompute_every",
                        type=int, default=1,
                        help="re-derive H/B task-loss regions every N epochs (0 = keep initial regions)")
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
    args.train_set = args.train_set or DETECTOR_TRAIN_SET_BY_DATASET[args.dataset]
    training_data = training_split(args.train_set, args.dataset)
    args.train_set = training_data.name
    read_cache, target_cache = resolve_vq_activation_caches(
        args.dataset,
        training_data,
        read_path=args.activation_cache,
        target_path=args.target_activation_cache,
    )
    training_cache = read_cache
    target_activation_cache = target_cache
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = (
        Path(args.out) if args.out
        else VQ_RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_vqvae_K{args.K}"
    )
    args.out = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    sequences = load_cross_layer_sequences(
        training_cache, target_activation_cache, args.read_layer, args.target_layer
    )
    activation_dim = sequences[0]["x"].shape[1]
    detector_training, _ = train_validation_split(sequences, args.seed)
    train, holdout = train_validation_split(detector_training, args.seed)
    validation, test = train_validation_split(holdout, args.seed, validation_fraction=0.5)
    print_training_config(args, device, activation_dim, len(train), len(validation), len(test))

    print("\nInitializing spherical codebook", flush=True)
    initialization_started_at = time.time()
    model = CrossLayerVQVAE(
        args.K, activation_dim, decoder_layers=args.decoder_layers, nhead=args.nhead, ff_mult=args.ff_mult,
        dropout=args.dropout, commitment_cost=args.commitment_cost, perplexity_weight=args.perplexity_weight,
        temperature=args.temperature, top_k=args.top_k,
    ).to(device)
    quantizer = model.quantizer
    init_info = initialize_spherical_codebook(model, train, args.seed)
    checkpoint_config = {
        **vars(args), "training_cache": str(training_cache),
        "target_activation_cache": str(target_activation_cache),
        "read_layer": args.read_layer, "target_layer": args.target_layer,
        "split_scheme": "nested_90_5_5",
        "split_partition_counts": {
            "vq_train": len(train), "vq_validation": len(validation), "vq_test": len(test)
        },
    }
    torch.save({
        "codebook": quantizer.codebook.detach().cpu(), "init_info": init_info,
        "config": checkpoint_config,
    }, run_dir / "codebook_init.pt")
    stats, stats_summary, enrichment_regions = summarize_initial_codebook(
        model, train, init_info, device, args.code_score_prior_strength
    )
    json.dump(stats, open(run_dir / "init_stats.json", "w"), indent=2, default=float, ensure_ascii=False)
    print(stats_summary, flush=True)
    init_time = time.time() - initialization_started_at
    print(f"  init time {init_time:.1f}s", flush=True)

    regions = enrichment_regions
    region_statistics = enrichment_regions
    region_source = regions["source"]
    region_assignment_space = "raw_activation"
    benign_codes = torch.tensor(regions["benign_codes"], device=device) if regions["benign_codes"] else None
    harmful_codes = torch.tensor(regions["harmful_codes"], device=device) if regions["harmful_codes"] else None
    task_enabled = args.task_weight > 0 and benign_codes is not None and harmful_codes is not None
    if args.task_weight > 0 and not task_enabled:
        print("  WARNING: a code region is empty; task loss is disabled", flush=True)
    elif task_enabled:
        print(f"  task-loss regions: {len(regions['benign_codes'])} benign / {len(regions['harmful_codes'])} harmful "
              f"(source={regions['source']})", flush=True)
    previous_harmful_codes = set(regions["harmful_codes"])
    region_diagnostic = summarize_region_changes(
        region_statistics, previous_harmful_codes
    )

    joint_reference_metrics = evaluate_vq_model(
        model, validation, args.batch_size, device, benign_codes, harmful_codes, args.task_tau,
    )
    print(
        f"  joint-selection reference: val_recon {joint_reference_metrics['recon']:.4f} | "
        f"val_task {joint_reference_metrics['task']:.4f}", flush=True,
    )

    # Region scores remain detector and steering features when the region loss is ablated. Refreshing
    # them does not affect optimization at task_weight=0, but avoids saving initialization-time scores.
    recompute_regions = (
        benign_codes is not None and harmful_codes is not None and args.region_recompute_every > 0
    )
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    micro_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    opt_steps_per_epoch = (micro_per_epoch + args.grad_accum - 1) // args.grad_accum
    total_steps = args.epochs * opt_steps_per_epoch
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, int(args.warmup_frac * total_steps), total_steps, min_lr_rate=args.lr_floor
    )

    print("\nTraining", flush=True)
    best_reconstruction, best_reconstruction_epoch = float("inf"), -1
    best_vtask, best_vtask_epoch = float("inf"), -1
    best_joint, best_joint_epoch = float("inf"), -1
    epochs_no_improve, history = 0, []
    for epoch in range(args.epochs):
        epoch_started_at = time.time()
        model.train()
        quantizer.reset_usage_stats()
        sums = {"loss": 0.0, "recon": 0.0, "task": 0.0, "commit": 0.0,
                "ppl_loss": 0.0, "perplexity": 0.0, "resets": 0,
                "gradient_norm": 0.0, "gradient_norm_max": 0.0, "clipped_steps": 0}
        optimizer_steps = 0
        batch_count, output = 0, None
        optimizer.zero_grad()
        for batch_number, batch in enumerate(
            iter_activation_batches(train, args.batch_size, True, args.seed + epoch)
        ):
            read_activations = pad_activation_sequences(batch, "x", device)
            target = pad_activation_sequences(batch, "y", device)
            labels = torch.full((len(batch), read_activations.shape[1]), -1, dtype=torch.long, device=device)
            for row, sequence in enumerate(batch):
                labels[row, :sequence["x"].shape[0]] = sequence["label"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                output = model(read_activations)
            reconstruction_loss = relative_reconstruction_loss(output["reconstructed"], target)
            region_loss = region_separation_loss(
                output["z_e"], labels, output["codebook_snapshot"], benign_codes, harmful_codes, args.task_tau,
            ) if task_enabled else read_activations.new_zeros(())
            loss = reconstruction_loss + output["loss"] + args.task_weight * region_loss
            accumulation_size = accumulation_window_size(batch_number, micro_per_epoch, args.grad_accum)
            (loss / accumulation_size).backward()
            if (batch_number + 1) % args.grad_accum == 0 or (batch_number + 1) == micro_per_epoch:
                if args.max_grad_norm > 0:
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
                    sums["gradient_norm"] += gradient_norm
                    sums["gradient_norm_max"] = max(sums["gradient_norm_max"], gradient_norm)
                    sums["clipped_steps"] += int(gradient_norm > args.max_grad_norm)
                optimizer_steps += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
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
            benign_region, harmful_region, new_region_statistics = derive_code_regions(
                model, train, args.K, device, args.code_score_prior_strength,
            )
            if benign_region is not None and harmful_region is not None:
                benign_codes, harmful_codes = benign_region, harmful_region
                region_statistics = new_region_statistics
                region_source = "smoothed_response_enrichment"
                region_assignment_space = "encoded_activation"
                new_harmful_codes = set(harmful_codes.detach().cpu().tolist())
                region_diagnostic = summarize_region_changes(
                    new_region_statistics, new_harmful_codes, previous_harmful_codes
                )
                previous_harmful_codes = new_harmful_codes

        encoder = model.activation_encoder
        encoder_mix = float((torch.sigmoid(encoder.alpha) * 0.5).detach().item())
        usage = model.get_codebook_usage()
        codebook = quantizer.codebook.detach()
        codebook_norms = codebook.norm(dim=1)
        max_norm_code = int(codebook_norms.argmax())
        benign_norm = codebook[benign_codes].norm(dim=1).mean().item() if benign_codes is not None else 0.0
        harmful_norm = codebook[harmful_codes].norm(dim=1).mean().item() if harmful_codes is not None else 0.0
        val_metrics = evaluate_vq_model(
            model, validation, args.batch_size, device, benign_codes, harmful_codes, args.task_tau,
        )
        joint_loss = normalized_joint_validation_loss(
            val_metrics, joint_reference_metrics, include_task=task_enabled
        )
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "sec": round(time.time() - epoch_started_at, 1),
               "train_loss": sums["loss"] / batch_count, "recon": sums["recon"] / batch_count,
               "task": sums["task"] / batch_count, "commit": sums["commit"] / batch_count,
               "perplexity_loss": sums["ppl_loss"] / batch_count,
               "perplexity": sums["perplexity"] / batch_count,
               "gradient_norm_mean": sums["gradient_norm"] / optimizer_steps if args.max_grad_norm > 0 else 0.0,
               "gradient_norm_max": sums["gradient_norm_max"],
               "gradient_clip_fraction": sums["clipped_steps"] / optimizer_steps if args.max_grad_norm > 0 else 0.0,
               "encoder_mix": encoder_mix, "val_recon": val_metrics["recon"], "val_ev": val_metrics["ev"],
               "val_l2_ratio": val_metrics["l2_ratio"], "val_task": val_metrics["task"],
               "val_joint": joint_loss, "active_codes": usage["active_codes"],
               "dead_codes": output["n_dead_codes"], "code_resets": sums["resets"],
               "b_norm": benign_norm, "h_norm": harmful_norm,
               "max_code_norm": float(codebook_norms[max_norm_code]), "max_code_norm_id": max_norm_code,
               "region_changes": region_diagnostic["n_changed_codes"],
               "near_boundary_codes": region_diagnostic["n_scores_within_0.01_of_boundary"]}
        history.append(row)
        print_epoch(epoch, row, args.K)

        checkpoint_data = {
            "model": model.state_dict(), "codebook": codebook.cpu(),
            "init_info": init_info, "config": checkpoint_config,
            "regions": {
                "benign_codes": benign_codes.detach().cpu().tolist() if benign_codes is not None else [],
                "harmful_codes": harmful_codes.detach().cpu().tolist() if harmful_codes is not None else [],
                "source": region_source,
                "assignment_space": region_assignment_space,
                "score_method": "response_presence",
                "prior_strength": args.code_score_prior_strength,
                "signed_harmfulness": np.asarray(
                    region_statistics["signed_harmfulness"], dtype=np.float32
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
        if joint_loss < best_joint - 1e-5:
            best_joint, best_joint_epoch = joint_loss, epoch
            torch.save(checkpoint_data, run_dir / "model_joint.pt")
        if args.patience and epochs_no_improve >= args.patience:
            print(f"  early stop @ epoch {epoch}", flush=True)
            break

    evaluation_sequences = test if test else validation
    metric_partition = "test" if test else "validation"

    def evaluate_checkpoint(checkpoint_name):
        checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_benign = checkpoint["regions"]["benign_codes"]
        checkpoint_harmful = checkpoint["regions"]["harmful_codes"]
        checkpoint_benign = torch.tensor(checkpoint_benign, device=device) if checkpoint_benign else None
        checkpoint_harmful = torch.tensor(checkpoint_harmful, device=device) if checkpoint_harmful else None
        return evaluate_vq_model(
            model, evaluation_sequences, args.batch_size, device,
            checkpoint_benign, checkpoint_harmful, args.task_tau,
        )

    recon_metrics = evaluate_checkpoint("model.pt")
    task_metrics = evaluate_checkpoint("model_task.pt")
    joint_metrics = evaluate_checkpoint("model_joint.pt")
    total_minutes = (time.time() - started_at) / 60
    gpu_peak_gb = torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else 0.0
    cpu_peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    checkpoint_metrics = {
        "model.pt": recon_metrics,
        "model_task.pt": task_metrics,
        "model_joint.pt": joint_metrics,
    }
    summary = {
        "best_combined": best_reconstruction, "best_epoch": best_reconstruction_epoch,
        "best_val_recon": best_reconstruction,
        "best_val_recon_epoch": best_reconstruction_epoch,
        "best_val_task": best_vtask, "best_val_task_epoch": best_vtask_epoch,
        "best_joint": best_joint, "best_joint_epoch": best_joint_epoch,
        "joint_selection": {
            "criterion": "mean(val_recon/reference_recon, val_task/reference_task)",
            "reference_source": "initialized model before training",
            "reference_metrics": joint_reference_metrics,
            "task_included": task_enabled,
        },
        "test_recon": recon_metrics["recon"] if test else None,
        "test_ev": recon_metrics["ev"] if test else None,
        "test_l2_ratio": recon_metrics["l2_ratio"] if test else None,
        "test_metrics": checkpoint_metrics if test else {name: None for name in checkpoint_metrics},
        "checkpoint_validation_metrics": checkpoint_metrics if not test else None,
        "checkpoint_metric_partition": metric_partition,
        "init_time_s": round(init_time, 1),
        "total_time_min": round(total_minutes, 1), "gpu_peak_gb": round(gpu_peak_gb, 1),
        "cpu_peak_gb": round(cpu_peak_gb, 1), "init_info": init_info,
        "config": checkpoint_config, "history": history,
    }
    json.dump(summary, open(run_dir / "train_summary.json", "w"), indent=2, default=float)
    print("\nDone", flush=True)
    print(
        f"model.pt: recon {best_reconstruction:.4f} at epoch {best_reconstruction_epoch} | "
        f"model_task.pt: task {best_vtask:.4f} at epoch {best_vtask_epoch} | "
        f"model_joint.pt: joint {best_joint:.4f} at epoch {best_joint_epoch}", flush=True,
    )
    for checkpoint_name, metrics in checkpoint_metrics.items():
        print(
            f"{checkpoint_name}: {metric_partition} recon {metrics['recon']:.4f}, "
            f"EV {metrics['ev']:.3f}, L2 {metrics['l2_ratio']:.3f}, task {metrics['task']:.3f}",
            flush=True,
        )
    print(
        f"peak memory: GPU {gpu_peak_gb:.1f} GB, CPU {cpu_peak_gb:.1f} GB | "
        f"{total_minutes:.1f} min | {run_dir}", flush=True,
    )


if __name__ == "__main__":
    main()
