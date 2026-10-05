#!/usr/bin/env python
"""Generate unsteered and TRACE-steered responses."""

import argparse
from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.concept_audit import (
    append_steering_cache,
    initialize_steering_cache,
)
from activations.cache import load_activation_sequences
from data.dataset_splits import DEFAULT_DATASET, evaluation_split, train_validation_split, training_split
from model_inputs import (
    restore_transformers_loss_kwargs, format_prompt_token_ids, repair_internlm3_rotary_embeddings,
)
from project_config import DEFAULT_BASE_MODEL, MODEL_DIR, READ_LAYER
from vq.codebook import (
    assign_codes_for_sequences,
    encode_and_assign_activations,
    nearest_code_indices,
    smoothed_response_code_statistics,
    split_assignments_by_response,
)
from vq.model import load_steering_vq_checkpoint


BASE_METHODS = ("additive", "ablation", "clamp")
METHOD_MODIFIERS = ("", "_renorm", "_gate", "_renorm_gate")
ALL_METHODS = tuple(method + modifier for method in BASE_METHODS for modifier in METHOD_MODIFIERS)
COUNTERPART_SELECTIONS = {
    "fixed": "euclid",
    "activation_nearest": "activation_nearest",
}


@dataclass(frozen=True)
class SteeringTargets:
    """Code regions, edit vectors, scores, and benign candidates used for steering."""

    harmful_codes: list[int]
    concept_vectors: torch.Tensor
    fixed_counterparts: torch.Tensor
    harmful_scores: torch.Tensor
    reliable_benign_codes: torch.Tensor


def nearest_benign_code_pairs(codebook, harmful_codes, device, benign_codes=None):
    """Return one nearest eligible benign code for every harmful code."""
    harmful_set = set(harmful_codes)
    if benign_codes is None:
        benign_codes = [code for code in range(codebook.shape[0]) if code not in harmful_set]
    benign_codes = torch.as_tensor(benign_codes, dtype=torch.long, device=device)
    benign_vectors = codebook[benign_codes]
    counterparts = torch.arange(codebook.shape[0], dtype=torch.long, device=device)
    for harmful_code in harmful_codes:
        nearest_benign = ((codebook[harmful_code] - benign_vectors) ** 2).sum(1).argmin()
        counterparts[harmful_code] = benign_codes[nearest_benign]
    return counterparts


def build_steering_targets(transcoder, num_codes, device, concept="code_mean", training_cache=None,
                           read_layer=READ_LAYER, prior_strength=None, min_benign_response_support=10):
    """Build harmful codes, edit vectors, fixed pairs, scores, and reliable benign candidates.

    ``codebook`` uses the trained code vectors. ``code_mean`` uses each code's mean training activation in
    the activation space where the edit acts. Code assignment always stays on the trained codebook.
    """
    # Match the nested VQ split when estimating concept means and response support.
    split_seed = int(transcoder.checkpoint_config.get("seed", 42))
    all_training_sequences = load_activation_sequences(training_cache, read_layer)
    detector_training, _ = train_validation_split(all_training_sequences, split_seed)
    training_sequences, _ = train_validation_split(detector_training, split_seed)

    assignments = assign_codes_for_sequences(
        transcoder, training_sequences, device, assignment_space="encoded_activation"
    )
    response_assignments = split_assignments_by_response(training_sequences, assignments)
    labels = [sequence["label"] for sequence in training_sequences]
    if prior_strength is None:
        prior_strength = transcoder.checkpoint_config.get("code_score_prior_strength", 10.0)

    harmful_codes = transcoder.checkpoint_regions["harmful_codes"]
    harmful_scores = np.clip(
        np.asarray(transcoder.checkpoint_regions["signed_harmfulness"]), 0.0, None
    )

    # A benign target must occur in enough training responses to have a stable mean.
    response_statistics = smoothed_response_code_statistics(
        response_assignments, labels, num_codes, prior_strength
    )
    harmful_code_set = set(harmful_codes)
    reliable_benign_codes = []
    for code, support in enumerate(response_statistics["response_counts"]):
        if code in harmful_code_set:
            continue
        if support >= min_benign_response_support:
            reliable_benign_codes.append(code)

    # Steering acts in residual-stream space, so use each concept's mean activation.
    concept_vectors = transcoder.quantizer.codebook.detach().to(device)
    if concept == "code_mean":
        activation_dimension = training_sequences[0]["x"].shape[1]
        code_means = torch.zeros(num_codes, activation_dimension, dtype=torch.float32)
        code_counts = torch.zeros(num_codes, dtype=torch.float32)
        assignment_start = 0

        for sequence in training_sequences:
            assignment_end = assignment_start + len(sequence["x"])
            sequence_assignments = torch.from_numpy(assignments[assignment_start:assignment_end])
            sequence_activations = sequence["x"].float()
            code_means.index_add_(0, sequence_assignments, sequence_activations)
            code_counts.index_add_(0, sequence_assignments, torch.ones(len(sequence_assignments)))
            assignment_start = assignment_end

        code_means = (code_means / code_counts.clamp(min=1).unsqueeze(1)).to(device)
        unused_codes = code_counts.to(device) == 0
        code_means[unused_codes] = concept_vectors[unused_codes]
        concept_vectors = code_means

    harmful_scores = torch.tensor(harmful_scores, dtype=torch.float32, device=device)
    reliable_benign_codes = torch.tensor(reliable_benign_codes, dtype=torch.long, device=device)
    fixed_counterparts = nearest_benign_code_pairs(
        concept_vectors, harmful_codes, device, benign_codes=reliable_benign_codes
    )
    return SteeringTargets(
        harmful_codes=harmful_codes,
        concept_vectors=concept_vectors,
        fixed_counterparts=fixed_counterparts,
        harmful_scores=harmful_scores,
        reliable_benign_codes=reliable_benign_codes,
    )


def normalized_gate_weights(harmful_scores, harmful_codes):
    """Map the stored signed scores to steering weights without changing code order."""
    gate_weights = torch.zeros_like(harmful_scores)
    harmful_indices = torch.as_tensor(harmful_codes, dtype=torch.long, device=harmful_scores.device)
    positive_scores = harmful_scores[harmful_indices].clamp(min=0)
    maximum_score = positive_scores.max()
    gate_weights[harmful_indices] = positive_scores / maximum_score
    return gate_weights


def spherical_interpolate_direction(activations, direction, fraction):
    """Rotate each activation toward an edit direction while preserving its original norm."""
    original_norm = activations.norm(dim=-1, keepdim=True)
    start = activations / (original_norm + 1e-6)
    target = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)
    angle = torch.acos((start * target).sum(-1, keepdim=True).clamp(-0.9999, 0.9999))
    sine = torch.sin(angle)
    rotated = (
        torch.sin((1 - fraction) * angle) * start + torch.sin(fraction * angle) * target
    ) / sine
    return original_norm * torch.where(sine.abs() < 1e-4, start, rotated)


class ActivationSteerer:
    """Forward hook that edits harmful-coded token activations toward benign targets."""

    def __init__(self, transcoder, harmful_codes, concept_vectors, counterparts, harmful_weights,
                 reliable_benign_codes, counterpart_selection, device, capture_trace=False):
        self.transcoder = transcoder
        self.concept_vectors = concept_vectors
        self.counterparts = counterparts
        self.harmful_weights = harmful_weights
        self.reliable_benign_codes = reliable_benign_codes
        self.counterpart_selection = counterpart_selection
        self.capture_trace = bool(capture_trace)
        self.harmful_code_tensor = torch.tensor(harmful_codes, device=device)
        self.strength = 0.0
        self.method = None
        self.active = True
        self.base_method = None
        self.preserve_norm = False
        self.weight_by_harmfulness = False
        self.reset()

    def reset(self):
        self.calls = 0
        self.fire_steps = []
        self.assigned_code_steps = []
        self.target_code_steps = []

    def configure(self, strength, method):
        self.strength = strength
        self.method = method
        if method is not None:
            self.base_method = method.split("_", 1)[0]
            self.preserve_norm = "renorm" in method
            self.weight_by_harmfulness = "gate" in method

    def __call__(self, module, inputs, output):
        if not self.active:
            return output
        self.calls += 1
        if self.method is None or self.calls == 1:
            return output

        hidden_states = output[0] if isinstance(output, tuple) else output
        current_activations = hidden_states[:, -1, :].float()

        # Assign the newest token and edit only harmful concepts.
        encoded_activations, codes = encode_and_assign_activations(self.transcoder, current_activations)
        edit_mask = torch.isin(codes, self.harmful_code_tensor)
        self.fire_steps.append(edit_mask.detach().cpu())
        target_codes = torch.full_like(codes, -1)

        if edit_mask.any():
            harmful_codes = codes[edit_mask]
            edited_activations = current_activations[edit_mask]
            harmful_vectors = self.concept_vectors[harmful_codes]

            # Choose the fixed benign counterpart or the nearest target for this activation.
            if self.counterpart_selection == "activation_nearest":
                edited_encoded = encoded_activations[edit_mask]
                candidate_vectors = self.transcoder.quantizer.codebook[self.reliable_benign_codes]
                candidate_distances = (
                    edited_encoded.pow(2).sum(1, keepdim=True)
                    - 2 * edited_encoded @ candidate_vectors.t()
                    + candidate_vectors.pow(2).sum(1)
                )
                benign_codes = self.reliable_benign_codes[candidate_distances.argmin(1)]
            else:
                benign_codes = self.counterparts[harmful_codes]

            target_codes[edit_mask] = benign_codes
            benign_vectors = self.concept_vectors[benign_codes]
            direction = benign_vectors - harmful_vectors

            edit_weight = 1.0
            if self.weight_by_harmfulness:
                edit_weight = self.harmful_weights[harmful_codes].unsqueeze(-1)

            # Norm-preserving addition rotates the activation toward the edit direction.
            if self.base_method == "additive" and self.preserve_norm:
                steered = spherical_interpolate_direction(
                    edited_activations, direction, self.strength * edit_weight
                )
            else:
                if self.base_method == "additive":
                    edit = self.strength * direction
                else:
                    harmful_axis = -direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)
                    projected = edited_activations
                    if self.base_method == "clamp":
                        projected = edited_activations - benign_vectors
                    edit = (
                        -self.strength
                        * (projected * harmful_axis).sum(-1, keepdim=True)
                        * harmful_axis
                    )
                steered = edited_activations + edit * edit_weight

                if self.preserve_norm:
                    original_norm = edited_activations.norm(dim=-1, keepdim=True)
                    steered = steered * (
                        original_norm / (steered.norm(dim=-1, keepdim=True) + 1e-6)
                    )

            edited_rows = edit_mask.nonzero(as_tuple=True)[0]
            hidden_states[edited_rows, -1, :] = steered.to(hidden_states.dtype)

        if self.capture_trace:
            self.assigned_code_steps.append(codes.detach().cpu())
            self.target_code_steps.append(target_codes.detach().cpu())
        return output


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, steerer, max_new_tokens):
    """Greedily generate a batch with the steering hook active and return edit metadata."""
    steerer.reset()
    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = tokenizer.eos_token_id

    encoded_prompts = [format_prompt_token_ids(tokenizer, prompt) for prompt in prompts]
    max_prompt_length = max(len(tokens) for tokens in encoded_prompts)

    padded_prompts = []
    prompt_masks = []
    for tokens in encoded_prompts:
        padding_length = max_prompt_length - len(tokens)
        padded_prompts.append([padding_token_id] * padding_length + tokens)
        prompt_masks.append([0] * padding_length + [1] * len(tokens))

    input_ids = torch.tensor(padded_prompts, device=model.device)
    attention_mask = torch.tensor(prompt_masks, device=model.device)
    generated = model.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=padding_token_id,
    )

    response_tokens = generated[:, max_prompt_length:]
    texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in response_tokens]
    fired_counts, generated_lengths = edits_per_prompt(steerer, texts, tokenizer)
    traces = steering_traces_per_prompt(steerer, generated_lengths)
    return texts, fired_counts, generated_lengths, traces


def edits_per_prompt(steerer, texts, tokenizer):
    """Return generated-token lengths and the number of tokens edited in each response."""
    generated_lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    if not steerer.fire_steps:
        return [0] * len(texts), generated_lengths
    fired_steps = torch.stack(steerer.fire_steps, 0)
    fired_counts = []
    for prompt_index, generated_length in enumerate(generated_lengths):
        fired_counts.append(int(fired_steps[:generated_length, prompt_index].sum()))
    return fired_counts, generated_lengths


def steering_traces_per_prompt(steerer, generated_lengths):
    """Return compact online code and source-to-target edit traces for each response."""
    if not steerer.capture_trace:
        return None
    if not steerer.assigned_code_steps:
        return [
            {"online_code_ids": [], "edit_positions": [], "source_code_ids": [], "target_code_ids": []}
            for _ in generated_lengths
        ]
    assigned_steps = torch.stack(steerer.assigned_code_steps, 0)
    target_steps = torch.stack(steerer.target_code_steps, 0)
    traces = []
    for prompt_index, generated_length in enumerate(generated_lengths):
        traced_length = min(generated_length, assigned_steps.shape[0])
        assigned = assigned_steps[:traced_length, prompt_index]
        targets = target_steps[:traced_length, prompt_index]
        edit_positions = torch.nonzero(targets >= 0, as_tuple=True)[0]
        traces.append({
            "online_code_ids": assigned.tolist(),
            "edit_positions": edit_positions.tolist(),
            "source_code_ids": assigned[edit_positions].tolist(),
            "target_code_ids": targets[edit_positions].tolist(),
        })
    return traces


def response_nll(model, final_hidden_states, input_ids, response_start, chunk_size=64):
    """Mean base-model NLL over response tokens, using small LM-head chunks to limit peak memory."""
    prediction_states = final_hidden_states[0, response_start - 1:-1]
    response_tokens = input_ids[0, response_start:]
    total_nll = 0.0
    for start in range(0, len(response_tokens), chunk_size):
        end = start + chunk_size
        logits = model.lm_head(prediction_states[start:end]).float()
        total_nll += F.cross_entropy(logits, response_tokens[start:end], reduction="sum").item()
    return total_nll / len(response_tokens)


@torch.no_grad()
def score_response(model, tokenizer, transcoder, harmful_codes, prompt, generated_text, read_layer):
    """Score harmful-code rate and base-model NLL with the steering hook disabled."""
    response_token_ids = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
    if not response_token_ids:
        return None, None, [], []
    prompt_token_ids = format_prompt_token_ids(tokenizer, prompt)
    input_ids = torch.tensor([prompt_token_ids + response_token_ids], device=model.device)
    model_output = model.model(input_ids, output_hidden_states=True, use_cache=False)
    read_layer_activations = model_output.hidden_states[read_layer]
    codes = nearest_code_indices(transcoder, read_layer_activations[0, len(prompt_token_ids):, :].float())
    harmful_code_rate = float(torch.isin(codes, harmful_codes).float().mean())
    mean_nll = response_nll(model, model_output.last_hidden_state, input_ids, len(prompt_token_ids))
    return harmful_code_rate, mean_nll, response_token_ids, codes.cpu().tolist()


def non_repetition_score(text):
    """Distinct-trigram ratio: 1.0 means no exact repetition; approximately 0 means a loop."""
    words = text.split()
    trigrams = [tuple(words[index:index + 3]) for index in range(len(words) - 2)]
    return round(len(set(trigrams)) / len(trigrams), 3) if trigrams else 1.0


def process_batch(model, tokenizer, transcoder, steerer, batch, lams, max_new_tokens, methods,
                  recipe_prefix, read_layer, prompt_id_key):
    """Generate and score the baseline plus every method/lambda pair for one prompt batch."""
    prompts = [row["prompt"] for row in batch]
    records = []
    for row in batch:
        records.append({
            "idx": row[prompt_id_key],
            "label": row["label"],
            "prompt": row["prompt"],
            "baseline": None,
            "steered": {},
        })

    audit_rows = []
    configurations = [(None, 0.0)] + [(method, lam) for method in methods for lam in lams]

    for method, lam in configurations:
        # Generate one matched output for each prompt under this steering setting.
        steerer.configure(lam, method)
        texts, fired_counts, generated_lengths, traces = generate_batch(
            model, tokenizer, prompts, steerer, max_new_tokens
        )

        # Score the generated text with no intervention active.
        steerer.active = False
        scores = []
        for prompt, text in zip(prompts, texts):
            scores.append(score_response(
                model, tokenizer, transcoder, steerer.harmful_code_tensor,
                prompt, text, read_layer,
            ))
        steerer.active = True

        # Store output quality, concept rate, and optional per-token audit records.
        for index, text in enumerate(texts):
            harmful_code_rate, mean_nll, token_ids, output_code_ids = scores[index]
            non_repetition = non_repetition_score(text)
            cell = {
                "text": text,
                "rate": harmful_code_rate,
                "non_repetition": non_repetition,
                "base_model_nll": round(mean_nll, 4) if mean_nll is not None else None,
            }
            if method is None:
                cell["base_model_nll_increase"] = 0.0 if mean_nll is not None else None
                records[index]["baseline"] = cell
                result_key = "baseline"
            else:
                baseline_nll = records[index]["baseline"]["base_model_nll"]
                nll_increase = None
                if mean_nll is not None and baseline_nll is not None:
                    nll_increase = round(mean_nll - baseline_nll, 4)

                cell["base_model_nll_increase"] = nll_increase
                cell["steered"] = f"{fired_counts[index]}/{generated_lengths[index]}"
                result_key = f"{recipe_prefix}_{method}_lam{lam:g}"
                records[index]["steered"][result_key] = cell

            if steerer.capture_trace:
                if method is None:
                    trace = {
                        "online_code_ids": [],
                        "edit_positions": [],
                        "source_code_ids": [],
                        "target_code_ids": [],
                    }
                else:
                    trace = traces[index]

                audit_rows.append({
                    "response_id": int(batch[index][prompt_id_key]),
                    "label": int(batch[index]["label"]),
                    "result_key": result_key,
                    "output_token_ids": token_ids,
                    "output_code_ids": output_code_ids,
                    **trace,
                })
    return records, audit_rows


def sample_labeled_rows(rows, n_harmful, n_safe, seed):
    """Take a reproducible random sample from each response-label group."""
    random_state = np.random.RandomState(seed)
    sampled = []
    for label, requested in ((1, n_harmful), (0, n_safe)):
        group = [row for row in rows if row["label"] == label]
        if requested > len(group):
            raise ValueError(f"requested {requested} label-{label} responses, but only {len(group)} exist")
        indices = random_state.permutation(len(group))[:requested]
        sampled.extend(group[index] for index in indices)
    return sampled


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-set", default="train")
    parser.add_argument("--activation-cache",
                        help="training activation cache used for code means (default: selected train set)")
    parser.add_argument("--test-set", default="test",
                        help="official test set or outer_validation")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="seed for the outer detector split and its class-balanced random sample")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-unsafe", type=int, default=None)
    parser.add_argument("--n-safe", type=int, default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="prompts generated together per call (1 = exact per-prompt reference)")
    parser.add_argument(
        "--methods", nargs="+", choices=ALL_METHODS,
        default=["additive", "additive_gate"], metavar="METHOD",
        help="base methods plus optional _renorm and _gate modifiers",
    )
    parser.add_argument("--concept", choices=["codebook", "code_mean"], default="code_mean",
                        help="trained code vectors or each code's mean training activation")
    parser.add_argument(
        "--counterpart-selection", choices=COUNTERPART_SELECTIONS, default="fixed",
        help="fixed code-level pair or nearest reliable benign code to the current activation",
    )
    parser.add_argument(
        "--min-benign-response-support", type=int, default=10,
        help="minimum training responses containing a code before it can be a dynamic benign target",
    )
    parser.add_argument("--code-score-prior-strength", type=float, default=None,
                        help="response-equivalent harmfulness prior (default: checkpoint value or 10)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an incomplete run after verifying its saved configuration")
    parser.add_argument(
        "--audit-cache-out",
        help="optionally save online code assignments and source-to-target edits outside intervene.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve the training and evaluation partitions used by this run.
    training_data = training_split(args.train_set or "train", args.dataset)
    outer_validation = args.test_set == "outer_validation"
    if outer_validation:
        evaluation_data = training_data
    else:
        evaluation_data = evaluation_split(args.test_set, args.dataset)
    args.train_set = training_data.name
    args.test_set = "outer_validation" if outer_validation else evaluation_data.name
    if outer_validation:
        args.n_unsafe = args.n_unsafe if args.n_unsafe is not None else 200
        args.n_safe = args.n_safe if args.n_safe is not None else 200
    checkpoint_path = Path(args.checkpoint)

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "intervene.json"
    audit_cache_path = Path(args.audit_cache_out) if args.audit_cache_out else None
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"{output_path} already exists; pass --resume to continue it safely")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = time.time()

    # Load the frozen generator and the VQ checkpoint that defines the concepts.
    model_path = MODEL_DIR / args.base_model
    restore_transformers_loss_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    repair_internlm3_rotary_embeddings(model)
    transcoder, num_codes = load_steering_vq_checkpoint(checkpoint_path, device)
    read_layer = int(transcoder.checkpoint_config.get("read_layer", READ_LAYER))
    hook_layer_index = read_layer - 1

    checkpoint_config = transcoder.checkpoint_config
    cache_override = args.activation_cache or checkpoint_config.get("training_cache")
    if cache_override is None:
        raise ValueError("--activation-cache is required when the VQ checkpoint does not record one")
    training_cache = Path(cache_override)
    if args.code_score_prior_strength is None:
        args.code_score_prior_strength = transcoder.checkpoint_config.get("code_score_prior_strength", 10.0)

    # Build harmful-to-benign directions and attach the activation hook.
    targets = build_steering_targets(
        transcoder, num_codes, device, args.concept, training_cache, read_layer,
        args.code_score_prior_strength, args.min_benign_response_support,
    )
    harmful_weights = normalized_gate_weights(targets.harmful_scores, targets.harmful_codes)
    elapsed = time.time() - started_at
    print(f"loaded: {len(targets.harmful_codes)}/{num_codes} harmful codes | "
          f"{len(targets.reliable_benign_codes)} reliable benign targets | concept={args.concept} | "
          f"counterpart={args.counterpart_selection} | {elapsed:.0f}s", flush=True)

    steerer = ActivationSteerer(
        transcoder, targets.harmful_codes, targets.concept_vectors, targets.fixed_counterparts,
        harmful_weights, targets.reliable_benign_codes, args.counterpart_selection, device,
        capture_trace=audit_cache_path is not None,
    )
    handle = model.model.layers[hook_layer_index].register_forward_hook(steerer)

    # Select the harmful and safe prompts used for steering evaluation.
    with evaluation_data.source.open() as input_file:
        rows = [json.loads(line) for line in input_file]
    if outer_validation:
        _, rows = train_validation_split(rows, args.selection_seed)
        requested_prompts = sample_labeled_rows(rows, args.n_unsafe, args.n_safe, args.selection_seed)
    else:
        harmful_rows = [row for row in rows if row["label"] == 1]
        safe_rows = [row for row in rows if row["label"] == 0]
        args.n_unsafe = len(harmful_rows) if args.n_unsafe is None else args.n_unsafe
        args.n_safe = len(safe_rows) if args.n_safe is None else args.n_safe
        harmful_prompts = harmful_rows[:args.n_unsafe]
        safe_prompts = safe_rows[:args.n_safe]
        requested_prompts = harmful_prompts + safe_prompts

        if len(requested_prompts) != args.n_unsafe + args.n_safe:
            raise ValueError(
                f"requested {args.n_unsafe} harmful and {args.n_safe} safe responses, "
                f"but {len(requested_prompts)} total were available"
            )
    prompt_id_key = evaluation_data.id_key
    prompts = requested_prompts
    results = []
    completed_indices = set()
    recipe_prefix = COUNTERPART_SELECTIONS[args.counterpart_selection]

    if args.resume:
        if not output_path.exists():
            raise FileNotFoundError(f"cannot resume because {output_path} does not exist")

        previous_run = json.loads(output_path.read_text())
        results = previous_run.get("results", [])
        completed_indices = {record["idx"] for record in results}
        prompts = [row for row in requested_prompts if row[prompt_id_key] not in completed_indices]
        print(f"resuming: kept {len(results)} completed prompts; {len(prompts)} remain", flush=True)

    # Initialize the optional token-level audit record.
    audit_events_path = None
    audit_keys = set()
    if audit_cache_path is not None:
        result_keys = ["baseline"] + [
            f"{recipe_prefix}_{method}_lam{lam:g}" for method in args.methods for lam in args.lambdas
        ]
        audit_events_path, audit_keys = initialize_steering_cache(
            audit_cache_path,
            {
                "intervene_file": str(output_path),
                "checkpoint": str(checkpoint_path),
                "dataset": args.dataset,
                "test_set": args.test_set,
                "base_model": args.base_model,
                "concept_vector_source": args.concept,
                "counterpart_selection": args.counterpart_selection,
                "result_keys": result_keys,
                "max_new_tokens": args.max_new_tokens,
            },
            resume=args.resume,
        )

    def save():
        data = {
            "dataset": args.dataset,
            "train_set": args.train_set,
            "test_set": args.test_set,
            "evaluation_source": str(evaluation_data.source),
            "evaluation_partition": "outer_validation" if outer_validation else "test",
            "base_model": args.base_model,
            "checkpoint": str(checkpoint_path),
            "concept": args.concept,
            "read_layer": read_layer,
            "hook_layer_index": hook_layer_index,
            "activation_cache": str(training_cache),
            "counterpart_selection": args.counterpart_selection,
            "min_benign_response_support": args.min_benign_response_support,
            "n_reliable_benign_codes": len(targets.reliable_benign_codes),
            "code_score_prior_strength": args.code_score_prior_strength,
            "code_score_method": transcoder.checkpoint_regions.get("score_method"),
            "code_score_source": "VQ checkpoint",
            "methods": args.methods,
            "lambdas": args.lambdas,
            "recipes": [f"{recipe_prefix}_{method}" for method in args.methods],
            "max_new_tokens": args.max_new_tokens,
            "n_unsafe": args.n_unsafe,
            "n_safe": args.n_safe,
            "batch_size": args.batch_size,
            "n_harmful_codes": len(targets.harmful_codes),
            "harmful_codes": targets.harmful_codes,
            "harmful_code_scores": [round(float(score), 8) for score in targets.harmful_scores.cpu()],
            "harmful_code_weights": [round(float(weight), 8) for weight in harmful_weights.cpu()],
            "results": results,
        }
        if outer_validation:
            data["selection_seed"] = args.selection_seed
        if audit_cache_path is not None:
            data["audit_cache"] = str(audit_cache_path)
        (run_dir / "intervene.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        )

    # Generate, score, and save each prompt batch.
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        records, audit_rows = process_batch(
            model, tokenizer, transcoder, steerer, batch, args.lambdas, args.max_new_tokens,
            args.methods, recipe_prefix, read_layer, prompt_id_key,
        )
        if audit_events_path is not None:
            append_steering_cache(audit_events_path, audit_rows, audit_keys)
        results.extend(records)
        prompt_order = {row[prompt_id_key]: index for index, row in enumerate(requested_prompts)}
        results.sort(key=lambda record: prompt_order[record["idx"]])
        save()
        for record in records:
            summary = {}
            for method in args.methods:
                rates = []
                for strength in args.lambdas:
                    key = f"{recipe_prefix}_{method}_lam{strength:g}"
                    rate = record["steered"][key]["rate"]
                    rates.append(None if rate is None else round(rate, 2))
                summary[method] = rates

            baseline_rate = record["baseline"]["rate"]
            if baseline_rate is not None:
                baseline_rate = round(baseline_rate, 2)
            print(
                f"[{record['idx']}] label={record['label']} "
                f"base={baseline_rate} {summary}",
                flush=True,
            )

    handle.remove()
    elapsed_minutes = (time.time() - started_at) / 60
    print(f"\n  {len(results)} prompts (batch {args.batch_size}) -> {run_dir}/intervene.json "
          f"({elapsed_minutes:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
