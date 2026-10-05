#!/usr/bin/env python
"""Generate matched unsteered and TRACE-steered responses."""

import argparse
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
from activations.activation_cache import load_activation_sequences, resolve_activation_cache
from data.dataset_splits import (
    DETECTOR_TRAIN_SET_BY_DATASET,
    evaluation_split,
    train_validation_split,
    training_split,
)
from model_inputs import (
    configure_transformers_compatibility, format_prompt_token_ids, initialize_internlm3_rotary_embeddings,
)
from project_config import DEFAULT_BASE_MODEL, DEFAULT_DATASET, READ_LAYER
from project_config import base_model_path, resolve_project_path
from vq.codebook import (
    assign_codes_for_sequences,
    encode_and_assign_activations,
    nearest_code_indices,
    smoothed_response_code_statistics,
    split_assignments_by_response,
)
from vq.model import load_steering_vq_checkpoint


METHODS = ("additive_renorm_gate", "additive_gate", "additive_renorm")
RECIPE_PREFIX = "euclid"


def nearest_benign_code_pairs(codebook, harmful_codes, benign_codes):
    """Return one nearest eligible benign code for every harmful code."""
    benign_codes = torch.as_tensor(benign_codes, dtype=torch.long, device=codebook.device)
    if not len(benign_codes):
        raise ValueError("at least one eligible benign code is required")
    benign_vectors = codebook[benign_codes]
    counterparts = torch.arange(codebook.shape[0], dtype=torch.long, device=codebook.device)
    for harmful_code in harmful_codes:
        nearest_benign = ((codebook[harmful_code] - benign_vectors) ** 2).sum(1).argmin()
        counterparts[harmful_code] = benign_codes[nearest_benign]
    return counterparts


def build_steering_targets(transcoder, num_codes, device, training_cache, read_layer,
                           prior_strength, min_benign_response_support=10):
    """Build harmful codes, activation-space edit vectors, and benign counterparts."""
    split_seed = int(transcoder.checkpoint_config.get("seed", 42))
    all_training_sequences = load_activation_sequences(training_cache, read_layer)
    if transcoder.checkpoint_config.get("split_scheme") != "nested_90_5_5":
        raise ValueError("TRACE steering requires a checkpoint trained with the nested split")
    detector_training, _ = train_validation_split(all_training_sequences, split_seed)
    training_sequences, _ = train_validation_split(detector_training, split_seed)
    assignments = assign_codes_for_sequences(
        transcoder, training_sequences, device, assignment_space="encoded_activation"
    )
    response_assignments = split_assignments_by_response(training_sequences, assignments)
    labels = [sequence["label"] for sequence in training_sequences]

    saved_regions = transcoder.checkpoint_regions or {}
    saved_harmfulness = saved_regions.get("signed_harmfulness")
    score_method = saved_regions.get("score_method")
    if score_method != "response_presence":
        raise ValueError("TRACE steering requires response-presence code scores in the VQ checkpoint")
    saved_prior = float(saved_regions.get("prior_strength", prior_strength))
    saved_region_matches = np.isclose(saved_prior, prior_strength)
    has_saved_scores = saved_regions.get("harmful_codes") and saved_harmfulness is not None
    if has_saved_scores and not saved_region_matches:
        raise ValueError(
            f"the checkpoint's stored code score uses prior strength {saved_prior:g}, but "
            f"{prior_strength:g} was requested"
        )
    if not has_saved_scores:
        raise ValueError("the VQ checkpoint does not contain harmfulness scores for steering")
    harmful_codes = saved_regions["harmful_codes"]
    harmful_scores = np.clip(np.asarray(saved_harmfulness), 0.0, None)

    response_statistics = smoothed_response_code_statistics(
        response_assignments, labels, num_codes, prior_strength
    )
    harmful_code_set = set(harmful_codes)
    reliable_benign_codes = [
        code for code, support in enumerate(response_statistics["response_counts"])
        if code not in harmful_code_set and support >= min_benign_response_support
    ]
    if not reliable_benign_codes:
        raise ValueError(
            "no benign code meets the minimum response-support requirement; "
            "lower --min-benign-response-support"
        )

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
    concept_vectors = (code_means / code_counts.clamp(min=1).unsqueeze(1)).to(device)
    unused_codes = code_counts.to(device) == 0
    concept_vectors[unused_codes] = transcoder.quantizer.codebook.detach().to(device)[unused_codes]

    harmful_scores = torch.tensor(harmful_scores, dtype=torch.float32, device=device)
    reliable_benign_codes = torch.tensor(reliable_benign_codes, dtype=torch.long, device=device)
    fixed_counterparts = nearest_benign_code_pairs(
        concept_vectors, harmful_codes, reliable_benign_codes
    )
    return harmful_codes, concept_vectors, fixed_counterparts, harmful_scores, reliable_benign_codes


def normalized_gate_weights(harmful_scores, harmful_codes):
    """Map one canonical signed score to steering weights without changing its code ordering."""
    gate_weights = torch.zeros_like(harmful_scores)
    harmful_indices = torch.as_tensor(harmful_codes, dtype=torch.long, device=harmful_scores.device)
    positive_scores = harmful_scores[harmful_indices].clamp(min=0)
    maximum_score = positive_scores.max() if len(positive_scores) else harmful_scores.new_tensor(0.0)
    if maximum_score <= 0:
        raise ValueError("harmful codes must include at least one positive harmfulness score")
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
                 device, capture_trace=False):
        self.transcoder = transcoder
        self.concept_vectors = concept_vectors
        self.counterparts = counterparts
        self.harmful_weights = harmful_weights
        self.capture_trace = capture_trace
        self.harmful_code_ids = torch.tensor(harmful_codes, device=device)
        self.lam, self.method, self.active = 0.0, None, True
        self.renorm, self.gate = False, False
        self.reset()

    def reset(self):
        self.calls = 0
        self.fire_steps = []
        self.assigned_code_steps = []
        self.target_code_steps = []

    def configure(self, lam, method):
        self.lam, self.method = lam, method
        if method is not None:
            self.renorm, self.gate = "renorm" in method, "gate" in method

    def __call__(self, _module, _inputs, output):
        if not self.active:
            return output
        self.calls += 1
        if self.method is None or self.calls == 1:
            return output

        hidden_states = output[0] if isinstance(output, tuple) else output
        current_activations = hidden_states[:, -1, :].float()
        _, codes = encode_and_assign_activations(self.transcoder, current_activations)
        fired = torch.isin(codes, self.harmful_code_ids)
        self.fire_steps.append(fired.detach().cpu())
        target_codes = torch.full_like(codes, -1)

        if fired.any():
            harmful_codes = codes[fired]
            fired_activations = current_activations[fired]
            harmful_vectors = self.concept_vectors[harmful_codes]
            benign_codes = self.counterparts[harmful_codes]
            target_codes[fired] = benign_codes
            benign_vectors = self.concept_vectors[benign_codes]
            direction = benign_vectors - harmful_vectors
            gate_weight = self.harmful_weights[harmful_codes].unsqueeze(-1) if self.gate else 1.0
            if self.renorm:
                steered = spherical_interpolate_direction(
                    fired_activations, direction, self.lam * gate_weight
                )
            else:
                steered = fired_activations + self.lam * direction * gate_weight
            fired_rows = fired.nonzero(as_tuple=True)[0]
            hidden_states[fired_rows, -1, :] = steered.to(hidden_states.dtype)
        if self.capture_trace:
            self.assigned_code_steps.append(codes.detach().cpu())
            self.target_code_steps.append(target_codes.detach().cpu())
        return output


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, steerer, max_new_tokens):
    """Greedily generate a batch with the steering hook active and return edit metadata."""
    steerer.reset()
    padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded_prompts = [format_prompt_token_ids(tokenizer, prompt) for prompt in prompts]
    max_prompt_length = max(len(tokens) for tokens in encoded_prompts)
    input_ids = torch.tensor([
        [padding_token_id] * (max_prompt_length - len(tokens)) + tokens for tokens in encoded_prompts
    ], device=model.device)
    attention_mask = torch.tensor([
        [0] * (max_prompt_length - len(tokens)) + [1] * len(tokens) for tokens in encoded_prompts
    ], device=model.device)
    generated = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=padding_token_id)
    texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated[:, max_prompt_length:]]
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
def score_response(model, tokenizer, transcoder, prompt, generated_text, read_layer):
    """Measure response NLL and assign output tokens to concepts."""
    response_token_ids = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
    if not response_token_ids:
        return None, [], []
    prompt_token_ids = format_prompt_token_ids(tokenizer, prompt)
    input_ids = torch.tensor([prompt_token_ids + response_token_ids], device=model.device)
    model_output = model.model(input_ids, output_hidden_states=True, use_cache=False)
    read_layer_activations = model_output.hidden_states[read_layer]
    codes = nearest_code_indices(transcoder, read_layer_activations[0, len(prompt_token_ids):, :].float())
    mean_nll = response_nll(model, model_output.last_hidden_state, input_ids, len(prompt_token_ids))
    return mean_nll, response_token_ids, codes.cpu().tolist()


def non_repetition_score(text):
    """Distinct-trigram ratio: 1.0 means no exact repetition; approximately 0 means a loop."""
    words = text.split()
    trigrams = [tuple(words[index:index + 3]) for index in range(len(words) - 2)]
    return round(len(set(trigrams)) / len(trigrams), 3) if trigrams else 1.0


def process_batch(model, tokenizer, transcoder, steerer, batch, lams, max_new_tokens, methods,
                  read_layer, prompt_id_key):
    """Generate and score the baseline plus every method/lambda pair for one prompt batch."""
    prompts = [row["prompt"] for row in batch]
    records = [{"idx": row[prompt_id_key], "label": row["label"], "prompt": row["prompt"],
                "baseline": None, "steered": {}} for row in batch]
    audit_rows = []
    configurations = [(None, 0.0)] + [(method, lam) for method in methods for lam in lams]
    for method, lam in configurations:
        steerer.configure(lam, method)
        texts, fired_counts, generated_lengths, traces = generate_batch(
            model, tokenizer, prompts, steerer, max_new_tokens
        )
        steerer.active = False
        scores = [score_response(model, tokenizer, transcoder, prompt, text, read_layer)
                  for prompt, text in zip(prompts, texts)]
        steerer.active = True
        for index, text in enumerate(texts):
            mean_nll, token_ids, output_code_ids = scores[index]
            non_repetition = non_repetition_score(text)
            cell = {
                "text": text,
                "non_repetition": non_repetition,
                "base_model_nll": round(mean_nll, 4) if mean_nll is not None else None,
            }
            if method is None:
                cell["base_model_nll_increase"] = 0.0 if mean_nll is not None else None
                records[index]["baseline"] = cell
                result_key = "baseline"
            else:
                baseline_nll = records[index]["baseline"]["base_model_nll"]
                cell["base_model_nll_increase"] = (
                    round(mean_nll - baseline_nll, 4)
                    if mean_nll is not None and baseline_nll is not None else None
                )
                cell["steered"] = f"{fired_counts[index]}/{generated_lengths[index]}"
                result_key = f"{RECIPE_PREFIX}_{method}_lam{lam:g}"
                records[index]["steered"][result_key] = cell
            if steerer.capture_trace:
                trace = traces[index] if method is not None else {
                    "online_code_ids": [], "edit_positions": [],
                    "source_code_ids": [], "target_code_ids": [],
                }
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
    parser.add_argument("--train-set", default=None,
                        help="training pool used by the VQ checkpoint")
    parser.add_argument("--activation-cache",
                        help="training activation cache used for code means (default: selected train set)")
    parser.add_argument("--test-set", default="test",
                        help="official test set or outer_validation")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="seed for the outer detector split and its class-balanced random sample")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--n-unsafe", "--n_unsafe", dest="n_unsafe", type=int, default=None)
    parser.add_argument("--n-safe", "--n_safe", dest="n_safe", type=int, default=None)
    parser.add_argument("--lambdas", "--lams", dest="lambdas", type=float, nargs="+", default=[0.5])
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=8,
                        help="prompts generated together per call (1 = exact per-prompt reference)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["additive_renorm_gate"],
                        metavar="METHOD")
    parser.add_argument("--min-benign-response-support", type=int, default=10,
                        help="minimum training responses containing a benign target concept")
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

    training_data = training_split(args.train_set or DETECTOR_TRAIN_SET_BY_DATASET[args.dataset], args.dataset)
    outer_validation = args.test_set == "outer_validation"
    if outer_validation:
        detector_train_set = DETECTOR_TRAIN_SET_BY_DATASET[args.dataset]
        evaluation_data = training_split(detector_train_set, args.dataset)
    else:
        evaluation_data = evaluation_split(args.test_set, args.dataset)
    args.train_set = training_data.name
    args.test_set = "outer_validation" if outer_validation else evaluation_data.name
    if outer_validation:
        args.n_unsafe = args.n_unsafe if args.n_unsafe is not None else 200
        args.n_safe = args.n_safe if args.n_safe is not None else 200
    checkpoint_path = resolve_project_path(args.checkpoint)

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "intervene.json"
    audit_cache_path = resolve_project_path(args.audit_cache_out) if args.audit_cache_out else None
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"{output_path} already exists; pass --resume to continue it")
    if not torch.cuda.is_available():
        raise RuntimeError("steering requires a GPU compute node")
    device = "cuda"
    started_at = time.time()

    model_path = base_model_path(args.base_model)
    configure_transformers_compatibility()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    initialize_internlm3_rotary_embeddings(model)
    transcoder, num_codes = load_steering_vq_checkpoint(checkpoint_path, device)
    read_layer = int(transcoder.checkpoint_config.get("read_layer", READ_LAYER))
    hook_layer_index = read_layer - 1

    checkpoint_config = transcoder.checkpoint_config
    expected_checkpoint_values = {
        "dataset": args.dataset,
        "train_set": args.train_set,
        "base_model": args.base_model,
    }
    for name, expected in expected_checkpoint_values.items():
        recorded = checkpoint_config.get(name)
        if recorded is not None and recorded != expected:
            raise ValueError(f"checkpoint {name} is {recorded!r}, but the run requested {expected!r}")

    if (checkpoint_config.get("dataset") == args.dataset
            and checkpoint_config.get("train_set") == args.train_set):
        recorded_cache = checkpoint_config.get("training_cache")
    else:
        recorded_cache = None
    cache_override = args.activation_cache or recorded_cache
    training_cache = resolve_activation_cache(
        args.dataset,
        training_data,
        explicit_path=cache_override,
    )
    if args.code_score_prior_strength is None:
        args.code_score_prior_strength = transcoder.checkpoint_config.get("code_score_prior_strength", 10.0)

    harmful_codes, concept_vectors, counterparts, harmful_scores, reliable_benign_codes = build_steering_targets(
        transcoder, num_codes, device, training_cache, read_layer,
        args.code_score_prior_strength, args.min_benign_response_support,
    )
    harmful_weights = normalized_gate_weights(harmful_scores, harmful_codes)
    elapsed = time.time() - started_at
    print(f"loaded: {len(harmful_codes)}/{num_codes} harmful codes | "
          f"{len(reliable_benign_codes)} reliable benign targets | {elapsed:.0f}s", flush=True)

    steerer = ActivationSteerer(
        transcoder, harmful_codes, concept_vectors, counterparts,
        harmful_weights, device, capture_trace=audit_cache_path is not None,
    )
    handle = model.model.layers[hook_layer_index].register_forward_hook(steerer)

    evaluation_rows = [json.loads(line) for line in evaluation_data.source.read_text().splitlines()]
    if outer_validation:
        _, evaluation_rows = train_validation_split(evaluation_rows, args.selection_seed)
        requested_prompts = sample_labeled_rows(
            evaluation_rows, args.n_unsafe, args.n_safe, args.selection_seed
        )
    else:
        unsafe_rows = [row for row in evaluation_rows if row["label"] == 1]
        safe_rows = [row for row in evaluation_rows if row["label"] == 0]
        args.n_unsafe = len(unsafe_rows) if args.n_unsafe is None else args.n_unsafe
        args.n_safe = len(safe_rows) if args.n_safe is None else args.n_safe
        requested_prompts = unsafe_rows[:args.n_unsafe] + safe_rows[:args.n_safe]
        if len(requested_prompts) != args.n_unsafe + args.n_safe:
            raise ValueError(
                f"requested {args.n_unsafe} harmful and {args.n_safe} safe responses, "
                f"but {len(requested_prompts)} total were available"
            )
    prompt_id_key = evaluation_data.id_key
    remaining_prompts = requested_prompts
    prompt_order = {row[prompt_id_key]: index for index, row in enumerate(requested_prompts)}
    results = []
    completed_ids = set()
    run_config = {
        "dataset": args.dataset,
        "train_set": args.train_set,
        "test_set": args.test_set,
        "base_model": args.base_model,
        "concept": "code_mean",
        "read_layer": read_layer,
        "counterpart_selection": "fixed",
        "min_benign_response_support": args.min_benign_response_support,
        "code_score_prior_strength": args.code_score_prior_strength,
        "methods": args.methods,
        "lambdas": args.lambdas,
        "max_new_tokens": args.max_new_tokens,
        "n_unsafe": args.n_unsafe,
        "n_safe": args.n_safe,
        "batch_size": args.batch_size,
    }
    if outer_validation:
        run_config["selection_seed"] = args.selection_seed

    if args.resume:
        if not output_path.exists():
            raise FileNotFoundError(f"cannot resume because {output_path} does not exist")

        previous_run = json.loads(output_path.read_text())
        expected_values = dict(run_config)
        if "selection_seed" in previous_run:
            expected_values["selection_seed"] = args.selection_seed
        mismatches = [
            f"{name}: saved={previous_run.get(name)!r}, requested={value!r}"
            for name, value in expected_values.items() if previous_run.get(name) != value
        ]
        saved_checkpoint = Path(previous_run.get("checkpoint", "")).resolve()
        if saved_checkpoint != checkpoint_path.resolve():
            mismatches.append(f"checkpoint: saved={saved_checkpoint}, requested={checkpoint_path.resolve()}")
        saved_cache = Path(previous_run.get("activation_cache", "")).resolve()
        if saved_cache != training_cache.resolve():
            mismatches.append(f"activation_cache: saved={saved_cache}, requested={training_cache.resolve()}")
        saved_audit_cache = previous_run.get("audit_cache")
        requested_audit_cache = str(audit_cache_path) if audit_cache_path is not None else None
        if saved_audit_cache != requested_audit_cache:
            mismatches.append(
                f"audit_cache: saved={saved_audit_cache!r}, requested={requested_audit_cache!r}"
            )
        if mismatches:
            raise ValueError("cannot resume with a different configuration:\n  " + "\n  ".join(mismatches))

        results = previous_run.get("results", [])
        completed_ids = {record["idx"] for record in results}
        remaining_prompts = [
            row for row in requested_prompts if row[prompt_id_key] not in completed_ids
        ]
        print(
            f"resuming: kept {len(results)} completed prompts; {len(remaining_prompts)} remain",
            flush=True,
        )

    audit_events_path = None
    audit_keys = set()
    if audit_cache_path is not None:
        result_keys = ["baseline"] + [
            f"{RECIPE_PREFIX}_{method}_lam{lam:g}"
            for method in args.methods for lam in args.lambdas
        ]
        audit_events_path, audit_keys = initialize_steering_cache(
            audit_cache_path,
            {
                "intervene_file": str(output_path.resolve()),
                "checkpoint": str(checkpoint_path),
                "dataset": args.dataset,
                "test_set": args.test_set,
                "base_model": args.base_model,
                "concept_vector_source": "code_mean",
                "counterpart_selection": "fixed",
                "result_keys": result_keys,
                "max_new_tokens": args.max_new_tokens,
            },
            resume=args.resume,
        )
        missing_audit_rows = {
            (int(response_id), result_key)
            for response_id in completed_ids for result_key in result_keys
        } - audit_keys
        if missing_audit_rows:
            raise ValueError(
                f"steering run has {len(missing_audit_rows)} completed results without audit traces"
            )

    run_data = {
        **run_config,
        "evaluation_source": str(evaluation_data.source),
        "evaluation_partition": "outer_validation" if outer_validation else "test",
        "checkpoint": str(checkpoint_path),
        "hook_layer_index": hook_layer_index,
        "activation_cache": str(training_cache),
        "n_reliable_benign_codes": len(reliable_benign_codes),
        "code_score_method": (transcoder.checkpoint_regions or {}).get("score_method"),
        "code_score_source": "VQ checkpoint",
        "recipes": [f"{RECIPE_PREFIX}_{method}" for method in args.methods],
        "lams": args.lambdas,
        "n_harmful_codes": len(harmful_codes),
        "harmful_codes": harmful_codes,
        "harmful_code_scores": [round(float(score), 8) for score in harmful_scores.cpu()],
        "harmful_code_weights": [round(float(weight), 8) for weight in harmful_weights.cpu()],
        "results": results,
    }
    if audit_cache_path is not None:
        run_data["audit_cache"] = str(audit_cache_path)

    def save():
        with output_path.open("w") as output_file:
            json.dump(run_data, output_file, indent=2, ensure_ascii=False)

    for start in range(0, len(remaining_prompts), args.batch_size):
        batch = remaining_prompts[start:start + args.batch_size]
        records, audit_rows = process_batch(
            model, tokenizer, transcoder, steerer, batch, args.lambdas, args.max_new_tokens,
            args.methods, read_layer, prompt_id_key,
        )
        if audit_events_path is not None:
            append_steering_cache(audit_events_path, audit_rows, audit_keys)
        results.extend(records)
        results.sort(key=lambda record: prompt_order[record["idx"]])
        save()
        for record in records:
            print(f"[{record['idx']}] label={record['label']} complete", flush=True)

    handle.remove()
    elapsed_minutes = (time.time() - started_at) / 60
    print(f"\n  {len(results)} prompts (batch {args.batch_size}) -> {run_dir}/intervene.json "
          f"({elapsed_minutes:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
