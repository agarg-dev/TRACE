"""Human-readable and numerical diagnostics for VQ training."""

import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch

from vq.codebook import (
    assign_codes_for_sequences,
    smoothed_response_code_statistics,
    smoothed_response_frequency_statistics,
    split_assignments_by_response,
    token_occurrence_code_statistics,
)


def _starts_word(tokenizer, token_id):
    decoded = tokenizer.decode([int(token_id)])
    return not decoded or decoded[0].isspace()


def _decode_tokens(tokenizer, token_ids):
    """Decode a possibly empty token slice without relying on tokenizer-specific behavior."""
    return "" if len(token_ids) == 0 else tokenizer.decode(token_ids)


def represent_firing(token_ids, position, tokenizer, repr_mode="phrase", context_len=4):
    """Render a firing as its token, complete word, or a short marked context."""
    if repr_mode == "token":
        return tokenizer.decode([token_ids[position]])
    if repr_mode == "word":
        start, end = position, position + 1
        while start > 0 and not _starts_word(tokenizer, token_ids[start]):
            start -= 1
        while end < len(token_ids) and not _starts_word(tokenizer, token_ids[end]):
            end += 1
        return tokenizer.decode(token_ids[start:end]).strip()
    start = max(0, position - context_len)
    end = min(len(token_ids), position + context_len + 1)
    prefix = _decode_tokens(tokenizer, token_ids[start:position])
    firing = tokenizer.decode([token_ids[position]])
    suffix = _decode_tokens(tokenizer, token_ids[position + 1:end])
    return prefix + "«" + firing + "»" + suffix


def summarize_initial_codebook(
    model, training_sequences, initialization_info, tokenizer, device, score_method, prior_strength,
    representation_mode="phrase", context_length=4, num_example_codes=6, examples_per_code=6,
):
    """Summarize initial code usage, label enrichment, regions, and representative firings."""
    token_labels = np.concatenate([
        np.full(len(sequence["x"]), sequence["label"], dtype=np.int8)
        for sequence in training_sequences
    ])
    sequence_by_token = np.concatenate([
        np.full(len(sequence["x"]), index, dtype=np.int32)
        for index, sequence in enumerate(training_sequences)
    ])
    position_by_token = np.concatenate([
        np.arange(len(sequence["x"]), dtype=np.int32) for sequence in training_sequences
    ])
    assignments = assign_codes_for_sequences(model, training_sequences, device)
    response_assignments = split_assignments_by_response(training_sequences, assignments)
    response_labels = (sequence["label"] for sequence in training_sequences)
    if score_method == "response_presence":
        region_statistics = smoothed_response_code_statistics(
            response_assignments, response_labels, model.quantizer.num_codes, prior_strength
        )
    elif score_method == "response_frequency":
        region_statistics = smoothed_response_frequency_statistics(
            response_assignments, response_labels, model.quantizer.num_codes, prior_strength
        )
    else:
        region_statistics = token_occurrence_code_statistics(
            response_assignments, response_labels, model.quantizer.num_codes
        )
    num_codes = model.quantizer.codebook.shape[0]
    cluster_sizes = np.bincount(assignments, minlength=num_codes)
    harmful_fraction = np.full(num_codes, np.nan)
    for code in range(num_codes):
        if cluster_sizes[code]:
            harmful_fraction[code] = token_labels[assignments == code].mean()

    def examples_of(code):
        occurrences = np.where(assignments == code)[0]
        if representation_mode == "phrase":
            if len(occurrences) > examples_per_code:
                occurrences = occurrences[
                    np.linspace(0, len(occurrences) - 1, examples_per_code).astype(int)
                ]
            examples = []
            for row in occurrences:
                sequence = training_sequences[sequence_by_token[row]]
                position = int(position_by_token[row])
                examples.append(
                    represent_firing(
                        sequence["token_ids"], position, tokenizer,
                        representation_mode, context_length,
                    )
                )
            return examples
        if len(occurrences) > 500:
            occurrences = occurrences[np.linspace(0, len(occurrences) - 1, 500).astype(int)]
        representations = []
        for row in occurrences:
            sequence = training_sequences[sequence_by_token[row]]
            position = int(position_by_token[row])
            representations.append(
                represent_firing(
                    sequence["token_ids"], position, tokenizer,
                    representation_mode, context_length,
                )
            )
        return [
            representation
            for representation, _ in Counter(representations).most_common(examples_per_code)
        ]

    if score_method in {"response_presence", "response_frequency"}:
        base_rate = region_statistics["base_harmful_response_rate"]
        code_harmfulness = region_statistics["smoothed_harmful_probability"]
        base_rate_unit = "responses"
        region_source = (
            "smoothed_response_enrichment" if score_method == "response_presence"
            else "smoothed_response_frequency_enrichment"
        )
    else:
        base_rate = region_statistics["base_harmful_token_rate"]
        code_harmfulness = region_statistics["harmful_probability"]
        base_rate_unit = "tokens"
        region_source = "token_occurrence_enrichment"
    harmful_codes = np.flatnonzero(region_statistics["signed_harmfulness"] > 0).tolist()
    harmful_code_set = set(harmful_codes)
    benign_codes = [code for code in range(num_codes) if code not in harmful_code_set]
    regions = {
        "benign_codes": benign_codes,
        "harmful_codes": harmful_codes,
        "base_rate": base_rate,
        "source": region_source,
        "prior_strength": prior_strength,
        "signed_harmfulness": region_statistics["signed_harmfulness"],
    }
    if score_method in {"response_presence", "response_frequency"}:
        regions["response_counts"] = region_statistics["response_counts"]
        if score_method == "response_frequency":
            regions["response_frequency_mass"] = region_statistics["response_frequency_mass"]
    else:
        regions["token_counts"] = region_statistics["token_counts"]

    ranked_codes = np.argsort(-code_harmfulness)
    sample_codes = [
        int(code)
        for code in list(ranked_codes[:num_example_codes // 2])
        + list(ranked_codes[-num_example_codes // 2:])
    ]
    examples = {code: examples_of(code) for code in sample_codes}
    example_codes = {}
    for code in sample_codes:
        example_codes[code] = {
            "region": "H" if code in harmful_code_set else "B",
            # This historical token-level diagnostic allows comparison between score definitions.
            "harm": round(float(harmful_fraction[code]), 2),
            "score_harm": round(float(code_harmfulness[code]), 2),
            "size": int(cluster_sizes[code]),
            "examples": examples[code],
        }

    statistics = {
        "K": num_codes,
        "n_train_tokens": int(len(assignments)),
        "init": initialization_info["init"],
        "code_repr": representation_mode,
        "base_rate": round(base_rate, 3),
        "base_rate_unit": base_rate_unit,
        "code_score_method": score_method,
        "code_score_prior_strength": (
            prior_strength if score_method in {"response_presence", "response_frequency"} else None
        ),
        "n_empty_codes": int((cluster_sizes == 0).sum()),
        "cluster_size_min_med_max": [
            int(cluster_sizes.min()), int(np.median(cluster_sizes)), int(cluster_sizes.max())
        ],
        "harm_enrichment_mean": round(float(np.nanmean(harmful_fraction)), 3),
        "enrichment_split": {"n_harmful": len(harmful_codes), "n_benign": len(benign_codes)},
        "example_codes": example_codes,
    }
    lines = [
        f"init clusters: {statistics['n_empty_codes']}/{num_codes} empty | sizes min/med/max "
        f"{statistics['cluster_size_min_med_max']} | mean harm-enrichment "
        f"{statistics['harm_enrichment_mean']}"
    ]
    for code in sample_codes:
        lines.append(
            f"  code {code:3d} [{statistics['example_codes'][code]['region']}] "
            f"score_harm={code_harmfulness[code]:.2f} "
            f"token_harm={harmful_fraction[code]:.2f} size={int(cluster_sizes[code]):4d}: "
            f"{examples[code][:2]}"
        )
    return statistics, "\n".join(lines), regions


def append_diagnostic_record(path, record):
    """Append one diagnostic record without retaining the full trace in memory."""
    with open(path, "a") as output_file:
        output_file.write(json.dumps(record, default=float) + "\n")


def summarize_region_changes(statistics, harmful_codes, previous_harmful_codes=None):
    """Summarize region movement and the response support behind the assignments."""
    harmful_codes = set(int(code) for code in harmful_codes)
    previous_harmful_codes = harmful_codes if previous_harmful_codes is None else previous_harmful_codes
    changed_codes = sorted(harmful_codes.symmetric_difference(previous_harmful_codes))
    signed_scores = np.asarray(statistics["signed_harmfulness"], dtype=np.float64)
    record = {
        "n_harmful_codes": len(harmful_codes),
        "n_benign_codes": len(signed_scores) - len(harmful_codes),
        "n_changed_codes": len(changed_codes),
        "changed_codes": changed_codes,
        "median_absolute_score": float(np.median(np.abs(signed_scores))),
        "n_scores_within_0.01_of_boundary": int((np.abs(signed_scores) <= 0.01).sum()),
    }
    if "response_counts" in statistics:
        support = np.asarray(statistics["response_counts"], dtype=np.int64)
        record.update({
            "response_support_min": int(support.min()),
            "response_support_median": float(np.median(support)),
            "response_support_max": int(support.max()),
            "n_codes_seen_in_fewer_than_10_responses": int((support < 10).sum()),
        })
    return record


def gradient_norms_by_component(model):
    """Compute pre-clipping gradient norms for the main trainable components."""
    squared_norms = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if "_ContinuousEmbedding" in name:
            group = "activation_encoder"
        elif "output_projection" in name:
            group = "output_projection"
        elif ".attn." in name:
            group = "attention"
        elif ".ffn." in name:
            group = "feed_forward"
        elif ".norm" in name:
            group = "decoder_norms"
        else:
            group = "other"
        norm = float(parameter.grad.detach().float().norm())
        squared_norms[group] += norm * norm
    return {group: math.sqrt(value) for group, value in squared_norms.items()}


@torch.no_grad()
def summarize_training_batch(
    batch, read_activations, target, labels, output, quantizer, codebook_before
):
    """Describe the forward pass that contributed to one accumulated optimizer step."""
    valid_tokens = labels >= 0

    def vector_norm_summary(tensor):
        values = tensor.detach().float().norm(dim=-1)[valid_tokens]
        return {"mean": float(values.mean()), "max": float(values.max())}

    reconstructed = output["reconstructed"].detach().float()
    target = target.detach().float()
    relative_error = (reconstructed - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-8)
    valid_relative_error = relative_error[valid_tokens]
    largest_flat_position = int(relative_error.masked_fill(~valid_tokens, -1).argmax())
    row, position = divmod(largest_flat_position, relative_error.shape[1])
    code = int(output["indices"][row, position])

    codebook_after = quantizer.codebook.detach().float()
    codebook_movement = (codebook_after - codebook_before.float()).norm(dim=1)
    most_moved_code = int(codebook_movement.argmax())
    used_codes = output["indices"][valid_tokens].long()
    decoder_assignment_distance = (
        output["z_e"].detach().float() - output["quantized"].detach().float()
    ).pow(2).sum(-1)[valid_tokens]

    return {
        "response_ids": [sequence["idx"] for sequence in batch],
        "labels": [int(sequence["label"]) for sequence in batch],
        "lengths": [int(len(sequence["x"])) for sequence in batch],
        "losses": {
            "relative_reconstruction_mean": float(valid_relative_error.mean()),
            "relative_reconstruction_max": float(valid_relative_error.max()),
            "commitment": float(output["commit_loss"]),
            "perplexity": float(output["perplexity"]),
        },
        "vector_norms": {
            "input": vector_norm_summary(read_activations),
            "encoded": vector_norm_summary(output["z_e"]),
            "quantized": vector_norm_summary(output["quantized"]),
            "target": vector_norm_summary(target),
            "reconstructed": vector_norm_summary(reconstructed),
        },
        "largest_error_token": {
            "response_id": batch[row]["idx"],
            "label": int(batch[row]["label"]),
            "position": int(position),
            "code": code,
            "relative_error": float(relative_error[row, position]),
            "encoded_norm": float(output["z_e"][row, position].float().norm()),
            "quantized_norm": float(output["quantized"][row, position].float().norm()),
            "target_norm": float(target[row, position].norm()),
            "reconstructed_norm": float(reconstructed[row, position].norm()),
        },
        "codebook_update": {
            "most_moved_code": most_moved_code,
            "largest_vector_movement": float(codebook_movement[most_moved_code]),
            "largest_used_vector_movement": float(codebook_movement[used_codes].max()),
            "max_code_norm": float(codebook_after.norm(dim=1).max()),
            "max_code_norm_id": int(codebook_after.norm(dim=1).argmax()),
            "assignment_distance_before_update_mean": float(output["min_distances"][valid_tokens].mean()),
            "decoder_assignment_distance_mean": float(decoder_assignment_distance.mean()),
            "decoder_assignment_distance_max": float(decoder_assignment_distance.max()),
        },
    }
