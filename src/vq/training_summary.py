"""Summaries of code usage and harmfulness regions during VQ training."""

import numpy as np

from vq.codebook import (
    assign_codes_for_sequences,
    smoothed_response_code_statistics,
    split_assignments_by_response,
)


def summarize_initial_codebook(model, training_sequences, initialization_info, device, prior_strength):
    """Summarize initial code usage and derive the first harmful and benign regions."""
    assignments = assign_codes_for_sequences(model, training_sequences, device)
    response_assignments = split_assignments_by_response(training_sequences, assignments)
    response_labels = [sequence["label"] for sequence in training_sequences]
    region_statistics = smoothed_response_code_statistics(
        response_assignments, response_labels, model.quantizer.num_codes, prior_strength
    )

    num_codes = model.quantizer.num_codes
    cluster_sizes = np.bincount(assignments, minlength=num_codes)
    harmful_codes = np.flatnonzero(region_statistics["signed_harmfulness"] > 0).tolist()
    harmful_code_set = set(harmful_codes)
    benign_codes = [code for code in range(num_codes) if code not in harmful_code_set]
    regions = {
        "benign_codes": benign_codes,
        "harmful_codes": harmful_codes,
        "base_rate": region_statistics["base_harmful_response_rate"],
        "source": "smoothed_response_enrichment",
        "prior_strength": prior_strength,
        "signed_harmfulness": region_statistics["signed_harmfulness"],
        "response_counts": region_statistics["response_counts"],
    }
    statistics = {
        "K": num_codes,
        "n_train_tokens": int(len(assignments)),
        "init": initialization_info["init"],
        "base_rate": float(region_statistics["base_harmful_response_rate"]),
        "code_score_method": "response_presence",
        "code_score_prior_strength": prior_strength,
        "n_empty_codes": int((cluster_sizes == 0).sum()),
        "cluster_size_min_med_max": [
            int(cluster_sizes.min()), int(np.median(cluster_sizes)), int(cluster_sizes.max())
        ],
        "enrichment_split": {"n_harmful": len(harmful_codes), "n_benign": len(benign_codes)},
    }
    summary = (
        f"init clusters: {statistics['n_empty_codes']}/{num_codes} empty | "
        f"sizes min/med/max {statistics['cluster_size_min_med_max']} | "
        f"regions {len(benign_codes)} benign/{len(harmful_codes)} harmful"
    )
    return statistics, summary, regions


def summarize_region_changes(statistics, harmful_codes, previous_harmful_codes=None):
    """Count region changes and scores near the harmful/benign boundary."""
    harmful_codes = set(harmful_codes)
    previous_harmful_codes = harmful_codes if previous_harmful_codes is None else previous_harmful_codes
    signed_scores = np.asarray(statistics["signed_harmfulness"], dtype=np.float64)
    return {
        "n_changed_codes": len(harmful_codes.symmetric_difference(previous_harmful_codes)),
        "n_scores_within_0.01_of_boundary": int((np.abs(signed_scores) <= 0.01).sum()),
    }
