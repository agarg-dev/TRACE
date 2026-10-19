"""Initialize the VQ codebook, assign activations, and estimate code harmfulness."""

import os

import numpy as np
import torch

MAX_INITIALIZATION_VECTORS = 200_000
FAISS_ITERATIONS = 20
FAISS_RESTARTS = 5

# Code-score methods accepted by the training CLI and stored in VQ checkpoints.
CODE_SCORE_METHODS = ("response_presence", "response_frequency", "token_occurrence")

# The subset of methods that score a code from its per-response enrichment.
RESPONSE_SCORE_METHODS = ("response_presence", "response_frequency")

# Region-source label recorded in the checkpoint for each score method.
CODE_SCORE_REGION_SOURCES = {
    "response_presence": "smoothed_response_enrichment",
    "response_frequency": "smoothed_response_frequency_enrichment",
    "token_occurrence": "token_occurrence_enrichment",
}

# Floor applied to the score normalizers. Without it a partition whose base rate
# is exactly 0 or 1 divides by zero and yields NaN scores.
SCORE_DENOMINATOR_FLOOR = 1e-12


def _require_faiss():
    """Import FAISS on demand so the codebook statistics stay importable without it."""
    try:
        import faiss
    except ImportError as error:
        raise ModuleNotFoundError(
            "codebook initialization requires faiss-cpu; install it with `pip install faiss-cpu`"
        ) from error
    return faiss


def _validate_num_codes(num_codes):
    num_codes = int(num_codes)
    if num_codes <= 0:
        raise ValueError(f"num_codes must be positive, got {num_codes}")
    return num_codes


def signed_harmfulness_scores(harmful_probability, base_rate):
    """Turn a code's harmful probability into a signed enrichment score.

    Positive scores mark codes that fire more often in harmful responses than the
    partition's base rate, negative scores mark the opposite. Both normalizers are
    floored so that a degenerate partition (base rate 0 or 1) scores every code 0
    instead of producing NaN, which would silently blank out the region split.
    """
    harmful_probability = np.asarray(harmful_probability, dtype=np.float64)
    base_rate = float(base_rate)
    difference_from_base = harmful_probability - base_rate
    above_base = difference_from_base / max(SCORE_DENOMINATOR_FLOOR, 1.0 - base_rate)
    below_base = difference_from_base / max(SCORE_DENOMINATOR_FLOOR, base_rate)
    return np.where(difference_from_base >= 0, above_base, below_base)


def code_score_region_source(score_method):
    """Return the checkpoint region-source label for a score method."""
    try:
        return CODE_SCORE_REGION_SOURCES[score_method]
    except KeyError as error:
        raise ValueError(
            f"unknown code score method: {score_method!r}; expected one of {CODE_SCORE_METHODS}"
        ) from error


def code_harmfulness_statistics(
    score_method, code_sequences, response_labels, num_codes, prior_strength=10.0
):
    """Dispatch to the requested code-harmfulness estimator.

    Unknown methods raise instead of silently falling back to token occurrence, which used to
    train regions with a different estimator than the one recorded in the checkpoint.
    """
    if score_method == "response_presence":
        return smoothed_response_code_statistics(
            code_sequences, response_labels, num_codes, prior_strength
        )
    if score_method == "response_frequency":
        return smoothed_response_frequency_statistics(
            code_sequences, response_labels, num_codes, prior_strength
        )
    if score_method == "token_occurrence":
        return token_occurrence_code_statistics(code_sequences, response_labels, num_codes)
    raise ValueError(
        f"unknown code score method: {score_method!r}; expected one of {CODE_SCORE_METHODS}"
    )



def _flat_float_sequence_activations(sequence, dimension):
    activations = sequence["x"].detach().cpu().float()
    return activations.reshape(-1, dimension)


def _nonzero_sequence_row_counts(sequences, dimension):
    counts = []
    for sequence in sequences:
        activations = _flat_float_sequence_activations(sequence, dimension)
        counts.append(int(torch.any(activations != 0, dim=1).sum()))
    return counts


def _gather_nonzero_sequence_rows(sequences, dimension, counts, selected_rows):
    """Map selected global token rows back to their response tensors."""
    selected_rows = torch.as_tensor(selected_rows, dtype=torch.long)
    if len(selected_rows) == 0:
        return torch.empty((0, dimension), dtype=torch.float32)

    sorted_order = torch.argsort(selected_rows)
    sorted_rows = selected_rows[sorted_order]
    gathered = torch.empty((len(selected_rows), dimension), dtype=torch.float32)
    global_start = 0
    for sequence, count in zip(sequences, counts):
        global_end = global_start + count
        lower = int(torch.searchsorted(sorted_rows, global_start, right=False))
        upper = int(torch.searchsorted(sorted_rows, global_end, right=False))
        if lower < upper:
            activations = _flat_float_sequence_activations(sequence, dimension)
            nonzero_rows = torch.nonzero(torch.any(activations != 0, dim=1), as_tuple=False).flatten()
            local_nonzero_rows = sorted_rows[lower:upper] - global_start
            output_rows = sorted_order[lower:upper]
            gathered[output_rows] = activations[nonzero_rows[local_nonzero_rows]]
        global_start = global_end
    return gathered


def _sample_nonzero_sequence_vectors(sequences, dimension, maximum, seed):
    """Select bounded vectors without concatenating the full activation collection."""
    counts = _nonzero_sequence_row_counts(sequences, dimension)
    total_rows = sum(counts)
    if total_rows > maximum:
        generator = torch.Generator().manual_seed(seed)
        selected_rows = torch.randperm(total_rows, generator=generator)[:maximum]
    else:
        selected_rows = torch.arange(total_rows)
    return _gather_nonzero_sequence_rows(sequences, dimension, counts, selected_rows), total_rows


def _set_codebook(model, centroids):
    """Set code vectors and synchronize the EMA buffers used during training."""
    centroids = torch.as_tensor(np.ascontiguousarray(centroids), dtype=torch.float)
    quantizer = model.quantizer
    with torch.no_grad():
        quantizer._embedding.weight.data.copy_(centroids)
        quantizer._ema_w.data.copy_(centroids)
        quantizer._ema_cluster_size.data.fill_(1.0)


def _spherical_kmeans(vectors, num_clusters, seed):
    """Cluster normalized vectors, then restore each cluster's mean input norm."""
    faiss = _require_faiss()
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    fit_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    fit_vectors = np.ascontiguousarray(fit_vectors, dtype="float32")
    use_gpu = os.environ.get("CS_FAISS_GPU", "1") != "0" and faiss.get_num_gpus() > 0

    kmeans = faiss.Kmeans(
        vectors.shape[1], num_clusters, niter=FAISS_ITERATIONS, nredo=FAISS_RESTARTS,
        seed=int(seed), gpu=use_gpu, verbose=False,
    )
    kmeans.train(fit_vectors)
    centroids = np.asarray(kmeans.centroids, dtype="float32").reshape(num_clusters, vectors.shape[1]).copy()

    _, membership = kmeans.index.search(fit_vectors, 1)
    membership = membership.ravel()
    fallback_norm = float(np.linalg.norm(vectors, axis=1).mean())
    for code in range(num_clusters):
        members = vectors[membership == code]
        mean_norm = float(np.linalg.norm(members, axis=1).mean()) if len(members) else fallback_norm
        centroids[code] = centroids[code] / (np.linalg.norm(centroids[code]) + 1e-8) * mean_norm
    return centroids


def initialize_codebook_from_sequences(model, training_sequences, seed):
    """Initialize the codebook with spherical k-means over sampled training activations."""
    dimension = model.quantizer.code_dimension
    num_codes = model.quantizer.num_codes
    vectors, num_vectors = _sample_nonzero_sequence_vectors(
        training_sequences, dimension, MAX_INITIALIZATION_VECTORS, seed
    )
    vectors = model.normalize_read_activations(vectors)
    if num_vectors < num_codes:
        codebook = torch.zeros(num_codes, dimension)
        codebook[:num_vectors] = vectors
        _set_codebook(model, codebook.numpy())
    else:
        centroids = _spherical_kmeans(vectors.numpy(), num_codes, seed)
        _set_codebook(model, centroids)

    faiss_max_points_per_centroid = int(_require_faiss().ClusteringParameters().max_points_per_centroid)
    num_faiss_optimization_vectors = min(len(vectors), num_codes * faiss_max_points_per_centroid)
    return {
        "init": "spherical",
        "n_available_vectors": int(num_vectors),
        "n_selected_vectors": int(len(vectors)),
        "faiss_max_points_per_centroid": faiss_max_points_per_centroid,
        "n_faiss_optimization_vectors": int(num_faiss_optimization_vectors),
    }


@torch.no_grad()
def assign_raw_activations_to_codes(model, activations, device, chunk_size=20_000):
    """Assign raw activations directly to their nearest codebook vectors."""
    codebook = model.quantizer.codebook.detach().float().to(device)
    codebook_norms = codebook.pow(2).sum(1)
    assignments = np.empty(activations.shape[0], dtype=np.int64)
    for start in range(0, activations.shape[0], chunk_size):
        block = activations[start:start + chunk_size].to(device=device, dtype=torch.float32)
        block = model.normalize_read_activations(block)
        squared_distance = block.pow(2).sum(1, keepdim=True) - 2 * block @ codebook.T + codebook_norms
        assignments[start:start + chunk_size] = squared_distance.argmin(1).cpu().numpy()
    return assignments


@torch.no_grad()
def encode_and_assign_activations(model, activations):
    """Encode activations and return their nearest VQ codes."""
    encoded_activations = model.encode_activations(activations)
    codebook = model.quantizer.codebook
    squared_distance = (
        encoded_activations.pow(2).sum(1, keepdim=True)
        - 2 * encoded_activations @ codebook.t()
        + codebook.pow(2).sum(1)
    )
    return encoded_activations, squared_distance.argmin(1)


@torch.no_grad()
def nearest_code_indices(model, activations):
    """Return the nearest encoded-space code for each activation."""
    return encode_and_assign_activations(model, activations)[1]


@torch.no_grad()
def assign_encoded_activations_to_codes(model, activations, device, chunk_size=20_000):
    """Encode flat activations and assign their nearest codes in bounded chunks."""
    assignments = np.empty(activations.shape[0], dtype=np.int64)
    for start in range(0, activations.shape[0], chunk_size):
        block = activations[start:start + chunk_size].to(device=device, dtype=torch.float32)
        assignments[start:start + chunk_size] = nearest_code_indices(model, block).cpu().numpy()
    return assignments


def assign_codes_for_sequences(model, sequences, device, activations=None, assignment_space="raw_activation"):
    """Assign concatenated response activations in raw or encoded representation space."""
    expected_tokens = sum(len(sequence["x"]) for sequence in sequences)
    if assignment_space == "raw_activation":
        assign_block = assign_raw_activations_to_codes
    elif assignment_space == "encoded_activation":
        assign_block = assign_encoded_activations_to_codes
    else:
        raise ValueError(f"unknown assignment space: {assignment_space}")

    # Stream response views into bounded blocks when no flat activation tensor is supplied.
    if activations is None:
        assignments = np.empty(expected_tokens, dtype=np.int64)
        pending, pending_tokens, assigned_tokens = [], 0, 0
        for sequence in sequences:
            sequence_activations = sequence["x"]
            sequence_start = 0
            while sequence_start < len(sequence_activations):
                take = min(20_000 - pending_tokens, len(sequence_activations) - sequence_start)
                pending.append(sequence_activations[sequence_start:sequence_start + take])
                pending_tokens += take
                sequence_start += take
                if pending_tokens == 20_000:
                    block = torch.cat(pending, dim=0).float()
                    assignments[assigned_tokens:assigned_tokens + pending_tokens] = assign_block(
                        model, block, device, chunk_size=20_000
                    )
                    assigned_tokens += pending_tokens
                    pending, pending_tokens = [], 0
        if pending_tokens:
            block = torch.cat(pending, dim=0).float()
            assignments[assigned_tokens:assigned_tokens + pending_tokens] = assign_block(
                model, block, device, chunk_size=20_000
            )
            assigned_tokens += pending_tokens
        return assignments

    return assign_block(model, activations, device)


def split_assignments_by_response(sequences, assignments):
    """Split flat token assignments back into response sequences."""
    response_assignments, start = [], 0
    for sequence in sequences:
        end = start + len(sequence["x"])
        response_assignments.append(assignments[start:end])
        start = end
    return response_assignments


def smoothed_response_code_statistics(code_sequences, response_labels, num_codes, prior_strength=10.0):
    """Estimate code harmfulness from response presence, counting each code once per response."""
    num_codes = _validate_num_codes(num_codes)
    code_sequences = list(code_sequences)
    response_labels = np.asarray(list(response_labels), dtype=np.int8)
    response_counts = np.zeros(num_codes, dtype=np.int64)
    harmful_response_counts = np.zeros(num_codes, dtype=np.int64)
    for codes, label in zip(code_sequences, response_labels):
        codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        present_codes = np.unique(codes)
        response_counts[present_codes] += 1
        if label == 1:
            harmful_response_counts[present_codes] += 1

    base_rate = float(response_labels.mean()) if response_labels.size else 0.0
    harmful_probability = (
        harmful_response_counts + prior_strength * base_rate
    ) / (response_counts + prior_strength)
    signed_harmfulness = signed_harmfulness_scores(harmful_probability, base_rate)
    max_count = max(1, int(response_counts.max()))
    normalized_support = np.log1p(response_counts) / np.log1p(max_count)
    return {
        "base_harmful_response_rate": base_rate,
        "n_responses": int(len(response_labels)),
        "n_harmful_responses": int(response_labels.sum()),
        "n_safe_responses": int((response_labels == 0).sum()),
        "response_counts": response_counts,
        "harmful_response_counts": harmful_response_counts,
        "smoothed_harmful_probability": harmful_probability,
        "signed_harmfulness": signed_harmfulness,
        "normalized_log_response_support": normalized_support,
        "prior_strength": float(prior_strength),
    }


def smoothed_response_frequency_statistics(code_sequences, response_labels, num_codes, prior_strength=10.0):
    """Estimate code harmfulness from per-response code frequencies with unit total response mass."""
    num_codes = _validate_num_codes(num_codes)
    code_sequences = list(code_sequences)
    response_labels = np.asarray(list(response_labels), dtype=np.int8)
    response_counts = np.zeros(num_codes, dtype=np.int64)
    response_frequency_mass = np.zeros(num_codes, dtype=np.float64)
    harmful_response_frequency_mass = np.zeros(num_codes, dtype=np.float64)
    for codes, label in zip(code_sequences, response_labels):
        codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        if not len(codes):
            continue
        counts = np.bincount(codes, minlength=num_codes)
        frequencies = counts / len(codes)
        response_counts[counts > 0] += 1
        response_frequency_mass += frequencies
        if label == 1:
            harmful_response_frequency_mass += frequencies

    base_rate = float(response_labels.mean()) if response_labels.size else 0.0
    harmful_probability = (
        harmful_response_frequency_mass + prior_strength * base_rate
    ) / (response_frequency_mass + prior_strength)
    signed_harmfulness = signed_harmfulness_scores(harmful_probability, base_rate)
    max_count = max(1, int(response_counts.max()))
    normalized_support = np.log1p(response_counts) / np.log1p(max_count)
    return {
        "base_harmful_response_rate": base_rate,
        "n_responses": int(len(response_labels)),
        "n_harmful_responses": int(response_labels.sum()),
        "n_safe_responses": int((response_labels == 0).sum()),
        "response_counts": response_counts,
        "response_frequency_mass": response_frequency_mass,
        "harmful_response_frequency_mass": harmful_response_frequency_mass,
        "smoothed_harmful_probability": harmful_probability,
        "signed_harmfulness": signed_harmfulness,
        "normalized_log_response_support": normalized_support,
        "prior_strength": float(prior_strength),
    }


def token_occurrence_code_statistics(code_sequences, response_labels, num_codes):
    """Reproduce the original score: every code occurrence inherits its response label."""
    num_codes = _validate_num_codes(num_codes)
    code_sequences = [np.asarray(codes, dtype=np.int64).reshape(-1) for codes in code_sequences]
    response_labels = np.asarray(list(response_labels), dtype=np.int8)
    token_counts = np.zeros(num_codes, dtype=np.int64)
    harmful_token_counts = np.zeros(num_codes, dtype=np.int64)
    total_tokens = 0
    total_harmful_tokens = 0
    for codes, label in zip(code_sequences, response_labels):
        counts = np.bincount(codes, minlength=num_codes)
        token_counts += counts
        if label == 1:
            harmful_token_counts += counts
            total_harmful_tokens += len(codes)
        total_tokens += len(codes)

    # An empty partition carries no token base rate, so every code scores as uninformative.
    base_rate = total_harmful_tokens / total_tokens if total_tokens else 0.0
    harmful_probability = np.divide(
        harmful_token_counts, token_counts,
        out=np.full(num_codes, base_rate, dtype=np.float64), where=token_counts > 0,
    )
    signed_harmfulness = signed_harmfulness_scores(harmful_probability, base_rate)
    return {
        "base_harmful_token_rate": float(base_rate),
        "n_tokens": int(total_tokens),
        "n_harmful_tokens": int(total_harmful_tokens),
        "n_safe_tokens": int(total_tokens - total_harmful_tokens),
        "token_counts": token_counts,
        "harmful_token_counts": harmful_token_counts,
        "harmful_probability": harmful_probability,
        "signed_harmfulness": signed_harmfulness,
    }
