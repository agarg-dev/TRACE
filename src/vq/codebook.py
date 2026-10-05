"""Initialize the VQ codebook, assign activations, and estimate code harmfulness."""

import faiss
import numpy as np
import torch

MAX_INITIALIZATION_VECTORS = 200_000
FAISS_ITERATIONS = 20
FAISS_RESTARTS = 5


def _faiss_gpu_available():
    get_num_gpus = getattr(faiss, "get_num_gpus", None)
    return torch.cuda.is_available() and get_num_gpus is not None and get_num_gpus() > 0


def _sample_initialization_vectors(sequences, dimension, maximum, seed):
    """Sample nonzero activation rows without joining the full collection."""
    counts = []
    for sequence in sequences:
        activations = sequence["x"].detach().cpu().float().reshape(-1, dimension)
        counts.append(int(torch.any(activations != 0, dim=1).sum()))

    total_rows = sum(counts)
    if total_rows > maximum:
        generator = torch.Generator().manual_seed(seed)
        selected_rows = torch.randperm(total_rows, generator=generator)[:maximum]
    else:
        selected_rows = torch.arange(total_rows)

    selected_rows = torch.as_tensor(selected_rows, dtype=torch.long)
    sorted_order = torch.argsort(selected_rows)
    sorted_rows = selected_rows[sorted_order]
    vectors = torch.empty((len(selected_rows), dimension), dtype=torch.float32)
    global_start = 0
    for sequence, count in zip(sequences, counts):
        global_end = global_start + count
        lower = int(torch.searchsorted(sorted_rows, global_start))
        upper = int(torch.searchsorted(sorted_rows, global_end))
        if lower < upper:
            activations = sequence["x"].detach().cpu().float().reshape(-1, dimension)
            nonzero_rows = torch.nonzero(torch.any(activations != 0, dim=1)).flatten()
            local_rows = sorted_rows[lower:upper] - global_start
            vectors[sorted_order[lower:upper]] = activations[nonzero_rows[local_rows]]
        global_start = global_end
    return vectors, total_rows


def _set_codebook(model, centroids):
    centroids = torch.as_tensor(np.ascontiguousarray(centroids), dtype=torch.float)
    quantizer = model.quantizer
    with torch.no_grad():
        quantizer.codebook.copy_(centroids)
        quantizer._ema_w.copy_(centroids)
        quantizer._ema_cluster_size.fill_(1.0)


def _spherical_kmeans(vectors, num_clusters, seed):
    """Cluster unit vectors and restore each cluster's mean input norm."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    fit_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    fit_vectors = np.ascontiguousarray(fit_vectors, dtype="float32")
    kmeans = faiss.Kmeans(
        vectors.shape[1], num_clusters, niter=FAISS_ITERATIONS, nredo=FAISS_RESTARTS,
        seed=int(seed), gpu=_faiss_gpu_available(), verbose=False,
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


def initialize_spherical_codebook(model, training_sequences, seed):
    """Initialize the codebook with spherical k-means over training activations."""
    dimension = model.quantizer.code_dimension
    num_codes = model.quantizer.num_codes
    vectors, num_vectors = _sample_initialization_vectors(
        training_sequences, dimension, MAX_INITIALIZATION_VECTORS, seed
    )
    if num_vectors < num_codes:
        codebook = torch.zeros(num_codes, dimension)
        codebook[:num_vectors] = vectors
        _set_codebook(model, codebook.numpy())
    else:
        _set_codebook(model, _spherical_kmeans(vectors.numpy(), num_codes, seed))

    max_points = int(faiss.ClusteringParameters().max_points_per_centroid)
    return {
        "init": "spherical",
        "n_available_vectors": int(num_vectors),
        "n_selected_vectors": int(len(vectors)),
        "faiss_max_points_per_centroid": max_points,
        "n_faiss_optimization_vectors": min(len(vectors), num_codes * max_points),
    }


@torch.no_grad()
def assign_raw_activations_to_codes(model, activations, device, chunk_size=20_000):
    """Assign raw activations directly to their nearest codebook vectors."""
    codebook = model.quantizer.codebook.detach().float().to(device)
    codebook_norms = codebook.pow(2).sum(1)
    assignments = np.empty(activations.shape[0], dtype=np.int64)
    for start in range(0, activations.shape[0], chunk_size):
        block = activations[start:start + chunk_size].to(device=device, dtype=torch.float32)
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


def assign_codes_for_sequences(model, sequences, device, assignment_space="raw_activation"):
    """Assign concatenated response activations in raw or encoded representation space."""
    expected_tokens = sum(len(sequence["x"]) for sequence in sequences)
    if assignment_space == "raw_activation":
        assign_block = assign_raw_activations_to_codes
    elif assignment_space == "encoded_activation":
        assign_block = assign_encoded_activations_to_codes
    else:
        raise ValueError(f"unknown assignment space: {assignment_space}")

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
    return assignments


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
    code_sequences = list(code_sequences)
    response_labels = np.asarray(list(response_labels), dtype=np.int8)
    if len(code_sequences) != len(response_labels):
        raise ValueError("code_sequences and response_labels must contain the same number of responses")
    if not len(code_sequences):
        raise ValueError("cannot derive code statistics from an empty response collection")
    if num_codes < 1 or prior_strength <= 0:
        raise ValueError("num_codes and prior_strength must be positive")
    if not np.isin(response_labels, [0, 1]).all():
        raise ValueError("response labels must be 0 or 1")

    response_counts = np.zeros(num_codes, dtype=np.int64)
    harmful_response_counts = np.zeros(num_codes, dtype=np.int64)
    for codes, label in zip(code_sequences, response_labels):
        codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        if len(codes) and (codes.min() < 0 or codes.max() >= num_codes):
            raise ValueError(f"code IDs must be in [0, {num_codes})")
        present_codes = np.unique(codes)
        response_counts[present_codes] += 1
        if label == 1:
            harmful_response_counts[present_codes] += 1

    base_rate = float(response_labels.mean())
    if not 0.0 < base_rate < 1.0:
        raise ValueError("code statistics require both safe and harmful responses")
    harmful_probability = (
        harmful_response_counts + prior_strength * base_rate
    ) / (response_counts + prior_strength)
    difference_from_base = harmful_probability - base_rate
    signed_harmfulness = np.where(
        difference_from_base >= 0,
        difference_from_base / (1 - base_rate),
        difference_from_base / base_rate,
    )
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
