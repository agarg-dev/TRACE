"""Resolve and load cached activation sequences."""

import json
from pathlib import Path

import torch

from project_config import ACTIVATION_CACHE_DIR, READ_LAYER, TARGET_LAYER, resolve_project_path


DEFAULT_CACHE_TAG = "paper"


def read_activation_cache_info(cache_path):
    """Read lightweight cache metadata without loading activation tensors."""
    with (resolve_project_path(cache_path) / "info.json").open() as info_file:
        return json.load(info_file)


def validate_activation_cache(cache_path, dataset, split):
    """Check that a cache belongs to the requested dataset split."""
    cache_path = resolve_project_path(cache_path)
    metadata = read_activation_cache_info(cache_path)
    declared_dataset = metadata.get("dataset")
    if declared_dataset is not None and declared_dataset != dataset:
        raise ValueError(
            f"activation cache {cache_path} declares dataset {declared_dataset!r}, expected {dataset!r}"
        )
    declared_split = metadata.get("split")
    if declared_split is not None and declared_split != split.name:
        raise ValueError(
            f"activation cache {cache_path} declares split {declared_split!r}, expected {split.name!r}"
        )
    return cache_path


def resolve_activation_cache(
    dataset,
    split,
    *,
    tag=None,
    explicit_path=None,
):
    """Resolve and validate one response-activation cache."""
    if explicit_path is not None:
        cache_path = resolve_project_path(explicit_path)
    else:
        cache_path = activation_cache_output_path(dataset, split, tag=tag)
    return validate_activation_cache(cache_path, dataset, split)


def activation_cache_output_path(dataset, split, *, tag=None):
    """Choose the cache directory for one dataset split."""
    cache_tag = tag or DEFAULT_CACHE_TAG
    return ACTIVATION_CACHE_DIR / cache_tag / dataset / split.name


def resolve_vq_activation_caches(
    dataset,
    split,
    *,
    read_path=None,
    target_path=None,
):
    """Resolve the read- and target-layer activation caches."""
    if read_path is not None and target_path is None:
        # A single cache path may contain both read and target layers.
        target_path = read_path

    read_cache = resolve_activation_cache(dataset, split, explicit_path=read_path)
    target_cache = resolve_activation_cache(dataset, split, explicit_path=target_path)
    return read_cache, target_cache


def load_activation_layer(cache_dir, layer):
    """Load one cached activation layer without retaining every shard during concatenation."""
    cache_dir = resolve_project_path(cache_dir)
    metadata = read_activation_cache_info(cache_dir)
    shards = sorted(cache_dir.glob(f"layer{layer}.part*.pt"))
    if not shards:
        return torch.load(cache_dir / f"layer{layer}.pt", map_location="cpu")

    expected_rows = int(metadata["n_tokens"])
    first_shard = torch.load(shards[0], map_location="cpu")
    activations = torch.empty((expected_rows, *first_shard.shape[1:]), dtype=first_shard.dtype)
    next_row = len(first_shard)
    activations[:next_row].copy_(first_shard)
    del first_shard

    for shard_path in shards[1:]:
        shard = torch.load(shard_path, map_location="cpu")
        end_row = next_row + len(shard)
        activations[next_row:end_row].copy_(shard)
        next_row = end_row
        del shard
    if next_row != expected_rows:
        raise ValueError(f"activation shards contain {next_row} rows, expected {expected_rows}")
    return activations


def _load_sequences(read_cache_dir, read_layer, target_cache_dir=None, target_layer=None, ids=None):
    """Group activation rows by response while retaining tensor-slice views."""
    read_cache_dir = resolve_project_path(read_cache_dir)
    target_cache_dir = resolve_project_path(target_cache_dir) if target_cache_dir is not None else None
    read_activations = load_activation_layer(read_cache_dir, read_layer)
    target_activations = (
        load_activation_layer(target_cache_dir, target_layer)
        if target_cache_dir is not None else None
    )
    if target_activations is not None and (
        len(read_activations) != len(target_activations)
        or read_activations.shape[1] != target_activations.shape[1]
    ):
        raise ValueError("read and target activation layers must have identical row and hidden dimensions")

    requested_ids = None if ids is None else {int(response_id) for response_id in ids}
    sequences, selected_ids = [], set()
    response_id = label = start_row = None
    token_ids = []

    def add_response(end_row):
        if response_id is None:
            return
        if requested_ids is not None and response_id not in requested_ids:
            return
        sequence = {
            "idx": response_id,
            "label": label,
            "x": read_activations[start_row:end_row],
            "token_ids": token_ids,
        }
        if target_activations is not None:
            sequence["y"] = target_activations[start_row:end_row]
        sequences.append(sequence)
        selected_ids.add(response_id)

    with (read_cache_dir / "meta.jsonl").open() as metadata_file:
        metadata_count = 0
        for row_number, line in enumerate(metadata_file):
            metadata = json.loads(line)
            next_response_id = int(metadata["idx"])
            if next_response_id != response_id:
                add_response(row_number)
                response_id = next_response_id
                label = int(metadata["label"])
                token_ids = []
                start_row = row_number
            token_ids.append(int(metadata["token_id"]))
            metadata_count = row_number + 1

    add_response(metadata_count)
    if metadata_count != len(read_activations):
        raise ValueError("meta.jsonl row count does not match the read-layer activation cache")
    if requested_ids is not None and selected_ids != requested_ids:
        missing = sorted(requested_ids - selected_ids)
        raise ValueError(f"activation cache is missing {len(missing)} requested response IDs")
    return sequences


def load_cross_layer_sequences(
    read_cache_dir, target_cache_dir=None, read_layer=READ_LAYER, target_layer=TARGET_LAYER, ids=None
):
    """Preload aligned layers once and expose each response as tensor-slice views."""
    read_cache_dir = Path(read_cache_dir)
    target_cache_dir = Path(target_cache_dir) if target_cache_dir is not None else read_cache_dir
    return _load_sequences(read_cache_dir, read_layer, target_cache_dir, target_layer, ids)


def load_activation_sequences(cache_dir, layer=READ_LAYER, ids=None):
    """Preload one activation layer and expose each response as a tensor-slice view."""
    return _load_sequences(cache_dir, layer, ids=ids)
