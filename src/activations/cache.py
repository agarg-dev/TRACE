"""Load cached activations and group them by response."""

import json
from itertools import zip_longest
from pathlib import Path

import torch

from project_config import READ_LAYER, TARGET_LAYER


def read_activation_cache_info(cache_path):
    """Read lightweight cache metadata without loading activation tensors."""
    cache_path = Path(cache_path)
    info_path = cache_path / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing activation-cache metadata: {info_path}")
    with info_path.open() as info_file:
        return json.load(info_file)


def validate_prompt_activation_cache(cache_path, required_layer):
    """Check that a cache contains formatted prompt activations at the requested layer."""
    metadata = read_activation_cache_info(cache_path)
    if metadata.get("token_scope") != "prompt":
        raise ValueError(f"activation cache is not prompt-only: {cache_path}")
    if metadata.get("prefix_definition") != "full_generation_prefix":
        raise ValueError(f"prompt cache has an unsupported prefix definition: {cache_path}")
    layers = {int(layer) for layer in metadata.get("layers", [])}
    if required_layer not in layers:
        raise ValueError(f"prompt cache does not contain layer {required_layer}: {cache_path}")
    return metadata


def load_activation_layer(cache_dir, layer):
    """Load one cached activation layer without retaining every shard during concatenation."""
    cache_dir = Path(cache_dir)
    shards = sorted(cache_dir.glob(f"layer{layer}.part*.pt"))
    if not shards:
        return torch.load(cache_dir / f"layer{layer}.pt", map_location="cpu")

    with (cache_dir / "info.json").open() as info_file:
        expected_rows = int(json.load(info_file)["n_tokens"])

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


def _aligned_metadata(read_cache_dir, target_cache_dir=None):
    """Yield read-cache metadata while verifying an optional target cache row by row."""
    read_path = Path(read_cache_dir) / "meta.jsonl"
    if target_cache_dir is None or Path(target_cache_dir).resolve() == Path(read_cache_dir).resolve():
        with read_path.open() as metadata_file:
            for line in metadata_file:
                yield json.loads(line)
        return

    target_path = Path(target_cache_dir) / "meta.jsonl"
    identity_fields = ("idx", "label", "pos", "token_id")
    with read_path.open() as read_file, target_path.open() as target_file:
        for row_number, lines in enumerate(zip_longest(read_file, target_file), start=1):
            read_line, target_line = lines
            if read_line is None or target_line is None:
                raise ValueError("read and target activation metadata have different row counts")
            read_metadata, target_metadata = json.loads(read_line), json.loads(target_line)
            read_identity = tuple(read_metadata.get(field) for field in identity_fields)
            target_identity = tuple(target_metadata.get(field) for field in identity_fields)
            if read_identity != target_identity:
                raise ValueError(
                    f"read and target activation metadata differ at token row {row_number}"
                )
            yield read_metadata


def _load_sequences(read_cache_dir, read_layer, target_cache_dir=None, target_layer=None, ids=None):
    """Group activation rows by response while retaining tensor-slice views."""
    # Open the read layer and, when requested, its aligned reconstruction target.
    read_cache_dir = Path(read_cache_dir)
    target_cache_dir = Path(target_cache_dir) if target_cache_dir is not None else None
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
    current_index = current_label = start_row = None
    current_token_ids = []

    # Finish one response whenever its ID changes in the token metadata.
    def finish_response(end_row):
        if current_index is None:
            return
        if requested_ids is not None and current_index not in requested_ids:
            return
        sequence = {
            "idx": current_index,
            "label": current_label,
            "x": read_activations[start_row:end_row],
            "token_ids": current_token_ids,
        }
        if target_activations is not None:
            sequence["y"] = target_activations[start_row:end_row]
        sequences.append(sequence)
        selected_ids.add(current_index)

    # The cache stores flat token rows, so rebuild response boundaries from meta.jsonl.
    metadata_rows = 0
    for row_number, metadata in enumerate(
        _aligned_metadata(read_cache_dir, target_cache_dir), start=0
    ):
        response_index = int(metadata["idx"])
        if current_index is None or response_index != current_index:
            finish_response(row_number)
            current_index = response_index
            current_label = int(metadata["label"])
            current_token_ids = []
            start_row = row_number
        current_token_ids.append(int(metadata["token_id"]))
        metadata_rows = row_number + 1

    finish_response(metadata_rows)
    if metadata_rows != len(read_activations):
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
