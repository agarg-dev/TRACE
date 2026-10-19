"""Tests for the activation-cache loader (grouping, alignment, and shard handling)."""

import json

import pytest
import torch

from activations.cache import (
    load_activation_layer,
    load_activation_sequences,
    load_cross_layer_sequences,
    read_activation_cache_info,
    validate_prompt_activation_cache,
)

DIMENSION = 3


def write_cache(directory, rows, dimension=DIMENSION, layer=20, shard_sizes=None, **info_overrides):
    """Write a cache directory: info.json, meta.jsonl, and one layer tensor (or shards)."""
    directory.mkdir(parents=True, exist_ok=True)
    info = {
        "n_tokens": len(rows),
        "layers": [layer],
        "token_scope": "prompt",
        "prefix_definition": "full_generation_prefix",
    }
    info.update(info_overrides)
    (directory / "info.json").write_text(json.dumps(info))
    (directory / "meta.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in rows)
    )
    generator = torch.Generator().manual_seed(0)
    activations = torch.randn(len(rows), dimension, generator=generator)
    if shard_sizes is None:
        torch.save(activations, directory / f"layer{layer}.pt")
    else:
        start = 0
        for number, size in enumerate(shard_sizes):
            torch.save(activations[start:start + size], directory / f"layer{layer}.part{number}.pt")
            start += size
    return activations


def meta_rows(idxs_labels, tokens_per_response):
    rows = []
    for idx, label in idxs_labels:
        for position in range(tokens_per_response):
            rows.append({"idx": idx, "label": label, "pos": position, "token_id": 100 + position})
    return rows


class TestReadActivationCacheInfo:
    def test_reads_metadata(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 1)], 2))
        info = read_activation_cache_info(tmp_path / "cache")
        assert info["n_tokens"] == 2
        assert info["layers"] == [20]

    def test_missing_metadata_raises(self, tmp_path):
        (tmp_path / "cache").mkdir()
        with pytest.raises(FileNotFoundError, match="missing activation-cache metadata"):
            read_activation_cache_info(tmp_path / "cache")


class TestValidatePromptActivationCache:
    def test_accepts_a_prompt_cache(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0)], 2))
        assert validate_prompt_activation_cache(tmp_path / "cache", 20)["token_scope"] == "prompt"

    def test_rejects_a_non_prompt_cache(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0)], 2), token_scope="response")
        with pytest.raises(ValueError, match="not prompt-only"):
            validate_prompt_activation_cache(tmp_path / "cache", 20)

    def test_rejects_an_unsupported_prefix_definition(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0)], 2), prefix_definition="other")
        with pytest.raises(ValueError, match="unsupported prefix definition"):
            validate_prompt_activation_cache(tmp_path / "cache", 20)

    def test_rejects_a_missing_layer(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0)], 2))
        with pytest.raises(ValueError, match="does not contain layer 24"):
            validate_prompt_activation_cache(tmp_path / "cache", 24)


class TestLoadActivationLayer:
    def test_loads_a_single_tensor(self, tmp_path):
        expected = write_cache(tmp_path / "cache", meta_rows([(0, 0)], 3))
        assert torch.equal(load_activation_layer(tmp_path / "cache", 20), expected)

    def test_concatenates_shards_in_order(self, tmp_path):
        expected = write_cache(tmp_path / "cache", meta_rows([(0, 0), (1, 1)], 3), shard_sizes=[2, 3, 1])
        assert torch.equal(load_activation_layer(tmp_path / "cache", 20), expected)

    def test_shard_row_count_is_verified(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0)], 3), shard_sizes=[2, 1],
                    n_tokens=9)
        with pytest.raises(ValueError, match="expected 9"):
            load_activation_layer(tmp_path / "cache", 20)


class TestLoadActivationSequences:
    def test_groups_rows_by_response(self, tmp_path):
        activations = write_cache(tmp_path / "cache", meta_rows([(0, 1), (1, 0)], 2))
        sequences = load_activation_sequences(tmp_path / "cache")
        assert [sequence["idx"] for sequence in sequences] == [0, 1]
        assert [sequence["label"] for sequence in sequences] == [1, 0]
        assert [len(sequence["x"]) for sequence in sequences] == [2, 2]
        assert sequence_token_ids(sequences) == [[100, 101], [100, 101]]
        assert torch.equal(sequences[0]["x"], activations[:2])
        assert "y" not in sequences[0]

    def test_requests_only_the_listed_ids(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0), (1, 0), (2, 0)], 2))
        sequences = load_activation_sequences(tmp_path / "cache", ids=[0, 2])
        assert [sequence["idx"] for sequence in sequences] == [0, 2]

    def test_missing_requested_id_raises(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0)], 2))
        with pytest.raises(ValueError, match="missing 1 requested response IDs"):
            load_activation_sequences(tmp_path / "cache", ids=[0, 5])

    def test_metadata_row_count_is_verified(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0), (1, 0)], 2), n_tokens=99)
        torch.save(torch.randn(3, DIMENSION), tmp_path / "cache" / "layer20.pt")
        with pytest.raises(ValueError, match="does not match the read-layer activation cache"):
            load_activation_sequences(tmp_path / "cache")

    def test_rows_must_stay_grouped_by_response(self, tmp_path):
        write_cache(tmp_path / "cache", meta_rows([(0, 0), (1, 0), (0, 0)], 2))
        sequences = load_activation_sequences(tmp_path / "cache")
        # A repeated index starts a new response instead of silently merging non-adjacent rows.
        assert [sequence["idx"] for sequence in sequences] == [0, 1, 0]


class TestLoadCrossLayerSequences:
    def test_pairs_read_and_target_rows(self, tmp_path):
        rows = meta_rows([(0, 1), (1, 0)], 2)
        write_cache(tmp_path / "read", rows)
        target = write_cache(tmp_path / "target", rows, layer=24)
        sequences = load_cross_layer_sequences(
            tmp_path / "read", tmp_path / "target", read_layer=20, target_layer=24
        )
        assert len(sequences) == 2
        assert torch.equal(sequences[0]["y"], target[:2])
        assert sequences[0]["x"].shape == sequences[0]["y"].shape

    def test_defaults_the_target_to_the_read_cache(self, tmp_path):
        rows = meta_rows([(0, 1)], 2)
        write_cache(tmp_path / "cache", rows)
        write_cache(tmp_path / "cache", rows, layer=24)
        sequences = load_cross_layer_sequences(tmp_path / "cache", read_layer=20, target_layer=24)
        assert "y" in sequences[0]

    def test_mismatched_hidden_dimensions_raise(self, tmp_path):
        rows = meta_rows([(0, 1)], 2)
        write_cache(tmp_path / "read", rows)
        write_cache(tmp_path / "target", rows, dimension=5, layer=24)
        with pytest.raises(ValueError, match="identical row and hidden dimensions"):
            load_cross_layer_sequences(tmp_path / "read", tmp_path / "target")

    def test_mismatched_row_counts_raise(self, tmp_path):
        write_cache(tmp_path / "read", meta_rows([(0, 1)], 2))
        write_cache(tmp_path / "target", meta_rows([(0, 1)], 3), layer=24)
        # Equal activation row counts but one extra metadata row: the per-row alignment catches it.
        torch.save(torch.randn(2, DIMENSION), tmp_path / "target" / "layer24.pt")
        with pytest.raises(ValueError, match="different row counts"):
            load_cross_layer_sequences(tmp_path / "read", tmp_path / "target")

    def test_activation_row_mismatch_raises(self, tmp_path):
        write_cache(tmp_path / "read", meta_rows([(0, 1)], 2))
        write_cache(tmp_path / "target", meta_rows([(0, 1)], 3), layer=24)
        with pytest.raises(ValueError, match="identical row and hidden dimensions"):
            load_cross_layer_sequences(tmp_path / "read", tmp_path / "target")

    def test_mismatched_identity_fields_raise(self, tmp_path):
        write_cache(tmp_path / "read", meta_rows([(0, 1)], 2))
        shifted = meta_rows([(0, 1)], 2)
        shifted[1]["token_id"] = 999
        write_cache(tmp_path / "target", shifted, layer=24)
        with pytest.raises(ValueError, match="differ at token row 2"):
            load_cross_layer_sequences(tmp_path / "read", tmp_path / "target")


def sequence_token_ids(sequences):
    return [sequence["token_ids"] for sequence in sequences]
