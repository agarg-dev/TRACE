"""Tests for dataset split selection and the stratified train/validation splitter."""

import numpy as np
import pytest

from data.dataset_splits import (
    DEFAULT_DATASET,
    S_EVAL_DATASET,
    WILDGUARD_INTERNLM_DATASET,
    _TEST_SPLITS,
    _TRAIN_SPLITS,
    dataset_split,
    evaluation_split,
    train_validation_split,
    training_split,
)


class TestSplitSelection:
    def test_default_training_split(self):
        split = training_split()
        assert split.name == "train"
        assert split.kind == "train"
        assert split.id_key == "idx"
        assert split.max_response_tokens == 2048
        assert split.source.name == "train_stratified.json"
        assert DEFAULT_DATASET in str(split.source)

    def test_default_evaluation_split(self):
        split = evaluation_split()
        assert split.name == "test"
        assert split.kind == "test"
        assert split.id_key == "test_index"
        assert split.source.name == "test_full.jsonl"

    def test_classifier_train_split_is_a_training_partition(self):
        split = training_split("classifier_train")
        assert split.source.name == "train_deduplicated.jsonl"

    def test_small_training_split_has_a_shorter_token_budget(self):
        assert training_split("train_small").max_response_tokens == 1024

    def test_dataset_selection_changes_the_source_path(self):
        split = training_split("train", WILDGUARD_INTERNLM_DATASET)
        assert WILDGUARD_INTERNLM_DATASET in str(split.source)

    def test_s_eval_dataset_uses_its_own_files(self):
        split = training_split("train", S_EVAL_DATASET)
        assert split.source.name == "train.jsonl"
        assert evaluation_split("test", S_EVAL_DATASET).source.name == "test.jsonl"

    def test_splits_are_frozen(self):
        with pytest.raises(Exception):
            training_split().name = "other"


class TestDatasetSplit:
    def test_prefers_training_partitions(self):
        assert dataset_split("train").kind == "train"
        assert dataset_split("classifier_train").kind == "train"

    def test_falls_back_to_evaluation_partitions(self):
        assert dataset_split("test").kind == "test"

    def test_unknown_split_lists_the_available_ones(self):
        """Regression: the error used to be a bare KeyError from a nested dict lookup."""
        with pytest.raises(KeyError, match="available splits"):
            training_split("train_smallz")

    def test_unknown_split_name_for_evaluation_partition(self):
        with pytest.raises(KeyError, match="unknown test split"):
            evaluation_split("validation")

    def test_unknown_dataset_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="unknown dataset"):
            training_split("train", "not_a_dataset")

    def test_unknown_dataset_through_dataset_split(self):
        with pytest.raises(KeyError, match="unknown dataset"):
            dataset_split("train", "not_a_dataset")

    def test_split_tables_cover_the_same_datasets(self):
        assert set(_TRAIN_SPLITS) == set(_TEST_SPLITS)

    def test_every_declared_split_resolves(self):
        for dataset in _TRAIN_SPLITS:
            for name in _TRAIN_SPLITS[dataset]:
                assert training_split(name, dataset).kind == "train"
            for name in _TEST_SPLITS[dataset]:
                assert evaluation_split(name, dataset).kind == "test"


def make_labelled_sequences(num_per_label=10):
    sequences = []
    for label in (0, 1):
        for index in range(num_per_label):
            sequences.append({"idx": label * 100 + index, "label": label, "x": np.zeros((2, 3))})
    return sequences


class TestTrainValidationSplit:
    def test_splits_are_disjoint_and_complete(self):
        sequences = make_labelled_sequences()
        training, validation = train_validation_split(sequences, seed=42, validation_fraction=0.1)
        training_ids = {record["idx"] for record in training}
        validation_ids = {record["idx"] for record in validation}
        assert not training_ids & validation_ids
        assert training_ids | validation_ids == {record["idx"] for record in sequences}
        assert len(validation) == 2

    def test_stratified_by_label(self):
        training, validation = train_validation_split(
            make_labelled_sequences(), seed=42, validation_fraction=0.3
        )
        assert sum(record["label"] == 0 for record in validation) == 3
        assert sum(record["label"] == 1 for record in validation) == 3
        assert sum(record["label"] == 1 for record in training) == 7

    def test_is_reproducible_for_a_fixed_seed(self):
        sequences = make_labelled_sequences()
        first = train_validation_split(sequences, seed=7)[1]
        second = train_validation_split(sequences, seed=7)[1]
        assert [record["idx"] for record in first] == [record["idx"] for record in second]

    def test_different_seeds_select_different_responses(self):
        sequences = make_labelled_sequences(num_per_label=40)
        seeds = {tuple(record["idx"] for record in train_validation_split(sequences, seed=s)[1])
                 for s in range(6)}
        assert len(seeds) > 1

    def test_zero_fraction_moves_everything_to_training(self):
        training, validation = train_validation_split(
            make_labelled_sequences(), seed=1, validation_fraction=0.0
        )
        assert validation == []
        assert len(training) == 20

    def test_empty_input(self):
        assert train_validation_split([], seed=1) == ([], [])
