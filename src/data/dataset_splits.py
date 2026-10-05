"""Dataset paths and stratified response splits used by TRACE."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATASET = "wildguard_qwen3_8b"
S_EVAL_DATASET = "s_eval_qwen3_8b"
WILDGUARD_LLAMA_DATASET = "wildguard_llama_3_1_8b_instruct"
WILDGUARD_INTERNLM_DATASET = "wildguard_internlm3_8b_instruct"
S_EVAL_LLAMA_DATASET = "s_eval_llama_3_1_8b_instruct"
S_EVAL_INTERNLM_DATASET = "s_eval_internlm3_8b_instruct"

TRAIN_SETS = frozenset({"train", "classifier_train"})
TEST_SETS = frozenset({"test"})

# Full response pools split 90/10 for detector training and held-out validation.
DETECTOR_TRAIN_SET_BY_DATASET = {
    DEFAULT_DATASET: "classifier_train",
    WILDGUARD_LLAMA_DATASET: "classifier_train",
    WILDGUARD_INTERNLM_DATASET: "classifier_train",
    S_EVAL_DATASET: "train",
    S_EVAL_LLAMA_DATASET: "train",
    S_EVAL_INTERNLM_DATASET: "train",
}


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    source: Path
    max_response_tokens: int
    id_key: str


_TRAIN_SPLITS = {
    DEFAULT_DATASET: {
        "classifier_train": ("classifier_training/train_deduplicated.jsonl", 2048),
    },
    S_EVAL_DATASET: {"train": ("train.jsonl", 2048)},
    WILDGUARD_LLAMA_DATASET: {
        "classifier_train": ("classifier_training/train_deduplicated.jsonl", 2048),
    },
    WILDGUARD_INTERNLM_DATASET: {
        "classifier_train": ("classifier_training/train_deduplicated.jsonl", 2048),
    },
    S_EVAL_LLAMA_DATASET: {"train": ("train.jsonl", 2048)},
    S_EVAL_INTERNLM_DATASET: {"train": ("train.jsonl", 2048)},
}

_TEST_SPLITS = {
    DEFAULT_DATASET: {
        "test": ("phase2_intervention/test_full.jsonl", 2048),
    },
    S_EVAL_DATASET: {"test": ("test.jsonl", 2048)},
    WILDGUARD_LLAMA_DATASET: {"test": ("phase2_intervention/test_full.jsonl", 2048)},
    WILDGUARD_INTERNLM_DATASET: {"test": ("phase2_intervention/test_full.jsonl", 2048)},
    S_EVAL_LLAMA_DATASET: {"test": ("test.jsonl", 2048)},
    S_EVAL_INTERNLM_DATASET: {"test": ("test.jsonl", 2048)},
}


def _data_root():
    from project_config import DATA_ROOT

    return DATA_ROOT


def _resolve_split(name, dataset, split_files, id_key):
    files = split_files[dataset]
    relative_source, max_response_tokens = files[name]
    return DatasetSplit(
        name=name,
        source=_data_root() / dataset / relative_source,
        max_response_tokens=max_response_tokens,
        id_key=id_key,
    )


def training_split(name="train", dataset=DEFAULT_DATASET):
    return _resolve_split(name, dataset, _TRAIN_SPLITS, "idx")


def evaluation_split(name="test", dataset=DEFAULT_DATASET):
    return _resolve_split(name, dataset, _TEST_SPLITS, "test_index")


def train_validation_split(sequences, seed, validation_fraction=0.1):
    """Seeded binary-stratified split used by the response classifier."""
    random_state = np.random.RandomState(seed)
    sequences_by_label = defaultdict(list)
    for sequence in sequences:
        sequences_by_label[sequence["label"]].append(sequence)

    training_sequences, validation_sequences = [], []
    for label in (0, 1):
        group = sequences_by_label[label]
        group = [group[index] for index in random_state.permutation(len(group))]
        validation_size = round(validation_fraction * len(group))
        validation_sequences.extend(group[:validation_size])
        training_sequences.extend(group[validation_size:])
    return training_sequences, validation_sequences
