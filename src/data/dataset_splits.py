"""Dataset paths and response splits used by TRACE."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from project_config import DATA_ROOT


DEFAULT_DATASET = "wildguard_qwen3_8b"
S_EVAL_DATASET = "s_eval_qwen3_8b"
WILDGUARD_LLAMA_DATASET = "wildguard_llama_3_1_8b_instruct"
WILDGUARD_INTERNLM_DATASET = "wildguard_internlm3_8b_instruct"
S_EVAL_LLAMA_DATASET = "s_eval_llama_3_1_8b_instruct"
S_EVAL_INTERNLM_DATASET = "s_eval_internlm3_8b_instruct"


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    kind: str
    source: Path
    max_response_tokens: int
    id_key: str


_TRAIN_SPLITS = {
    DEFAULT_DATASET: {
        "train": ("phase1_vqvae/train_stratified.json", 2048),
        "classifier_train": ("classifier_training/train_deduplicated.jsonl", 2048),
        "train_small": ("phase1_vqvae/train_balanced_800.json", 1024),
    },
    S_EVAL_DATASET: {"train": ("train.jsonl", 2048)},
    WILDGUARD_LLAMA_DATASET: {
        "train": ("phase1_vqvae/train_stratified.json", 2048),
        "classifier_train": ("classifier_training/train_deduplicated.jsonl", 2048),
    },
    WILDGUARD_INTERNLM_DATASET: {
        "train": ("phase1_vqvae/train_stratified.json", 2048),
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


def _make_split(name, dataset, kind, split_files, id_key):
    relative_source, max_response_tokens = split_files[dataset][name]
    return DatasetSplit(
        name=name,
        kind=kind,
        source=DATA_ROOT / dataset / relative_source,
        max_response_tokens=max_response_tokens,
        id_key=id_key,
    )


def training_split(name="train", dataset=DEFAULT_DATASET):
    return _make_split(name, dataset, "train", _TRAIN_SPLITS, "idx")


def evaluation_split(name="test", dataset=DEFAULT_DATASET):
    return _make_split(name, dataset, "test", _TEST_SPLITS, "test_index")


def dataset_split(name, dataset=DEFAULT_DATASET):
    if name in _TRAIN_SPLITS[dataset]:
        return training_split(name, dataset)
    return evaluation_split(name, dataset)


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
