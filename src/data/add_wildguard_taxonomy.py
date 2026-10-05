#!/usr/bin/env python
"""Join WildGuard prompt metadata to model-specific response datasets."""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_from_disk
from huggingface_hub import hf_hub_download

from data.dataset_splits import DEFAULT_DATASET
from project_config import DATA_ROOT

# prompt_harm_label may be null when upstream annotators did not agree.
FIELDS = {"subcategory": str, "prompt_harm_label": str, "adversarial": bool}
REQUIRED = {"subcategory", "adversarial"}


def prompt_maps(token):
    """prompt -> value for each transferable field, from WildGuardMix train+test (first non-null wins)."""
    files = ("train/wildguard_train.parquet", "test/wildguard_test.parquet")
    frames = [pd.read_parquet(hf_hub_download("allenai/wildguardmix", filename,
                                              repo_type="dataset", token=token)) for filename in files]
    wildguard = pd.concat(frames, ignore_index=True)
    values_by_field = {field: {} for field in FIELDS}
    for row in wildguard.itertuples(index=False):
        for field, cast in FIELDS.items():
            value = getattr(row, field)
            if pd.notna(value) and row.prompt not in values_by_field[field]:
                values_by_field[field][row.prompt] = cast(value)
    return values_by_field


def validate_source(raw, source):
    if set(raw) != {"train", "test"}:
        raise ValueError(f"{source} must contain train/test splits, found {sorted(raw)}")
    required = {"prompt", "response", "label"}
    for split_name, split in raw.items():
        missing = required - set(split.column_names)
        if missing:
            raise ValueError(f"{source}/{split_name} is missing columns {sorted(missing)}")
        labels = {int(label) for label in split["label"]}
        if not labels <= {0, 1} or labels != {0, 1}:
            raise ValueError(f"{source}/{split_name} must contain binary labels 0 and 1, found {sorted(labels)}")
        empty_prompts = sum(not str(prompt).strip() for prompt in split["prompt"])
        if empty_prompts:
            raise ValueError(f"{source}/{split_name} contains {empty_prompts} empty prompts")


def export_test_split(enriched, output_path):
    """Export the official test split with stable row IDs for activation caching."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        for test_index, row in enumerate(enriched["test"]):
            record = {
                "test_index": test_index,
                "label": int(row["label"]),
                "prompt": row["prompt"],
                "response": row["response"],
                "subcategory": row["subcategory"],
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset_dir = DATA_ROOT / DEFAULT_DATASET
    parser.add_argument("--source", default=str(default_dataset_dir / "raw"),
                        help="official StreamGuardBench DatasetDict to enrich")
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir),
                        help="destination containing raw/ and raw_enriched/")
    args = parser.parse_args()

    source = Path(args.source)
    dataset_dir = Path(args.dataset_dir)
    raw_path = dataset_dir / "raw"
    output_path = dataset_dir / "raw_enriched"

    # Keep a local copy of the source dataset before adding taxonomy fields.
    raw = load_from_disk(str(source))
    validate_source(raw, source)
    if source.resolve() != raw_path.resolve():
        if raw_path.exists():
            persisted_raw = load_from_disk(str(raw_path))
            validate_source(persisted_raw, raw_path)
            for split_name in raw:
                if len(raw[split_name]) != len(persisted_raw[split_name]):
                    raise ValueError(f"existing raw dataset does not match source: {raw_path}/{split_name}")
                for field in ("prompt", "response", "label"):
                    if raw[split_name][field] != persisted_raw[split_name][field]:
                        raise ValueError(
                            f"existing raw dataset does not match source field {field}: "
                            f"{raw_path}/{split_name}"
                        )
            raw = persisted_raw
        else:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw.save_to_disk(str(raw_path))

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing enriched dataset: {output_path}")

    token = (Path.home() / ".cache/huggingface/token").read_text().strip()
    values_by_field = prompt_maps(token)

    # Join the WildGuardMix fields by exact prompt text.
    enriched = DatasetDict({split_name: split for split_name, split in raw.items()})
    for split_name in enriched:
        prompts = enriched[split_name]["prompt"]
        for field in FIELDS:
            column = [values_by_field[field].get(prompt) for prompt in prompts]
            missing = sum(value is None for value in column)
            if missing and field in REQUIRED:
                raise ValueError(
                    f"{split_name}/{field}: {missing} prompts unmatched -- join is not lossless, aborting"
                )
            if missing:
                print(
                    f"  note: {split_name}/{field} has {missing} upstream nulls "
                    "(no WildGuard annotator agreement)"
                )
            enriched[split_name] = enriched[split_name].add_column(field, column)

    # Save the enriched dataset and the JSONL test split consumed downstream.
    enriched.save_to_disk(str(output_path))
    test_path = dataset_dir / "phase2_intervention/test_full.jsonl"
    export_test_split(enriched, test_path)
    print(f"raw source: {raw_path}")
    print(f"wrote {output_path}")
    print(f"wrote {test_path}")
    for split_name, split in enriched.items():
        empty_responses = sum(not str(response).strip() for response in split["response"])
        print(
            f"  {split_name}: {len(split)} rows, columns {split.column_names}, "
            f"empty responses {empty_responses}"
        )


if __name__ == "__main__":
    main()
