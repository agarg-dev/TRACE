#!/usr/bin/env python
"""Add WildGuard's prompt-level labels to a StreamGuardBench response dataset.

StreamGuardBench ships only prompt/response/label. Its WildGuard prompts are exactly WildGuardMix's,
so an exact prompt-string join recovers the prompt-level fields. Only prompt-level fields transfer:
WildGuardMix response annotations describe different generations and are deliberately not copied.

WildGuardMix is gated. Accept its terms and authenticate with Hugging Face
before running this script.
"""
import argparse
import json

import pandas as pd
from datasets import DatasetDict, load_from_disk
from huggingface_hub import get_token, hf_hub_download

from project_config import DATA_DIR, resolve_project_path

DEFAULT_RAW = DATA_DIR / "raw"
DEFAULT_DATASET_DIR = DATA_DIR
DEFAULT_TAXONOMY_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"

# prompt_harm_label may be null when upstream annotators did not agree.
FIELDS = {"subcategory": str, "prompt_harm_label": str, "adversarial": bool}
REQUIRED = {"subcategory", "adversarial"}


def prompt_maps(token, revision):
    """prompt -> value for each transferable field, from WildGuardMix train+test (first non-null wins)."""
    files = ("train/wildguard_train.parquet", "test/wildguard_test.parquet")
    frames = [pd.read_parquet(hf_hub_download("allenai/wildguardmix", filename,
                                              repo_type="dataset", token=token, revision=revision))
              for filename in files]
    wildguard = pd.concat(frames, ignore_index=True)
    values_by_field = {field: {} for field in FIELDS}
    for row in wildguard.itertuples(index=False):
        for field, cast in FIELDS.items():
            value = getattr(row, field)
            if pd.notna(value) and row.prompt not in values_by_field[field]:
                values_by_field[field][row.prompt] = cast(value)
    return values_by_field


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
    parser.add_argument("--source", default=str(DEFAULT_RAW),
                        help="official StreamGuardBench DatasetDict to enrich")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR),
                        help="destination containing raw/ and raw_enriched/")
    parser.add_argument("--taxonomy-revision", default=DEFAULT_TAXONOMY_REVISION,
                        help="WildGuardMix commit used for the prompt taxonomy")
    args = parser.parse_args()

    source = resolve_project_path(args.source)
    dataset_dir = resolve_project_path(args.dataset_dir)
    raw_path = dataset_dir / "raw"
    output_path = dataset_dir / "raw_enriched"

    raw = load_from_disk(str(source))
    if source.resolve() != raw_path.resolve():
        if raw_path.exists():
            raise FileExistsError(f"raw dataset already exists: {raw_path}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save_to_disk(str(raw_path))

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing enriched dataset: {output_path}")

    token = get_token()
    values_by_field = prompt_maps(token, args.taxonomy_revision)

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

    enriched.save_to_disk(str(output_path))
    metadata_path = dataset_dir / "source.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["taxonomy_repository"] = "allenai/wildguardmix"
        metadata["taxonomy_revision"] = args.taxonomy_revision
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
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
