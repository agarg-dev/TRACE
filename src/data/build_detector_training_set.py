#!/usr/bin/env python
"""Build the deduplicated WildGuard detector-training pool."""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

from data.dataset_splits import DEFAULT_DATASET
from project_config import DATA_ROOT


def normalize_text(text):
    return " ".join((text or "").split()).casefold()


def response_class(row):
    if row.label == 1:
        return "harmful"
    return "refusal" if row.prompt_harm_label == "harmful" else "benign"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="default: data/<dataset>/classifier_training/train_deduplicated.jsonl")
    args = parser.parse_args()

    # Normalize and deduplicate the source responses.
    data_dir = DATA_ROOT / args.dataset
    dataset = load_from_disk(str(data_dir / "raw_enriched"))["train"]
    columns = ["prompt", "response", "label", "subcategory", "prompt_harm_label", "adversarial"]
    frame = pd.DataFrame({column: dataset[column] for column in columns})
    frame["idx"] = range(len(frame))
    frame["normalized_prompt"] = frame.prompt.map(normalize_text)
    frame["normalized_response"] = frame.response.map(normalize_text)

    original_rows = len(frame)
    frame = frame[(frame.normalized_prompt != "") & (frame.normalized_response != "")].copy()
    nonempty_rows = len(frame)
    frame = frame.drop_duplicates(["normalized_prompt", "normalized_response"], keep="first")
    unique_pair_rows = len(frame)
    frame["cls"] = [response_class(row) for row in frame.itertuples(index=False)]

    # Preserve the scarce harmful outcome when a source prompt has several differently labelled responses.
    class_priority = {"harmful": 0, "refusal": 1, "benign": 2}
    frame["class_priority"] = frame.cls.map(class_priority)
    frame = frame.sort_values(["class_priority", "idx"])
    frame = frame.drop_duplicates("normalized_prompt", keep="first").reset_index(drop=True)

    # Match refusal and benign examples to the harmful class size.
    by_class = {name: frame[frame.cls == name] for name in class_priority}
    examples_per_class = len(by_class["harmful"])
    if any(len(group) < examples_per_class for group in by_class.values()):
        counts = {name: len(group) for name, group in by_class.items()}
        raise ValueError(f"not enough safe examples to match all harmful examples: {counts}")

    selected = [by_class["harmful"]]
    selected += [by_class[name].sample(examples_per_class, random_state=args.seed)
                 for name in ("refusal", "benign")]
    pool = pd.concat(selected, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save the selected pool and a record of the selection counts.
    output_path = Path(args.out) if args.out else data_dir / "classifier_training/train_deduplicated.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep = ["idx", "prompt", "response", "label", "subcategory", "prompt_harm_label", "adversarial", "cls"]
    with open(output_path, "w") as output_file:
        for record in pool[keep].to_dict("records"):
            record["idx"] = int(record["idx"])
            record["label"] = int(record["label"])
            record["adversarial"] = bool(record["adversarial"])
            output_file.write(json.dumps(record) + "\n")

    manifest = {
        "source": str(data_dir / "raw_enriched"),
        "source_split": "train",
        "seed": args.seed,
        "selection": [
            "normalize prompt and response with whitespace folding and case folding",
            "remove duplicate normalized prompt-response pairs",
            "keep one response per normalized prompt, prioritizing harmful then refusal then benign",
            "keep all harmful responses and sample the same number of refusals and benign responses",
        ],
        "original_rows": original_rows,
        "nonempty_rows": nonempty_rows,
        "empty_rows_dropped": original_rows - nonempty_rows,
        "unique_prompt_response_rows": unique_pair_rows,
        "unique_prompt_rows": len(frame),
        "examples_per_class": examples_per_class,
        "output_rows": len(pool),
        "class_counts": {name: int((pool.cls == name).sum()) for name in class_priority},
        "binary_label_counts": {
            "safe": int((pool.label == 0).sum()),
            "harmful": int((pool.label == 1).sum()),
        },
    }
    manifest_path = output_path.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {output_path}")
    print(f"  source rows             : {original_rows:,}")
    print(f"  non-empty rows          : {nonempty_rows:,}")
    print(f"  unique prompt-response  : {unique_pair_rows:,}")
    print(f"  unique prompts          : {len(frame):,}")
    print(f"  selected                : {len(pool):,} ({examples_per_class:,} per class)")
    print(f"  manifest                : {manifest_path}")


if __name__ == "__main__":
    main()
