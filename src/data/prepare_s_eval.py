#!/usr/bin/env python
"""Download and prepare official model-specific S-Eval responses.

The Hugging Face source is a saved DatasetDict. This script exports plain JSONL
files with stable row identifiers so the activation-caching pipeline can use
the data without depending on the Hub at compute time.

"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import snapshot_download

from data.dataset_splits import S_EVAL_DATASET
from project_config import DATA_ROOT, resolve_project_path


SOURCE_REPOSITORY = "Alibaba-AAIG/StreamGuardBench"
MODEL_CONFIGS = {
    "qwen3_8b": {
        "dataset": S_EVAL_DATASET,
        "display_name": "Qwen3-8B",
    },
    "llama_3_1_8b_instruct": {
        "dataset": "s_eval_llama_3_1_8b_instruct",
        "display_name": "Llama-3.1-8B-Instruct",
    },
    "internlm3_8_instruct": {
        "dataset": "s_eval_internlm3_8b_instruct",
        "display_name": "InternLM3-8B-Instruct",
    },
}


def export_jsonl(dataset, output_path, id_key):
    with output_path.open("w") as output_file:
        for index, row in enumerate(dataset):
            record = {
                id_key: index,
                "prompt": row["prompt"],
                "response": row["response"],
                "label": int(row["label"]),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_dataset_readme(dataset_dir, dataset_name, model_name, source_subdirectory, counts, resolved_revision):
    train_counts, test_counts = counts["train"], counts["test"]
    text = f"""# {dataset_name}

Official S-Eval prompts answered by {model_name}, from
`{SOURCE_REPOSITORY}/{source_subdirectory}` at revision `{resolved_revision}`.

| Split | Rows | Safe | Harmful |
|---|---:|---:|---:|
| train | {sum(train_counts.values()):,} | {train_counts[0]:,} | {train_counts[1]:,} |
| test | {sum(test_counts.values()):,} | {test_counts[0]:,} | {test_counts[1]:,} |

Files:

- `train.jsonl`: the complete official training split with stable `idx` values. It is the training pool
  for S-Eval in-domain VQ and classifier experiments.
- `test.jsonl`: the complete official test split with stable `test_index` values. This is the
  cross-dataset detection set.
- `source.json`: source repository, subset, and exact downloaded revision.

The JSONL responses are not truncated. Activation caching applies the configured 2,048-response-token cap.
Cross-dataset experiments use WildGuard training data and S-Eval only for testing. In-domain experiments
train on `train.jsonl`; in both cases, `test.jsonl` remains held out for final metrics.
"""
    (dataset_dir / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="qwen3_8b")
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or commit to download")
    parser.add_argument("--source-root",
                        help="optional pre-downloaded StreamGuardBench root; avoids another Hub download")
    parser.add_argument("--source-revision",
                        help="resolved Hub commit for --source-root provenance")
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]
    dataset_name = model_config["dataset"]
    model_name = model_config["display_name"]
    source_subdirectory = f"s_eval/{args.model}"

    if args.source_root:
        if not args.source_revision:
            parser.error("--source-root requires --source-revision")
        source_root = resolve_project_path(args.source_root)
        source_path = source_root / source_subdirectory
        resolved_revision = args.source_revision
    else:
        snapshot_path = Path(snapshot_download(
            repo_id=SOURCE_REPOSITORY,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=[f"{source_subdirectory}/**"],
        ))
        source_path = snapshot_path / source_subdirectory
        resolved_revision = snapshot_path.name

    source_dataset = load_from_disk(str(source_path))
    counts = {
        split_name: Counter(int(label) for label in source_dataset[split_name]["label"])
        for split_name in ("train", "test")
    }
    dataset_dir = DATA_ROOT / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    export_jsonl(source_dataset["train"], dataset_dir / "train.jsonl", "idx")
    export_jsonl(source_dataset["test"], dataset_dir / "test.jsonl", "test_index")

    source_metadata = {
        "repository": SOURCE_REPOSITORY,
        "subset": source_subdirectory,
        "model": model_name,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "splits": {
            split_name: {
                "rows": sum(split_counts.values()),
                "safe": split_counts[0],
                "harmful": split_counts[1],
            }
            for split_name, split_counts in counts.items()
        },
    }
    (dataset_dir / "source.json").write_text(json.dumps(source_metadata, indent=2) + "\n")
    write_dataset_readme(
        dataset_dir, dataset_name, model_name, source_subdirectory, counts, resolved_revision
    )

    print(f"prepared {dataset_name} from revision {resolved_revision}")
    for split_name in ("train", "test"):
        split_counts = counts[split_name]
        print(f"  {split_name}: {sum(split_counts.values()):,} rows "
              f"({split_counts[0]:,} safe / {split_counts[1]:,} harmful)")
    print(f"  -> {dataset_dir}")


if __name__ == "__main__":
    main()
