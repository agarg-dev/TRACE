"""Download the model-specific WildGuard splits from StreamGuardBench."""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import snapshot_download

from project_config import DATA_ROOT


SOURCE_REPOSITORY = "Alibaba-AAIG/StreamGuardBench"
MODEL_DATASETS = {
    "qwen3_8b": "wildguard_qwen3_8b",
    "llama_3_1_8b_instruct": "wildguard_llama_3_1_8b_instruct",
    "internlm3_8_instruct": "wildguard_internlm3_8b_instruct",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_DATASETS)
    parser.add_argument("--revision", default="main", help="StreamGuardBench branch, tag, or commit")
    parser.add_argument("--source-root", help="pre-downloaded StreamGuardBench dataset root")
    parser.add_argument("--source-revision", help="source commit when using --source-root")
    args = parser.parse_args()

    subdirectory = f"wildguard/{args.model}"
    if args.source_root:
        if not args.source_revision:
            parser.error("--source-root requires --source-revision")
        source_path = Path(args.source_root).expanduser() / subdirectory
        resolved_revision = args.source_revision
    else:
        snapshot_path = Path(snapshot_download(
            repo_id=SOURCE_REPOSITORY,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=[f"{subdirectory}/**"],
        ))
        source_path = snapshot_path / subdirectory
        resolved_revision = snapshot_path.name

    source = load_from_disk(str(source_path))
    counts = {}
    for name in ("train", "test"):
        labels = Counter(int(label) for label in source[name]["label"])
        counts[name] = {"rows": len(source[name]), "safe": labels[0], "harmful": labels[1]}

    dataset_dir = DATA_ROOT / MODEL_DATASETS[args.model]
    raw_path = dataset_dir / "raw"
    if raw_path.exists():
        raise FileExistsError(f"raw dataset already exists: {raw_path}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    source.save_to_disk(str(raw_path))
    metadata = {
        "repository": SOURCE_REPOSITORY,
        "subset": subdirectory,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "splits": counts,
    }
    (dataset_dir / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {MODEL_DATASETS[args.model]} from revision {resolved_revision}")
    for name, split_counts in counts.items():
        print(f"  {name}: {split_counts['rows']:,} responses")
    print(f"  -> {raw_path}")


if __name__ == "__main__":
    main()
