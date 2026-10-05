#!/usr/bin/env python
"""Summarize Gemini descriptions of VQ codebook entries."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run")
    source.add_argument("--consensus-runs", nargs=3, metavar=("RUN1", "RUN2", "RUN3"))
    parser.add_argument("--output", help="output directory for consensus results")
    return parser.parse_args()


def read_jsonl(path):
    with path.open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def unique_rows(rows):
    return {int(row["code_id"]): row for row in rows}


def valid_complete_judgment(row):
    if row.get("state") != "complete" or not isinstance(row.get("judgment"), dict):
        return False, "judgment is not complete"
    judgment = row["judgment"]
    if set(judgment) != {"status", "name", "description", "safety_label"}:
        return False, "judgment fields do not match the schema"
    if judgment["status"] not in {"clear", "mixed", "no_pattern"}:
        return False, "invalid status"

    if judgment["status"] == "clear":
        if not isinstance(judgment["name"], str) or not judgment["name"].strip():
            return False, "clear concept has no name"
        if not isinstance(judgment["description"], str) or not judgment["description"].strip():
            return False, "clear concept has no description"
        if judgment["safety_label"] not in {
            "harmful", "benign", "neutral"
        }:
            return False, "clear concept has an invalid safety label"
    elif (judgment["name"] is not None or judgment["description"] is not None or
          judgment["safety_label"] is not None):
        return False, "non-clear concept has concept fields"
    return True, None


def valid_agreement_judgment(row, valid_run_ids):
    if row.get("state") != "complete" or not isinstance(row.get("judgment"), dict):
        return False, "agreement judgment is not complete"
    judgment = row["judgment"]
    required = {"agreement", "name", "description", "agreeing_run_ids"}
    if set(judgment) != required:
        return False, "agreement judgment fields do not match the schema"
    if not isinstance(judgment["agreement"], bool):
        return False, "agreement is not a boolean"

    ids = judgment["agreeing_run_ids"]
    if not isinstance(ids, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        return False, "agreeing run IDs are not integers"
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(valid_run_ids)):
        return False, "agreeing run IDs are duplicated or invalid"

    if judgment["agreement"]:
        if len(ids) < 2:
            return False, "semantic agreement has fewer than two agreeing runs"
        if not isinstance(judgment["name"], str) or not judgment["name"].strip():
            return False, "semantic agreement has no name"
        if not isinstance(judgment["description"], str) or not judgment["description"].strip():
            return False, "semantic agreement has no description"
    elif judgment["name"] is not None or judgment["description"] is not None or ids:
        return False, "no agreement has concept fields or agreeing run IDs"
    return True, None


def weighted_fraction(rows, weights):
    denominator = sum(weights(row) for row in rows)
    numerator = sum(weights(row) for row in rows if row["status"] == "clear")
    return numerator / denominator if denominator else None


def status_summary(rows):
    counts = Counter(row["status"] for row in rows)
    selected_total = len(rows)
    valid_rows = [row for row in rows if row["status"] in {"clear", "mixed", "no_pattern"}]
    valid_total = len(valid_rows)
    safety_counts = Counter(
        row["safety_label"] if row["safety_label"] is not None else "unlabeled"
        for row in valid_rows if row["status"] == "clear"
    )
    return {
        "selected_total": selected_total,
        "valid_total": valid_total,
        "counts": {status: counts.get(status, 0)
                   for status in ("clear", "mixed", "no_pattern", "error", "missing")},
        "clear_fraction_of_valid": counts.get("clear", 0) / valid_total if valid_total else None,
        "clear_fraction_of_selected": (
            counts.get("clear", 0) / selected_total if selected_total else None
        ),
        "clear_response_support_weighted_fraction_of_valid": weighted_fraction(
            valid_rows, lambda row: row["response_support"]
        ),
        "clear_token_weighted_fraction_of_valid": weighted_fraction(
            valid_rows, lambda row: row["token_count"]
        ),
        "clear_safety_label_counts": {
            label: safety_counts.get(label, 0)
            for label in ("harmful", "benign", "neutral", "unlabeled")
        },
    }


def build_rows(examples, judgments):
    selected = {code for code, row in examples.items() if row["selected_for_judging"]}
    rows = []
    for code in sorted(selected):
        example_row = examples[code]
        example_count = len(example_row["examples"])

        judgment_row = judgments.get(code)
        if judgment_row is None:
            status, name, description, safety_label = "missing", None, None, None
            error = "no judgment saved"
        elif judgment_row.get("state") == "error":
            status, name, description, safety_label = "error", None, None, None
            error = judgment_row.get("error") or "Gemini request failed"
        else:
            valid, error = valid_complete_judgment(judgment_row)
            if not valid:
                status, name, description, safety_label = "error", None, None, None
            else:
                judgment = judgment_row["judgment"]
                status = judgment["status"]
                name = judgment["name"]
                description = judgment["description"]
                safety_label = judgment["safety_label"]

        row = {
            "code_id": code,
            "region": example_row["region"],
            "harmfulness_score": example_row["harmfulness_score"],
            "response_support": int(example_row["response_support"]),
            "token_count": int(example_row["token_count"]),
            "run_count": int(example_row["run_count"]),
            "example_count": example_count,
            "status": status,
            "name": name,
            "description": description,
            "safety_label": safety_label,
            "error": error,
        }
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = [
        "code_id", "region", "harmfulness_score", "response_support", "token_count", "run_count",
        "example_count", "status", "name", "description", "safety_label", "error",
    ]
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def consensus_rows(runs, agreement_judgments):
    manifests = [json.loads((run / "manifest.json").read_text()) for run in runs]
    analyses = [json.loads((run / "analysis.json").read_text()) for run in runs]
    seeds = [int(manifest["seed"]) for manifest in manifests]

    concepts = [unique_rows(analysis["concepts"]) for analysis in analyses]
    common_codes = set.intersection(*(set(rows) for rows in concepts))
    rows = []
    for code in sorted(common_codes):
        code_rows = [concept[code] for concept in concepts]
        statuses = [row["status"] for row in code_rows]
        clear_votes = statuses.count("clear")
        clear_run_ids = [run_id for run_id, status in enumerate(statuses, start=1)
                         if status == "clear"]
        clear_status_majority = clear_votes >= 2

        agreement_state = "not_applicable"
        agreement_error = None
        agreement = None
        consensus_name = None
        consensus_description = None
        agreeing_run_ids = []
        agreement_row = agreement_judgments.get(code)
        if clear_status_majority:
            if agreement_row is None:
                agreement_state = "missing"
                agreement_error = "no semantic-agreement judgment saved"
            elif agreement_row.get("state") == "error":
                agreement_state = "error"
                agreement_error = agreement_row.get("error") or "Gemini agreement request failed"
            else:
                valid, agreement_error = valid_agreement_judgment(
                    agreement_row, clear_run_ids
                )
                if valid:
                    judgment = agreement_row["judgment"]
                    agreement = judgment["agreement"]
                    agreement_state = "agreement" if agreement else "no_agreement"
                    consensus_name = judgment["name"]
                    consensus_description = judgment["description"]
                    agreeing_run_ids = judgment["agreeing_run_ids"]
                else:
                    agreement_state = "error"

        agreeing_labels = [code_rows[run_id - 1]["safety_label"]
                           for run_id in agreeing_run_ids]
        label_counts = Counter(label for label in agreeing_labels if label is not None)
        safety_label, safety_votes = None, 0
        if label_counts:
            safety_label, safety_votes = label_counts.most_common(1)[0]
            if safety_votes <= len(agreeing_run_ids) / 2:
                safety_label = None

        rows.append({
            "code_id": code,
            "region": code_rows[0]["region"],
            "harmfulness_score": code_rows[0]["harmfulness_score"],
            "response_support": code_rows[0]["response_support"],
            "clear_votes": clear_votes,
            "clear_status_majority": clear_status_majority,
            "unanimous_clear_status": clear_votes == 3,
            "clear_consensus": agreement is True,
            "unanimously_clear": agreement is True and len(agreeing_run_ids) == 3,
            "consensus_name": consensus_name,
            "consensus_description": consensus_description,
            "agreeing_run_ids": agreeing_run_ids,
            "agreement_state": agreement_state,
            "agreement_error": agreement_error,
            "safety_consensus": safety_label,
            "safety_consensus_votes": safety_votes if safety_label is not None else 0,
            "unanimous_safety_consensus": (
                safety_label is not None and len(agreeing_run_ids) == 3 and safety_votes == 3
            ),
            "runs": [
                {
                    "seed": seed,
                    "status": row["status"],
                    "name": row["name"],
                    "description": row["description"],
                    "safety_label": row["safety_label"],
                }
                for seed, row in zip(seeds, code_rows)
            ],
        })
    return manifests, analyses, rows


def write_consensus_csv(path, rows):
    base_fields = [
        "code_id", "region", "harmfulness_score", "response_support", "clear_votes",
        "clear_status_majority", "unanimous_clear_status", "clear_consensus",
        "unanimously_clear", "consensus_name", "consensus_description", "agreeing_run_ids",
        "agreement_state", "agreement_error", "safety_consensus", "safety_consensus_votes",
        "unanimous_safety_consensus",
    ]
    fields = list(base_fields)
    for index in range(1, 4):
        fields.extend([
            f"run_{index}_seed", f"run_{index}_status", f"run_{index}_name",
            f"run_{index}_description", f"run_{index}_safety_label",
        ])

    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row[field] for field in base_fields}
            flat["agreeing_run_ids"] = json.dumps(row["agreeing_run_ids"])
            for index, run in enumerate(row["runs"], start=1):
                for field in ("seed", "status", "name", "description", "safety_label"):
                    flat[f"run_{index}_{field}"] = run[field]
            writer.writerow(flat)


def analyze_consensus(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    judgment_path = output / "agreement_judgments.jsonl"
    agreement_judgments = unique_rows(read_jsonl(judgment_path))
    manifests, analyses, rows = consensus_rows(runs, agreement_judgments)

    clear_status_majority = [row for row in rows if row["clear_status_majority"]]
    clear_consensus = [row for row in rows if row["clear_consensus"]]
    safety_counts = Counter(
        row["safety_consensus"] for row in clear_consensus
        if row["safety_consensus"] is not None
    )
    result = {
        "format_version": 2,
        "runs": [str(run) for run in runs],
        "seeds": [int(manifest["seed"]) for manifest in manifests],
        "checkpoint": manifests[0]["checkpoint"],
        "dataset": manifests[0]["dataset"],
        "judge_model": manifests[0]["judge_model"],
        "examples_per_run": int(manifests[0]["examples_per_code"]),
        "codes_selected_in_all_runs": len(rows),
        "clear_status_majority_count": len(clear_status_majority),
        "clear_status_majority_fraction": (
            len(clear_status_majority) / len(rows) if rows else None
        ),
        "unanimous_clear_status_count": sum(row["unanimous_clear_status"] for row in rows),
        "clear_consensus_count": len(clear_consensus),
        "clear_consensus_fraction": len(clear_consensus) / len(rows) if rows else None,
        "clear_consensus_fraction_of_candidates": (
            len(clear_consensus) / len(clear_status_majority)
            if clear_status_majority else None
        ),
        "unanimously_clear_count": sum(row["unanimously_clear"] for row in rows),
        "unanimously_clear_fraction": (
            sum(row["unanimously_clear"] for row in rows) / len(rows) if rows else None
        ),
        "safety_consensus_count": sum(
            row["safety_consensus"] is not None for row in clear_consensus
        ),
        "safety_consensus_fraction_of_clear": (
            sum(row["safety_consensus"] is not None for row in clear_consensus) /
            len(clear_consensus) if clear_consensus else None
        ),
        "safety_consensus_counts": {
            label: safety_counts.get(label, 0) for label in ("harmful", "benign", "neutral")
        },
        "unanimous_safety_consensus_count": sum(
            row["unanimous_safety_consensus"] for row in rows
        ),
        "agreement_error_count": sum(row["agreement_state"] == "error" for row in rows),
        "agreement_missing_count": sum(row["agreement_state"] == "missing" for row in rows),
        "run_summaries": [analysis["overall"] for analysis in analyses],
        "concepts": rows,
    }
    write_json(output / "consensus.json", result)
    write_consensus_csv(output / "consensus.csv", rows)
    print(
        f"Consensus over {len(rows)} codes: {len(clear_status_majority)} clear-status candidates, "
        f"{len(clear_consensus)} with semantic agreement"
    )
    print(f"Saved {output / 'consensus.json'} and {output / 'consensus.csv'}")


def analyze(run):
    manifest = json.loads((run / "manifest.json").read_text())
    examples = unique_rows(read_jsonl(run / "examples.jsonl"))
    judgment_path = run / "judgments.jsonl"
    judgments = unique_rows(read_jsonl(judgment_path)) if judgment_path.exists() else {}
    rows = build_rows(examples, judgments)

    by_region = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(row)
    insufficient = sum(
        row["selection_status"] == "insufficient_support" for row in examples.values()
    )
    result = {
        "format_version": int(manifest.get("format_version", 1)),
        "run": str(run),
        "checkpoint": manifest["checkpoint"],
        "dataset": manifest["dataset"],
        "judge_model": manifest["judge_model"],
        "num_codes": manifest["num_codes"],
        "eligible_codes": manifest["eligible_code_count"],
        "selected_codes": manifest["selected_code_count"],
        "insufficient_support_codes": insufficient,
        "partial_run": manifest["partial_run"],
        "overall": status_summary(rows),
        "by_region": {region: status_summary(region_rows)
                      for region, region_rows in sorted(by_region.items())},
        "concepts": rows,
    }
    write_json(run / "analysis.json", result)
    write_csv(run / "concepts.csv", rows)

    counts = result["overall"]["counts"]
    print(f"Analyzed {len(rows)} selected codes from {manifest['dataset']}")
    print(
        f"clear {counts['clear']} | mixed {counts['mixed']} | no pattern {counts['no_pattern']} | "
        f"error {counts['error']} | missing {counts['missing']}"
    )
    print(f"Saved {run / 'analysis.json'} and {run / 'concepts.csv'}")


def main():
    args = parse_args()
    if args.run:
        if args.output:
            raise ValueError("--output is only used with --consensus-runs")
        run = Path(args.run)
        if not run.is_dir():
            raise FileNotFoundError(f"analysis run does not exist: {run}")
        analyze(run)
        return

    if not args.output:
        raise ValueError("--output is required with --consensus-runs")
    runs = [Path(path) for path in args.consensus_runs]
    missing = [run for run in runs if not run.is_dir()]
    if missing:
        raise FileNotFoundError(f"analysis run does not exist: {missing[0]}")
    analyze_consensus(runs, Path(args.output))


if __name__ == "__main__":
    main()
