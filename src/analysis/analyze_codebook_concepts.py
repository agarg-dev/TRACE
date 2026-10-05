#!/usr/bin/env python
"""Summarize Gemini descriptions of VQ codebook entries."""

import argparse
import csv
import json
from collections import Counter, defaultdict

from project_config import resolve_project_path


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


def rows_by_code(rows):
    return {int(row["code_id"]): row for row in rows}


def weighted_fraction(rows, weight):
    denominator = sum(weight(row) for row in rows)
    numerator = sum(weight(row) for row in rows if row["status"] == "clear")
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
    rows = []
    selected = [code for code, row in examples.items() if row["selected_for_judging"]]
    for code in sorted(selected):
        example_row = examples[code]
        example_count = len(example_row["examples"])
        judgment_row = judgments.get(code)
        if judgment_row is None:
            status, name, description, safety_label = "missing", None, None, None
            error = "no judgment saved"
        elif judgment_row["state"] == "error":
            status, name, description, safety_label = "error", None, None, None
            error = judgment_row.get("error") or "Gemini request failed"
        else:
            judgment = judgment_row["judgment"]
            status = judgment["status"]
            name = judgment["name"]
            description = judgment["description"]
            safety_label = judgment["safety_label"]
            error = None

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
    if len({manifest["dataset"] for manifest in manifests}) != 1:
        raise ValueError("consensus runs use different datasets")
    if len({manifest["judge_model"] for manifest in manifests}) != 1:
        raise ValueError("consensus runs use different judge models")
    if len({int(manifest["examples_per_code"]) for manifest in manifests}) != 1:
        raise ValueError("consensus runs use different example counts")
    seeds = [int(manifest["seed"]) for manifest in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError("consensus runs must use distinct sampling seeds")

    concepts = [rows_by_code(analysis["concepts"]) for analysis in analyses]
    common_codes = set.intersection(*(set(rows) for rows in concepts))
    rows = []
    for code in sorted(common_codes):
        code_rows = [concept[code] for concept in concepts]
        if len({row["region"] for row in code_rows}) != 1:
            raise ValueError(f"region differs across runs for code {code}")
        if len({row["harmfulness_score"] for row in code_rows}) != 1:
            raise ValueError(f"harmfulness score differs across runs for code {code}")

        statuses = [row["status"] for row in code_rows]
        clear_votes = statuses.count("clear")
        has_clear_majority = clear_votes >= 2

        agreement_state = "not_applicable"
        agreement_error = None
        agreement = None
        consensus_name = None
        consensus_description = None
        agreeing_run_ids = []
        agreement_row = agreement_judgments.get(code)
        if has_clear_majority:
            if agreement_row is None:
                agreement_state = "missing"
                agreement_error = "no semantic-agreement judgment saved"
            elif agreement_row.get("state") == "error":
                agreement_state = "error"
                agreement_error = agreement_row.get("error") or "Gemini agreement request failed"
            else:
                judgment = agreement_row["judgment"]
                agreement = judgment["agreement"]
                agreement_state = "agreement" if agreement else "no_agreement"
                consensus_name = judgment["name"]
                consensus_description = judgment["description"]
                agreeing_run_ids = judgment["agreeing_run_ids"]
        elif agreement_row is not None:
            raise ValueError(f"unexpected semantic-agreement judgment for code {code}")

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
            "clear_status_majority": has_clear_majority,
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
    agreement_judgments = rows_by_code(read_jsonl(judgment_path))
    manifests, analyses, rows = consensus_rows(runs, agreement_judgments)

    majority_rows = [row for row in rows if row["clear_status_majority"]]
    consensus_concepts = [row for row in rows if row["clear_consensus"]]
    safety_counts = Counter(
        row["safety_consensus"] for row in consensus_concepts
        if row["safety_consensus"] is not None
    )
    unanimous_clear_count = sum(row["unanimously_clear"] for row in rows)
    safety_consensus_count = sum(
        row["safety_consensus"] is not None for row in consensus_concepts
    )
    result = {
        "runs": [str(run) for run in runs],
        "seeds": [int(manifest["seed"]) for manifest in manifests],
        "checkpoint": manifests[0]["checkpoint"],
        "dataset": manifests[0]["dataset"],
        "judge_model": manifests[0]["judge_model"],
        "examples_per_run": int(manifests[0]["examples_per_code"]),
        "codes_selected_in_all_runs": len(rows),
        "clear_status_majority_count": len(majority_rows),
        "clear_status_majority_fraction": (
            len(majority_rows) / len(rows) if rows else None
        ),
        "unanimous_clear_status_count": sum(row["unanimous_clear_status"] for row in rows),
        "clear_consensus_count": len(consensus_concepts),
        "clear_consensus_fraction": len(consensus_concepts) / len(rows) if rows else None,
        "clear_consensus_fraction_of_candidates": (
            len(consensus_concepts) / len(majority_rows) if majority_rows else None
        ),
        "unanimously_clear_count": unanimous_clear_count,
        "unanimously_clear_fraction": (
            unanimous_clear_count / len(rows) if rows else None
        ),
        "safety_consensus_count": safety_consensus_count,
        "safety_consensus_fraction_of_clear": (
            safety_consensus_count / len(consensus_concepts) if consensus_concepts else None
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
        f"Consensus over {len(rows)} codes: {len(majority_rows)} clear-status candidates, "
        f"{len(consensus_concepts)} with semantic agreement"
    )
    print(f"Saved {output / 'consensus.json'} and {output / 'consensus.csv'}")


def analyze(run):
    manifest = json.loads((run / "manifest.json").read_text())
    examples = rows_by_code(read_jsonl(run / "examples.jsonl"))
    judgment_path = run / "judgments.jsonl"
    judgments = rows_by_code(read_jsonl(judgment_path)) if judgment_path.exists() else {}
    concept_rows = build_rows(examples, judgments)

    by_region = defaultdict(list)
    for row in concept_rows:
        by_region[row["region"]].append(row)
    insufficient = sum(
        row["selection_status"] == "insufficient_support" for row in examples.values()
    )
    result = {
        "run": str(run),
        "checkpoint": manifest["checkpoint"],
        "dataset": manifest["dataset"],
        "judge_model": manifest["judge_model"],
        "num_codes": manifest["num_codes"],
        "eligible_codes": manifest["eligible_code_count"],
        "selected_codes": manifest["selected_code_count"],
        "insufficient_support_codes": insufficient,
        "overall": status_summary(concept_rows),
        "by_region": {region: status_summary(region_rows)
                      for region, region_rows in sorted(by_region.items())},
        "concepts": concept_rows,
    }
    write_json(run / "analysis.json", result)
    write_csv(run / "concepts.csv", concept_rows)

    counts = result["overall"]["counts"]
    print(f"Analyzed {len(concept_rows)} selected codes from {manifest['dataset']}")
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
        run = resolve_project_path(args.run).resolve()
        analyze(run)
        return

    if not args.output:
        raise ValueError("--output is required with --consensus-runs")
    runs = [resolve_project_path(path).resolve() for path in args.consensus_runs]
    analyze_consensus(runs, resolve_project_path(args.output).resolve())


if __name__ == "__main__":
    main()
