#!/usr/bin/env python
"""Cache and analyze the discrete concepts used by TRACE detection and steering."""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SAFETY_LABELS = ("harmful", "neutral", "benign", "safety_ambiguous", "unresolved")


def _json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)


def write_detection_cache(output_dir, scored, code_features, metadata):
    """Save all lightweight detector outputs needed for later token-level analyses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "sequences.npz"
    manifest_path = output_dir / "manifest.json"
    if arrays_path.exists() or manifest_path.exists():
        raise FileExistsError(f"detection audit cache already exists: {output_dir}")

    response_ids, labels, offsets = [], [], [0]
    token_ids, code_ids = [], []
    hazards, cumulative_risks, response_probabilities = [], [], []
    for index, sequence in enumerate(scored["sequences"]):
        codes = np.asarray(sequence["codes"], dtype=np.int32).reshape(-1)
        tokens = np.asarray(sequence["token_ids"], dtype=np.int32).reshape(-1)
        hazard = np.asarray(scored["conditional_hazards"][index], dtype=np.float32).reshape(-1)
        cumulative = np.asarray(scored["token_scores"][index], dtype=np.float32).reshape(-1)
        response = np.asarray(scored["response_token_scores"][index], dtype=np.float32).reshape(-1)

        response_ids.append(int(sequence["idx"]))
        labels.append(int(sequence["label"]))
        token_ids.append(tokens)
        code_ids.append(codes)
        hazards.append(hazard)
        cumulative_risks.append(cumulative)
        response_probabilities.append(response)
        offsets.append(offsets[-1] + len(codes))

    np.savez_compressed(
        arrays_path,
        response_ids=np.asarray(response_ids, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int8),
        offsets=np.asarray(offsets, dtype=np.int64),
        token_ids=np.concatenate(token_ids),
        code_ids=np.concatenate(code_ids),
        conditional_hazards=np.concatenate(hazards),
        cumulative_risks=np.concatenate(cumulative_risks),
        response_probabilities=np.concatenate(response_probabilities),
        code_features=np.asarray(code_features, dtype=np.float32),
    )
    manifest = {
        "format": "trace_detection_audit_cache",
        "arrays_file": arrays_path.name,
        "n_responses": len(response_ids),
        "n_tokens": offsets[-1],
        "fields": {
            "conditional_hazards": "per-token conditional harm probability q_t",
            "cumulative_risks": "cumulative streaming risk R_t",
            "response_probabilities": "harmful-response probability at each token",
            "token_ids": "generator tokenizer IDs for cached response tokens",
            "code_ids": "nearest VQ code for each response token",
        },
        **metadata,
    }
    _json_dump(manifest, manifest_path)
    return manifest_path


def initialize_steering_cache(output_dir, metadata, resume=False):
    """Create or reopen the append-only trace written by an opt-in steering run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    events_path = output_dir / "responses.jsonl.gz"
    manifest = {
        "format": "trace_steering_audit_cache",
        "events_file": events_path.name,
        "fields": {
            "output_code_ids": "codes assigned when the completed output is rescored",
            "online_code_ids": "codes assigned online before each possible edit",
            "edit_positions": "zero-based positions in the online code sequence",
            "source_code_ids": "harmful codes that triggered edits",
            "target_code_ids": "benign codes selected as edit targets",
        },
        **metadata,
    }
    existing_keys = set()
    if resume:
        if not manifest_path.is_file() or not events_path.is_file():
            raise FileNotFoundError(f"cannot resume without a complete steering audit cache: {output_dir}")
        saved_manifest = json.loads(manifest_path.read_text())
        if saved_manifest != manifest:
            raise ValueError("saved steering audit cache does not match the resumed steering run")
        with gzip.open(events_path, "rt") as input_file:
            for line in input_file:
                row = json.loads(line)
                key = (int(row["response_id"]), row["result_key"])
                existing_keys.add(key)
    else:
        if manifest_path.exists() or events_path.exists():
            raise FileExistsError(f"steering audit cache already exists: {output_dir}")
        _json_dump(manifest, manifest_path)
    return events_path, existing_keys


def append_steering_cache(events_path, rows, existing_keys):
    """Append one trace per response and generation configuration."""
    new_rows = []
    for row in rows:
        key = (int(row["response_id"]), row["result_key"])
        if key not in existing_keys:
            new_rows.append(row)
            existing_keys.add(key)
    if new_rows:
        with gzip.open(events_path, "at") as output_file:
            for row in new_rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_catalog(path):
    raw = json.loads(Path(path).read_text())
    catalog = {int(row["code_id"]): row for row in raw["concepts"]}
    return raw, catalog


def _concept_reference(code_id, catalog):
    row = catalog.get(int(code_id))
    if row is None or not row.get("clear_consensus"):
        return {"code_id": int(code_id), "name": None, "safety_label": "unresolved"}
    return {
        "code_id": int(code_id),
        "name": row.get("consensus_name"),
        "description": row.get("consensus_description"),
        "safety_label": row.get("safety_consensus") or "safety_ambiguous",
    }


def _label_summary(references):
    counts = Counter()
    total = named = 0
    for reference in references:
        counts[reference["safety_label"]] += 1
        named += reference.get("name") is not None
        total += 1
    return {
        "n": total,
        "counts": {label: counts[label] for label in SAFETY_LABELS},
        "fractions": {
            label: counts[label] / total if total else None for label in SAFETY_LABELS
        },
        "named_fraction": named / total if total else None,
    }


def _prefix_run_summary(codes, catalog):
    """Summarize recurring concepts after collapsing adjacent copies of one code."""
    codes = np.asarray(codes, dtype=np.int32)
    run_codes = codes[np.r_[True, codes[1:] != codes[:-1]]]
    code_counts = Counter(int(code) for code in run_codes)
    references = {code: _concept_reference(code, catalog) for code in code_counts}
    label_counts = Counter()
    for code, count in code_counts.items():
        label_counts[references[code]["safety_label"]] += count

    dominant = {}
    for safety_label in ("harmful", "benign"):
        candidates = [
            (count, code) for code, count in code_counts.items()
            if references[code]["safety_label"] == safety_label
        ]
        if not candidates:
            dominant[safety_label] = []
            continue
        max_count = max(count for count, _ in candidates)
        dominant[safety_label] = [
            {**references[code], "run_count": count}
            for count, code in sorted(candidates, key=lambda item: item[1])
            if count == max_count
        ]
    return {
        "n_runs": len(run_codes),
        "n_distinct_codes": len(code_counts),
        "label_counts": {label: label_counts[label] for label in SAFETY_LABELS},
        "code_counts": code_counts,
        "dominant": dominant,
    }


def _response_balanced_prefix_summary(prefixes):
    totals = Counter()
    for prefix in prefixes:
        for label in SAFETY_LABELS:
            totals[label] += prefix["label_counts"][label] / prefix["n_runs"]
    return {
        "n": len(prefixes),
        "fractions": {
            label: totals[label] / len(prefixes) if prefixes else None for label in SAFETY_LABELS
        },
    }


def analyze_detection(cache_dir, concept_path, threshold_name, output_dir):
    """Connect detector decisions to concepts at their strongest conditional hazard."""
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("format") != "trace_detection_audit_cache":
        raise ValueError(f"not a detection audit cache: {cache_dir}")
    _, catalog = _load_catalog(concept_path)
    threshold = float(manifest["streaming_thresholds"][threshold_name])

    with np.load(cache_dir / manifest["arrays_file"], allow_pickle=False) as arrays:
        response_ids = arrays["response_ids"]
        labels = arrays["labels"]
        offsets = arrays["offsets"]
        code_ids = arrays["code_ids"]
        hazards = arrays["conditional_hazards"]
        risks = arrays["cumulative_risks"]
        events = []
        prefixes_by_outcome = defaultdict(list)
        recurring_concepts = defaultdict(lambda: {
            "runs": 0, "responses": 0, "recurrent_responses": 0,
            "dominant_response_votes": 0.0,
        })
        for index, response_id in enumerate(response_ids):
            start, end = int(offsets[index]), int(offsets[index + 1])
            response_risks = risks[start:end]
            response_hazards = hazards[start:end]
            crossings = np.flatnonzero(response_risks >= threshold)
            crossing = int(crossings[0]) if len(crossings) else None
            predicted = crossing is not None
            label = int(labels[index])
            outcome = {(1, True): "true_positive", (0, True): "false_positive",
                       (1, False): "false_negative", (0, False): "true_negative"}[(label, predicted)]
            evidence_end = crossing + 1 if crossing is not None else len(response_hazards)
            strongest = int(np.argmax(response_hazards[:evidence_end]))
            strongest_code = int(code_ids[start + strongest])
            prefix = _prefix_run_summary(code_ids[start:start + evidence_end], catalog)
            prefixes_by_outcome[outcome].append(prefix)
            for code, count in prefix["code_counts"].items():
                reference = _concept_reference(code, catalog)
                if reference["safety_label"] not in {"harmful", "benign"}:
                    continue
                key = (outcome, reference["safety_label"], code)
                recurring_concepts[key]["runs"] += count
                recurring_concepts[key]["responses"] += 1
                recurring_concepts[key]["recurrent_responses"] += int(count >= 2)
            for safety_label, dominant in prefix["dominant"].items():
                # Split a response's vote across ties instead of choosing a code arbitrarily.
                vote = 1.0 / len(dominant) if dominant else 0.0
                for concept in dominant:
                    recurring_concepts[(outcome, safety_label, concept["code_id"])][
                        "dominant_response_votes"
                    ] += vote
            event = {
                "response_id": int(response_id),
                "label": label,
                "outcome": outcome,
                "total_tokens": end - start,
                "threshold_name": threshold_name,
                "threshold": threshold,
                "crossing_token": crossing + 1 if crossing is not None else None,
                "crossing_fraction": (crossing + 1) / (end - start) if crossing is not None else None,
                "risk_at_crossing": float(response_risks[crossing]) if crossing is not None else None,
                "final_risk": float(response_risks[-1]),
                "strongest_signal_token": strongest + 1,
                "strongest_conditional_hazard": float(response_hazards[strongest]),
                "strongest_signal_concept": _concept_reference(strongest_code, catalog),
                "prefix_concept_runs": {
                    "n_runs": prefix["n_runs"],
                    "n_distinct_codes": prefix["n_distinct_codes"],
                    "label_counts": prefix["label_counts"],
                    "most_repeated_harmful_concepts": prefix["dominant"]["harmful"],
                    "most_repeated_benign_concepts": prefix["dominant"]["benign"],
                },
            }
            if crossing is not None:
                event["crossing_concept"] = _concept_reference(int(code_ids[start + crossing]), catalog)
            events.append(event)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "detection_events.jsonl").open("w") as output_file:
        for event in events:
            output_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    recurring_rows = []
    for (outcome, safety_label, code), counts in recurring_concepts.items():
        if outcome not in {"true_positive", "false_positive"}:
            continue
        reference = _concept_reference(code, catalog)
        recurring_rows.append({
            "outcome": outcome,
            "safety_label": safety_label,
            "code_id": code,
            "concept_name": reference.get("name"),
            **counts,
        })
    recurring_rows.sort(key=lambda row: (
        row["outcome"], row["safety_label"], -row["recurrent_responses"],
        -row["dominant_response_votes"],
        -row["responses"], -row["runs"], row["code_id"],
    ))
    with (output_dir / "recurring_detection_concepts.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=[
            "outcome", "safety_label", "code_id", "concept_name",
            "recurrent_responses", "dominant_response_votes", "responses", "runs",
        ])
        writer.writeheader()
        writer.writerows(recurring_rows)

    most_repeated = {}
    for outcome in ("true_positive", "false_positive"):
        most_repeated[outcome] = {}
        for safety_label in ("harmful", "benign"):
            candidates = [
                row for row in recurring_rows
                if row["outcome"] == outcome and row["safety_label"] == safety_label
            ]
            most_repeated[outcome][safety_label] = candidates[0] if candidates else None

    grouped = {}
    for outcome in ("true_positive", "false_positive", "false_negative", "true_negative"):
        references = [event["strongest_signal_concept"] for event in events if event["outcome"] == outcome]
        grouped[outcome] = _label_summary(references)
    summary = {
        "analysis": "concepts at the strongest conditional hazard before the detector decision",
        "cache": str(cache_dir),
        "concept_catalog": str(concept_path),
        "threshold_name": threshold_name,
        "threshold": threshold,
        "outcomes": {name: grouped[name]["n"] for name in grouped},
        "all_token_concept_coverage": _label_summary(
            _concept_reference(code_id, catalog) for code_id in code_ids
        ),
        "strongest_signal_concepts": grouped,
        "prefix_concept_runs_response_balanced": {
            outcome: _response_balanced_prefix_summary(prefixes_by_outcome[outcome])
            for outcome in grouped
        },
        "most_repeated_before_detection": most_repeated,
        "events_file": "detection_events.jsonl",
        "recurring_concepts_file": "recurring_detection_concepts.csv",
    }
    _json_dump(summary, output_dir / "detection_summary.json")
    return summary


def _response_balanced_labels(rows, field, catalog):
    totals = Counter()
    used_responses = 0
    named_total = 0.0
    for row in rows:
        references = [_concept_reference(code, catalog) for code in row[field]]
        if not references:
            continue
        used_responses += 1
        counts = Counter(reference["safety_label"] for reference in references)
        named_total += sum(reference.get("name") is not None for reference in references) / len(references)
        for label in SAFETY_LABELS:
            totals[label] += counts[label] / len(references)
    return {
        "n": used_responses,
        "counts": {label: totals[label] for label in SAFETY_LABELS},
        "fractions": {
            label: totals[label] / used_responses if used_responses else None
            for label in SAFETY_LABELS
        },
        "named_fraction": named_total / used_responses if used_responses else None,
    }


def _steering_group_summary(rows, catalog):
    responses_with_edits = sum(bool(row["edit_positions"]) for row in rows)
    return {
        "n_responses": len(rows),
        "responses_with_edits": responses_with_edits,
        "response_edit_coverage": responses_with_edits / len(rows) if rows else None,
        "n_edits": sum(len(row["edit_positions"]) for row in rows),
        "source_concepts_response_balanced": _response_balanced_labels(rows, "source_code_ids", catalog),
        "target_concepts_response_balanced": _response_balanced_labels(rows, "target_code_ids", catalog),
    }


def _normalized_concept_name(name):
    return " ".join(name.split()).casefold()


def _response_level_intervention_records(rows, catalog):
    """Summarize independently labeled harmful sources and benign targets by response."""
    harmful_rows = [row for row in rows if row["label"] == 1]
    edited_rows = [row for row in harmful_rows if row["edit_positions"]]
    source_responses, target_responses, pair_responses = defaultdict(set), defaultdict(set), defaultdict(set)
    source_names, target_names = defaultdict(Counter), defaultdict(Counter)
    pair_names = defaultdict(lambda: {"source": Counter(), "target": Counter()})
    pair_codes = defaultdict(set)

    for row in edited_rows:
        response_id = int(row["response_id"])
        source_codes = set(map(int, row["source_code_ids"]))
        target_codes = set(map(int, row["target_code_ids"]))
        code_pairs = set(zip(map(int, row["source_code_ids"]), map(int, row["target_code_ids"])))

        for code in source_codes:
            reference = _concept_reference(code, catalog)
            if reference["safety_label"] != "harmful" or not reference.get("name"):
                continue
            key = _normalized_concept_name(reference["name"])
            source_responses[key].add(response_id)
            source_names[key][reference["name"]] += 1

        for code in target_codes:
            reference = _concept_reference(code, catalog)
            if reference["safety_label"] != "benign" or not reference.get("name"):
                continue
            key = _normalized_concept_name(reference["name"])
            target_responses[key].add(response_id)
            target_names[key][reference["name"]] += 1

        for source, target in code_pairs:
            source_reference = _concept_reference(source, catalog)
            target_reference = _concept_reference(target, catalog)
            if (source_reference["safety_label"] != "harmful"
                    or target_reference["safety_label"] != "benign"
                    or not source_reference.get("name") or not target_reference.get("name")):
                continue
            key = (
                _normalized_concept_name(source_reference["name"]),
                _normalized_concept_name(target_reference["name"]),
            )
            pair_responses[key].add(response_id)
            pair_names[key]["source"][source_reference["name"]] += 1
            pair_names[key]["target"][target_reference["name"]] += 1
            pair_codes[key].add((source, target))

    denominator = len(edited_rows)

    def ranked_concepts(response_sets, spellings):
        ranked = []
        for key, response_ids in response_sets.items():
            ranked.append({
                "name": spellings[key].most_common(1)[0][0],
                "responses": len(response_ids),
                "response_fraction": len(response_ids) / denominator if denominator else None,
            })
        return sorted(ranked, key=lambda item: (-item["responses"], item["name"].casefold()))

    records = []
    for key, response_ids in pair_responses.items():
        records.append({
            "source_name": pair_names[key]["source"].most_common(1)[0][0],
            "target_name": pair_names[key]["target"].most_common(1)[0][0],
            "responses": len(response_ids),
            "response_fraction": len(response_ids) / denominator if denominator else None,
            "response_ids": sorted(response_ids),
            "code_pairs": [list(pair) for pair in sorted(pair_codes[key])],
        })
    records.sort(key=lambda item: (
        -item["responses"], item["source_name"].casefold(), item["target_name"].casefold()
    ))

    source_coverage = set().union(*source_responses.values()) if source_responses else set()
    target_coverage = set().union(*target_responses.values()) if target_responses else set()
    pair_coverage = set().union(*pair_responses.values()) if pair_responses else set()
    return {
        "n_harmful_responses": len(harmful_rows),
        "n_edited_harmful_responses": denominator,
        "responses_with_named_harmful_source": len(source_coverage),
        "responses_with_named_benign_target": len(target_coverage),
        "responses_with_named_harmful_to_benign_pair": len(pair_coverage),
        "harmful_source_concepts": ranked_concepts(source_responses, source_names),
        "benign_target_concepts": ranked_concepts(target_responses, target_names),
        "harmful_to_benign_records": records,
    }


def analyze_steering(cache_dir, concept_path, result_key, output_dir):
    """Summarize the source and target concepts recorded by a steering run."""
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("format") != "trace_steering_audit_cache":
        raise ValueError(f"not a steering audit cache: {cache_dir}")
    _, catalog = _load_catalog(concept_path)
    with gzip.open(cache_dir / manifest["events_file"], "rt") as input_file:
        rows = [json.loads(line) for line in input_file]
    selected = [row for row in rows if row["result_key"] == result_key]
    if not selected:
        choices = sorted({row["result_key"] for row in rows})
        raise ValueError(f"no steering traces for {result_key!r}; choose from {choices}")

    pair_counts, pair_responses = Counter(), defaultdict(set)
    source_counts, target_counts = Counter(), Counter()
    source_named = target_named = n_edits = 0
    for row in selected:
        sources, targets = row["source_code_ids"], row["target_code_ids"]
        for source, target in zip(sources, targets):
            pair = (int(source), int(target))
            pair_counts[pair] += 1
            pair_responses[pair].add(int(row["response_id"]))
            source_reference = _concept_reference(source, catalog)
            target_reference = _concept_reference(target, catalog)
            source_counts[source_reference["safety_label"]] += 1
            target_counts[target_reference["safety_label"]] += 1
            source_named += source_reference.get("name") is not None
            target_named += target_reference.get("name") is not None
            n_edits += 1

    responses_with_edits = sum(bool(row["edit_positions"]) for row in selected)
    source_balanced = _response_balanced_labels(selected, "source_code_ids", catalog)
    target_balanced = _response_balanced_labels(selected, "target_code_ids", catalog)
    top_pairs = []
    for (source, target), edits in sorted(
        pair_counts.items(), key=lambda item: (-len(pair_responses[item[0]]), -item[1], item[0])
    ):
        top_pairs.append({
            "source": _concept_reference(source, catalog),
            "target": _concept_reference(target, catalog),
            "responses": len(pair_responses[(source, target)]),
            "edits": edits,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    named_pair_edits = sum(
        edits for (source, target), edits in pair_counts.items()
        if _concept_reference(source, catalog).get("name") and _concept_reference(target, catalog).get("name")
    )
    groups = {
        "harmful": _steering_group_summary([row for row in selected if row["label"] == 1], catalog),
        "safe": _steering_group_summary([row for row in selected if row["label"] == 0], catalog),
    }
    intervention_records = _response_level_intervention_records(selected, catalog)
    summary = {
        "analysis": "concepts that trigger steering and their selected targets",
        "cache": str(cache_dir),
        "concept_catalog": str(concept_path),
        "result_key": result_key,
        "n_responses": len(selected),
        "responses_with_edits": responses_with_edits,
        "response_edit_coverage": responses_with_edits / len(selected),
        "n_edits": n_edits,
        "named_source_target_edit_fraction": named_pair_edits / n_edits if n_edits else None,
        "by_response_label": groups,
        "source_concepts_event_weighted": {
            "n": n_edits,
            "counts": {label: source_counts[label] for label in SAFETY_LABELS},
            "fractions": {
                label: source_counts[label] / n_edits if n_edits else None for label in SAFETY_LABELS
            },
            "named_fraction": source_named / n_edits if n_edits else None,
        },
        "target_concepts_event_weighted": {
            "n": n_edits,
            "counts": {label: target_counts[label] for label in SAFETY_LABELS},
            "fractions": {
                label: target_counts[label] / n_edits if n_edits else None for label in SAFETY_LABELS
            },
            "named_fraction": target_named / n_edits if n_edits else None,
        },
        "source_concepts_response_balanced": source_balanced,
        "target_concepts_response_balanced": target_balanced,
        "response_level_intervention_records": intervention_records,
        "top_pairs_file": "steering_concept_pairs.csv",
        "intervention_records_file": "steering_intervention_records.csv",
    }
    _json_dump(summary, output_dir / "steering_summary.json")
    with (output_dir / "steering_concept_pairs.csv").open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "source_code", "source_name", "source_safety", "target_code", "target_name",
            "target_safety", "responses", "edits",
        ])
        for pair in top_pairs:
            writer.writerow([
                pair["source"]["code_id"], pair["source"].get("name"), pair["source"]["safety_label"],
                pair["target"]["code_id"], pair["target"].get("name"), pair["target"]["safety_label"],
                pair["responses"], pair["edits"],
            ])
    with (output_dir / "steering_intervention_records.csv").open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "source_concept", "target_concept", "responses", "response_fraction",
            "response_ids", "code_pairs",
        ])
        for record in intervention_records["harmful_to_benign_records"]:
            writer.writerow([
                record["source_name"], record["target_name"], record["responses"],
                record["response_fraction"], ";".join(map(str, record["response_ids"])),
                ";".join(f"{source}->{target}" for source, target in record["code_pairs"]),
            ])
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    detection = subparsers.add_parser("detection", help="analyze a saved detector token cache")
    detection.add_argument("--cache", required=True)
    detection.add_argument("--concepts", required=True)
    detection.add_argument("--threshold", default="argmax")
    detection.add_argument("--out", required=True)

    steering = subparsers.add_parser("steering", help="analyze a saved steering event cache")
    steering.add_argument("--cache", required=True)
    steering.add_argument("--concepts", required=True)
    steering.add_argument("--result-key", required=True)
    steering.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.operation == "detection":
        summary = analyze_detection(args.cache, args.concepts, args.threshold, args.out)
        print(f"detection audit: {summary['outcomes']} -> {args.out}")
    else:
        summary = analyze_steering(args.cache, args.concepts, args.result_key, args.out)
        print(
            f"steering audit: {summary['n_edits']} edits in "
            f"{summary['responses_with_edits']}/{summary['n_responses']} responses -> {args.out}"
        )


if __name__ == "__main__":
    main()
