#!/usr/bin/env python
"""Cache and analyze the discrete concepts used by TRACE detection and steering."""

import argparse
import csv
import gzip
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trace_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trace_plot_cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SAFETY_LABELS = ("harmful", "neutral", "benign", "safety_ambiguous", "unresolved")
SAFETY_COLORS = {
    "harmful": "#cc6b5e",
    "neutral": "#9aa8af",
    "benign": "#77a982",
    "safety_ambiguous": "#c4a56a",
    "unresolved": "#d9dfe2",
}
SAFETY_NAMES = {
    "harmful": "Harmful",
    "neutral": "Neutral",
    "benign": "Benign",
    "safety_ambiguous": "Safety ambiguous",
    "unresolved": "No agreed concept",
}


def _json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)


def _numpy(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def write_detection_cache(output_dir, scored, code_features, metadata):
    """Save all lightweight detector outputs needed for later token-level analyses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "sequences.npz"
    manifest_path = output_dir / "manifest.json"
    if arrays_path.exists() or manifest_path.exists():
        raise FileExistsError(f"detection audit cache already exists: {output_dir}")

    required = ("conditional_hazards", "response_token_scores", "token_scores")
    missing = [name for name in required if name not in scored]
    if missing:
        raise ValueError(f"detector scores are missing token details: {missing}")

    response_ids, labels, offsets = [], [], [0]
    token_ids, code_ids = [], []
    hazards, cumulative_risks, response_probabilities = [], [], []
    for index, sequence in enumerate(scored["sequences"]):
        codes = _numpy(sequence.get("codes"), np.int32).reshape(-1)
        tokens = _numpy(sequence.get("token_ids"), np.int32).reshape(-1)
        hazard = _numpy(scored["conditional_hazards"][index], np.float32).reshape(-1)
        cumulative = _numpy(scored["token_scores"][index], np.float32).reshape(-1)
        response = _numpy(scored["response_token_scores"][index], np.float32).reshape(-1)
        lengths = {len(codes), len(tokens), len(hazard), len(cumulative), len(response)}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError(f"unaligned token details for response {sequence.get('idx', index)}")
        if not np.isclose(cumulative[-1], scored["streaming_scores"][index], atol=1e-6):
            raise ValueError("cached cumulative risk does not match the evaluated streaming score")
        if not np.isclose(response[-1], scored["response_scores"][index], atol=1e-6):
            raise ValueError("cached response probability does not match the evaluated response score")

        response_ids.append(int(sequence.get("idx", index)))
        labels.append(int(sequence["label"]))
        token_ids.append(tokens)
        code_ids.append(codes)
        hazards.append(hazard)
        cumulative_risks.append(cumulative)
        response_probabilities.append(response)
        offsets.append(offsets[-1] + len(codes))

    if len(response_ids) != len(set(response_ids)):
        raise ValueError("response IDs must be unique in a detection audit cache")

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
        code_features=_numpy(code_features, np.float32),
    )
    manifest = {
        "format": "trace_detection_audit_cache",
        "format_version": 1,
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
        "format_version": 1,
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
                if key in existing_keys:
                    raise ValueError(f"duplicate steering audit record: {key}")
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
    safety_label = row.get("safety_consensus")
    if safety_label is None:
        safety_label = "safety_ambiguous"
    elif safety_label not in {"harmful", "neutral", "benign"}:
        raise ValueError(f"unknown safety label for concept C{code_id}: {safety_label!r}")
    return {
        "code_id": int(code_id),
        "name": row.get("consensus_name"),
        "description": row.get("consensus_description"),
        "safety_label": safety_label,
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
    label_counts = Counter()
    for code, count in code_counts.items():
        label_counts[_concept_reference(code, catalog)["safety_label"]] += count

    dominant = {}
    for safety_label in ("harmful", "benign"):
        candidates = []
        for code, count in code_counts.items():
            concept = _concept_reference(code, catalog)
            if concept["safety_label"] == safety_label:
                candidates.append((count, code))
        if not candidates:
            dominant[safety_label] = []
            continue
        max_count = max(count for count, _ in candidates)
        dominant_concepts = []
        for count, code in sorted(candidates, key=lambda item: item[1]):
            if count == max_count:
                dominant_concepts.append({
                    **_concept_reference(code, catalog),
                    "run_count": count,
                })
        dominant[safety_label] = dominant_concepts
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


def _plot_label_bars(rows, output_path, title):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(4.35, 1.75))
    positions = np.arange(len(rows))
    left = np.zeros(len(rows))
    for label in SAFETY_LABELS:
        values = np.asarray([100 * (row["fractions"].get(label) or 0.0) for row in rows])
        axis.barh(
            positions, values, left=left, height=0.55, label=label.capitalize(),
            color=SAFETY_COLORS[label], edgecolor="white", linewidth=0.6,
        )
        left += values
    axis.set_yticks(positions, [row["name"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Responses (%)")
    axis.set_title(title, pad=9)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#e2e7e9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=SAFETY_COLORS[label]) for label in SAFETY_LABELS],
        labels=[SAFETY_NAMES[label] for label in SAFETY_LABELS],
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
        frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.2,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_recurring_detection_concepts(rows, outcome_counts, output_path, top_n=3):
    """Compare named concepts that recur before correct detections and false alarms."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    outcomes = ("true_positive", "false_positive")
    outcome_labels = {
        "true_positive": "Correct harmful detections",
        "false_positive": "False alarms",
    }
    outcome_colors = {"true_positive": "#318579", "false_positive": "#c76a5b"}
    lookup = {
        (row["outcome"], row["safety_label"], row["code_id"]): row for row in rows
    }

    selected = {}
    maximum = 0.0
    for safety_label in ("harmful", "benign"):
        candidates = [row for row in rows if row["safety_label"] == safety_label]
        code_ids = sorted({row["code_id"] for row in candidates}, key=lambda code_id: (
            -lookup.get(("true_positive", safety_label, code_id), {}).get("recurrent_responses", 0),
            -lookup.get(("false_positive", safety_label, code_id), {}).get("recurrent_responses", 0),
            code_id,
        ))[:top_n]
        selected[safety_label] = code_ids
        for code_id in code_ids:
            for outcome in outcomes:
                count = lookup.get((outcome, safety_label, code_id), {}).get("recurrent_responses", 0)
                maximum = max(maximum, 100 * count / outcome_counts[outcome])

    figure, axes = plt.subplots(2, 1, figsize=(6.7, 3.65), sharex=True)
    for axis, safety_label in zip(axes, ("harmful", "benign")):
        code_ids = selected[safety_label]
        positions = np.arange(len(code_ids))
        for offset, outcome in zip((-0.17, 0.17), outcomes):
            values = []
            for code_id in code_ids:
                row = lookup.get((outcome, safety_label, code_id), {})
                percentage = 100 * row.get("recurrent_responses", 0) / outcome_counts[outcome]
                values.append(percentage)
            axis.barh(
                positions + offset, values, height=0.30,
                color=outcome_colors[outcome], label=outcome_labels[outcome],
            )
        names = []
        for code_id in code_ids:
            row = next(
                (lookup.get((outcome, safety_label, code_id)) for outcome in outcomes
                 if lookup.get((outcome, safety_label, code_id)) is not None),
                None,
            )
            name = row["concept_name"] if row else None
            names.append(f"C{code_id}  {name or 'Unnamed concept'}")
        axis.set_yticks(positions, names)
        axis.invert_yaxis()
        axis.set_title(f"{safety_label.capitalize()} concepts", loc="left", pad=5)
        axis.grid(axis="x", color="#e1e7e9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
    upper = max(5, int(np.ceil(maximum / 5) * 5))
    axes[0].set_xlim(0, upper)
    figure.supxlabel("Detected responses with at least two separated occurrences (%)", y=0.02, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    figure.suptitle("Concepts recurring before detection", y=1.0, fontsize=10)
    figure.subplots_adjust(left=0.44, right=0.98, bottom=0.15, top=0.78, hspace=0.60)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def analyze_detection(cache_dir, concept_path, threshold_name, output_dir):
    """Connect detector decisions to concepts at their strongest conditional hazard."""
    # Load the concept catalog and the detector threshold used for this analysis.
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("format") != "trace_detection_audit_cache":
        raise ValueError(f"not a detection audit cache: {cache_dir}")
    _, catalog = _load_catalog(concept_path)
    try:
        threshold = float(manifest["streaming_thresholds"][threshold_name])
    except KeyError as error:
        choices = ", ".join(manifest.get("streaming_thresholds", {}))
        raise ValueError(f"unknown threshold {threshold_name!r}; choose {choices}") from error

    with np.load(cache_dir / manifest["arrays_file"], allow_pickle=False) as arrays:
        response_ids = arrays["response_ids"]
        labels = arrays["labels"]
        offsets = arrays["offsets"]
        code_ids = arrays["code_ids"]
        hazards = arrays["conditional_hazards"]
        risks = arrays["cumulative_risks"]
        if len(offsets) != len(response_ids) + 1 or offsets[-1] != len(code_ids):
            raise ValueError("invalid response offsets in detection audit cache")
        if not (len(code_ids) == len(hazards) == len(risks)):
            raise ValueError("unaligned arrays in detection audit cache")

        events = []
        prefixes_by_outcome = defaultdict(list)
        recurring_concepts = defaultdict(lambda: {
            "runs": 0, "responses": 0, "recurrent_responses": 0,
            "dominant_response_votes": 0.0,
        })

        # Reconstruct each response trace up to its first threshold crossing.
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

    # Save the response-level trace summaries before aggregating across outcomes.
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "detection_events.jsonl").open("w") as output_file:
        for event in events:
            output_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    # Rank concepts that recur within correctly and incorrectly detected responses.
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
            candidates = []
            for row in recurring_rows:
                if row["outcome"] == outcome and row["safety_label"] == safety_label:
                    candidates.append(row)
            most_repeated[outcome][safety_label] = candidates[0] if candidates else None

    grouped = {}
    for outcome in ("true_positive", "false_positive", "false_negative", "true_negative"):
        references = [event["strongest_signal_concept"] for event in events if event["outcome"] == outcome]
        grouped[outcome] = _label_summary(references)

    # Collect the aggregate tables and figures used by the paper.
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
    _plot_recurring_detection_concepts(
        recurring_rows, summary["outcomes"], output_dir / "detection_concepts.pdf"
    )
    _plot_recurring_detection_concepts(
        recurring_rows, summary["outcomes"], output_dir / "detection_concepts.svg"
    )
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
    harmful_rows = [row for row in rows if int(row["label"]) == 1]
    edited_rows = [row for row in harmful_rows if row["edit_positions"]]
    source_responses, target_responses, pair_responses = defaultdict(set), defaultdict(set), defaultdict(set)
    source_names, target_names = defaultdict(Counter), defaultdict(Counter)
    pair_names = defaultdict(lambda: {"source": Counter(), "target": Counter()})
    pair_codes = defaultdict(set)

    # Count each named concept or concept pair at most once per response.
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


def _plot_intervention_roles(records, output_path, top_n=5):
    """Plot harmful source concepts and benign targets separately at response level."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    panels = [
        ("harmful_source_concepts", "Harmful concept triggering edit", SAFETY_COLORS["harmful"]),
        ("benign_target_concepts", "Benign concept used as target", SAFETY_COLORS["benign"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), sharex=True)
    denominator = records["n_edited_harmful_responses"]
    for axis, (field, title, color) in zip(axes, panels):
        selected = records[field][:top_n][::-1]
        values = [100 * row["response_fraction"] for row in selected]
        positions = np.arange(len(selected))
        labels = ["\n".join(_wrap_label(row["name"], 29)) for row in selected]
        axis.barh(positions, values, height=0.62, color=color, edgecolor="white", linewidth=0.6)
        axis.set_yticks(positions, labels)
        axis.set_xlim(0, 100)
        axis.set_title(title, pad=7, fontweight="semibold")
        axis.set_xlabel("Edited harmful responses (%)")
        axis.grid(axis="x", color="#e1e7e9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        for position, value, row in zip(positions, values, selected):
            inside = value > 84
            axis.text(value - 2 if inside else value + 1.5, position,
                      f"{row['responses']}/{denominator}", va="center",
                      ha="right" if inside else "left", fontsize=7,
                      color="white" if inside else "#40515a")
    figure.subplots_adjust(left=0.23, right=0.98, bottom=0.20, top=0.88, wspace=0.75)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_steering_paths(pairs, output_path, top_n=6):
    """Show the most common named source-to-target edit paths."""
    named_pairs = [pair for pair in pairs if pair["source"].get("name") and pair["target"].get("name")]
    selected = named_pairs[:top_n]
    if not selected:
        return False

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(7.2, 0.72 * len(selected) + 1.35))
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.5, len(selected) - 0.1)
    axis.axis("off")
    axis.text(0.22, len(selected) - 0.25, "Concept triggering edit", ha="center", fontweight="semibold")
    axis.text(0.78, len(selected) - 0.25, "Selected target", ha="center", fontweight="semibold")

    maximum = max(pair["responses"] for pair in selected)
    for row_index, pair in enumerate(selected):
        y = len(selected) - row_index - 1.05
        source, target = pair["source"], pair["target"]
        source_color = SAFETY_COLORS[source["safety_label"]]
        target_color = SAFETY_COLORS[target["safety_label"]]
        source_name = "\n".join(_wrap_label(source["name"], 30))
        target_name = "\n".join(_wrap_label(target["name"], 30))
        axis.text(0.22, y, source_name, ha="center", va="center", fontsize=8,
                  bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
                        "edgecolor": source_color, "linewidth": 1.5})
        axis.text(0.78, y, target_name, ha="center", va="center", fontsize=8,
                  bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
                        "edgecolor": target_color, "linewidth": 1.5})
        width = 0.8 + 3.2 * pair["responses"] / maximum
        axis.annotate("", xy=(0.64, y), xytext=(0.36, y),
                      arrowprops={"arrowstyle": "-|>", "color": "#60717a", "lw": width,
                                  "shrinkA": 3, "shrinkB": 3})
        axis.text(0.50, y + 0.09, f"{pair['responses']} responses", ha="center", va="bottom",
                  fontsize=7, color="#51636c")

    used_labels = []
    for pair in selected:
        for endpoint in (pair["source"], pair["target"]):
            if endpoint["safety_label"] not in used_labels:
                used_labels.append(endpoint["safety_label"])
    handles = [Patch(facecolor="white", edgecolor=SAFETY_COLORS[label], linewidth=1.5,
                     label=SAFETY_NAMES[label]) for label in used_labels]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                  ncol=len(handles), frameon=False, fontsize=7)
    figure.suptitle("Most frequent named concept edits", y=0.99, fontsize=10)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.88)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return True


def _wrap_label(text, width):
    words, lines, current = text.split(), [], []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def analyze_steering(cache_dir, concept_path, result_key, output_dir):
    """Summarize the source and target concepts recorded by a steering run."""
    # Load the selected steering setting and its concept labels.
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

    # Count source concepts, target concepts, and source-to-target pairs across edits.
    for row in selected:
        sources, targets = row["source_code_ids"], row["target_code_ids"]
        if not (len(row["edit_positions"]) == len(sources) == len(targets)):
            raise ValueError(f"unaligned steering edits for response {row['response_id']}")
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

    # Summarize event-level and response-level coverage before writing the artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    named_pair_edits = 0
    for (source, target), edits in pair_counts.items():
        source_name = _concept_reference(source, catalog).get("name")
        target_name = _concept_reference(target, catalog).get("name")
        if source_name and target_name:
            named_pair_edits += edits
    groups = {
        "harmful": _steering_group_summary([row for row in selected if int(row["label"]) == 1], catalog),
        "safe": _steering_group_summary([row for row in selected if int(row["label"]) == 0], catalog),
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
        "path_figure_files": ["steering_concept_paths.pdf", "steering_concept_paths.svg"],
        "role_figure_files": ["steering_concept_roles.pdf", "steering_concept_roles.svg"],
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
    plot_rows = [
        {"name": "Concept triggering edit", **source_balanced},
        {"name": "Selected target concept", **target_balanced},
    ]
    _plot_label_bars(plot_rows, output_dir / "steering_concepts.pdf", "Concepts used for steering")
    _plot_label_bars(plot_rows, output_dir / "steering_concepts.svg", "Concepts used for steering")
    for extension in ("pdf", "svg"):
        _plot_steering_paths(top_pairs, output_dir / f"steering_concept_paths.{extension}")
        _plot_intervention_roles(
            intervention_records, output_dir / f"steering_concept_roles.{extension}"
        )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    detection = subparsers.add_parser("detection", help="analyze a saved detector token cache")
    detection.add_argument("--cache", required=True)
    detection.add_argument("--concepts", required=True)
    detection.add_argument("--threshold", default="fpr05")
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
