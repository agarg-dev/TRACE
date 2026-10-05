#!/usr/bin/env python
"""Summarize the response-quality diagnostics used for steering selection."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def mean(values):
    available = [value for value in values if value is not None]
    return statistics.mean(available) if available else float("nan")


def prose_ok(text):
    """Apply the length, alphabetic-content, and repeated-token checks."""
    text = (text or "").strip()
    non_space_characters = [character for character in text if not character.isspace()]
    if len(non_space_characters) < 20:
        return False
    letter_fraction = sum(character.isalpha() for character in non_space_characters) / len(non_space_characters)
    words = text.split()
    return letter_fraction >= 0.6 and Counter(words).most_common(1)[0][1] / len(words) <= 0.30


def non_repetition(variant):
    return variant["non_repetition"] if prose_ok(variant["text"]) else 0.0


def edited_fraction(variant):
    edited, generated = variant["steered"].split("/", 1)
    return int(edited) / int(generated) if int(generated) else 0.0


def quality_ok(variant, repetition_floor, max_nll_increase, max_edited_fraction):
    fraction = edited_fraction(variant) if "steered" in variant else 0.0
    return (
        non_repetition(variant) > repetition_floor
        and variant["base_model_nll_increase"] <= max_nll_increase
        and fraction <= max_edited_fraction
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="steering run directory or intervene.json")
    parser.add_argument("--repetition-floor", type=float, default=0.5)
    parser.add_argument("--max-nll-increase", type=float, default=1.0)
    parser.add_argument("--max-steered-fraction", type=float, default=0.9)
    parser.add_argument("--out", help="report path (default: <run>/steer_stats.txt)")
    args = parser.parse_args()

    run_path = Path(args.run)
    input_path = run_path if run_path.suffix == ".json" else run_path / "intervene.json"
    output_path = Path(args.out) if args.out else input_path.parent / "steer_stats.txt"
    run = json.loads(input_path.read_text())
    harmful_records = [record for record in run["results"] if record["label"] == 1]
    safe_records = [record for record in run["results"] if record["label"] == 0]

    lines = [
        f"Steering quality: {input_path}",
        "method lambda harmful_non_repetition harmful_delta_nll harmful_edited_fraction safe_broken",
    ]
    for recipe in run["recipes"]:
        for strength in run["lams"]:
            variant_key = f"{recipe}_lam{strength:g}"
            harmful_variants = [record["steered"][variant_key] for record in harmful_records]
            safe_pairs = [(record["baseline"], record["steered"][variant_key]) for record in safe_records]
            safe_broken = sum(
                quality_ok(before, args.repetition_floor, args.max_nll_increase, args.max_steered_fraction)
                and not quality_ok(after, args.repetition_floor, args.max_nll_increase,
                                   args.max_steered_fraction)
                for before, after in safe_pairs
            )
            line = (
                f"{recipe} {strength:g} "
                f"{mean([non_repetition(variant) for variant in harmful_variants]):.3f} "
                f"{mean([variant['base_model_nll_increase'] for variant in harmful_variants]):.3f} "
                f"{mean([edited_fraction(variant) for variant in harmful_variants]):.3f} "
                f"{safe_broken}/{len(safe_records)}"
            )
            lines.append(line)

    report = "\n".join(lines) + "\n"
    print(report, end="")
    output_path.write_text(report)


if __name__ == "__main__":
    main()
