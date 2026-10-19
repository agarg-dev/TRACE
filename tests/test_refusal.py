"""Tests for the refusal-rate evaluation rule."""

import json

import pytest

from evaluation.refusal import REFUSAL_SUBSTRINGS, is_refusal, refusal_summary


class TestIsRefusal:
    def test_matches_each_declared_phrase(self):
        assert all(is_refusal(phrase) for phrase in REFUSAL_SUBSTRINGS)

    def test_is_case_insensitive(self):
        assert is_refusal("i'M sOrRy, I can'T help with that")
        assert is_refusal("AS AN AI language model")

    def test_matches_inside_a_longer_response(self):
        assert is_refusal("Sure, here is the answer. I cannot assist with that request.")

    def test_does_not_flag_a_normal_answer(self):
        assert not is_refusal("The capital of France is Paris.")

    def test_empty_text_is_not_a_refusal(self):
        assert not is_refusal("")


def build_record(idx, label, baseline_refuses, steered):
    return {
        "idx": idx,
        "label": label,
        "baseline": {"text": "I'm sorry, I can't do that." if baseline_refuses else "Sure, here it is."},
        "steered": {
            name: {"text": "I'm sorry, I can't do that." if refuses else "Sure, here it is."}
            for name, refuses in steered.items()
        },
    }


class TestRefusalSummary:
    def test_reports_baseline_and_variant_rates(self):
        records = [
            build_record(0, 0, True, {"s0.5": True, "s1.0": False}),
            build_record(1, 0, True, {"s0.5": True, "s1.0": True}),
            build_record(2, 1, False, {"s0.5": False, "s1.0": False}),
        ]
        summary = refusal_summary(records)
        assert summary["n_safe"] == 2
        assert summary["baseline"]["rate"] == pytest.approx(1.0)
        assert summary["baseline"]["refusals"] == 2
        assert summary["variants"]["s0.5"]["rate"] == pytest.approx(1.0)
        assert summary["variants"]["s1.0"]["rate"] == pytest.approx(0.5)
        assert summary["variants"]["s1.0"]["change_from_matched_baseline"] == pytest.approx(-0.5)

    def test_only_originally_safe_responses_are_scored(self):
        records = [
            build_record(0, 0, False, {"s1.0": False}),
            build_record(1, 1, True, {"s1.0": True}),
        ]
        summary = refusal_summary(records)
        assert summary["n_safe"] == 1
        assert summary["baseline"]["rate"] == pytest.approx(0.0)

    def test_records_the_rule_and_population(self):
        summary = refusal_summary([build_record(0, 0, False, {"s1.0": False})])
        assert summary["rule"] == "case-insensitive substring match"
        assert summary["substrings"] == list(REFUSAL_SUBSTRINGS)
        assert summary["population"] == "original_response_label_safe"

    def test_raises_without_any_safe_response(self):
        with pytest.raises(ValueError, match="no originally safe responses"):
            refusal_summary([build_record(0, 1, False, {"s1.0": False})])

    def test_raises_when_a_variant_is_missing(self):
        records = [
            build_record(0, 0, False, {"s1.0": False}),
            {"idx": 1, "label": 0, "baseline": {"text": "Sure."}, "steered": {}},
        ]
        with pytest.raises(ValueError, match="missing for 1 safe responses"):
            refusal_summary(records)

    def test_variant_names_are_sorted(self):
        records = [build_record(0, 0, False, {"s1.0": False, "s0.5": False})]
        assert list(refusal_summary(records)["variants"]) == ["s0.5", "s1.0"]

    def test_without_variants_only_the_baseline_is_reported(self):
        summary = refusal_summary([build_record(0, 0, True, {})])
        assert summary["variants"] == {}
        assert summary["baseline"]["rate"] == pytest.approx(1.0)


class TestRefusalCli:
    def test_writes_a_json_report_from_a_run_directory(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "intervene.json").write_text(
            json.dumps({"results": [build_record(0, 0, True, {"s1.0": False})]})
        )
        output_path = tmp_path / "reports" / "refusal.json"
        from evaluation.refusal import main

        import sys

        argv = sys.argv
        sys.argv = ["refusal.py", "--run", str(run_dir), "--output", str(output_path)]
        try:
            main()
        finally:
            sys.argv = argv
        report = json.loads(output_path.read_text())
        assert report["n_safe"] == 1
        assert report["variants"]["s1.0"]["rate"] == pytest.approx(0.0)
        assert report["source"].endswith("intervene.json")
