"""Tests for the VQ codebook code-harmfulness estimators."""

import sys
import warnings

import numpy as np
import pytest

from vq import codebook
from vq.codebook import (
    CODE_SCORE_METHODS,
    SCORE_DENOMINATOR_FLOOR,
    _require_faiss,
    _validate_num_codes,
    code_harmfulness_statistics,
    code_score_region_source,
    signed_harmfulness_scores,
    smoothed_response_code_statistics,
    smoothed_response_frequency_statistics,
    split_assignments_by_response,
    token_occurrence_code_statistics,
)

# Two responses: a harmful one firing codes {0, 1}, a safe one firing codes {0, 2}.
SEQUENCES = [np.array([0, 1]), np.array([0, 2])]
LABELS = [1, 0]
NUM_CODES = 3


def test_module_imports_without_faiss():
    """FAISS is only needed to initialize a codebook, never to score one."""
    assert not hasattr(codebook, "faiss")
    assert "faiss" not in sys.modules


def test_require_faiss_reports_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(ModuleNotFoundError, match="faiss-cpu"):
        _require_faiss()


def test_validate_num_codes_rejects_nonpositive():
    with pytest.raises(ValueError, match="num_codes must be positive"):
        _validate_num_codes(0)
    with pytest.raises(ValueError, match="num_codes must be positive"):
        _validate_num_codes(-1)
    assert _validate_num_codes(4) == 4


class TestSignedHarmfulnessScores:
    def test_zero_at_base_rate(self):
        scores = signed_harmfulness_scores(np.array([0.5, 0.5]), 0.5)
        assert np.array_equal(scores, np.zeros(2))

    def test_positive_above_and_negative_below_base_rate(self):
        scores = signed_harmfulness_scores(np.array([0.75, 0.25]), 0.5)
        assert scores[0] == pytest.approx(0.5)
        assert scores[1] == pytest.approx(-0.5)

    @pytest.mark.parametrize("base_rate", [0.0, 1.0])
    def test_degenerate_base_rate_stays_finite(self, base_rate):
        """Regression: an all-safe or all-harmful partition used to divide by zero."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            scores = signed_harmfulness_scores(np.full(3, base_rate), base_rate)
        assert np.isfinite(scores).all()
        assert np.array_equal(scores, np.zeros(3))

    def test_returns_float64(self):
        assert signed_harmfulness_scores(np.array([0], dtype=np.int8), 0.5).dtype == np.float64
        assert SCORE_DENOMINATOR_FLOOR > 0


class TestResponsePresenceStatistics:
    def test_counts_each_code_once_per_response(self):
        statistics = smoothed_response_code_statistics(SEQUENCES, LABELS, NUM_CODES)
        assert statistics["n_responses"] == 2
        assert statistics["n_harmful_responses"] == 1
        assert statistics["n_safe_responses"] == 1
        assert np.array_equal(statistics["response_counts"], [2, 1, 1])
        assert np.array_equal(statistics["harmful_response_counts"], [1, 1, 0])
        assert statistics["base_harmful_response_rate"] == pytest.approx(0.5)

    def test_smoothed_probability_and_signed_score(self):
        statistics = smoothed_response_code_statistics(SEQUENCES, LABELS, NUM_CODES)
        # code 0 fires in both responses, code 1 only harmful, code 2 only safe.
        assert statistics["smoothed_harmful_probability"] == pytest.approx([0.5, 6 / 11, 5 / 11])
        assert statistics["signed_harmfulness"] == pytest.approx([0.0, 1 / 11, -1 / 11])

    def test_repeated_codes_within_a_response_count_once(self):
        statistics = smoothed_response_code_statistics([np.array([1, 1, 1])], [1], NUM_CODES)
        assert statistics["response_counts"][1] == 1

    def test_normalized_support_is_bounded(self):
        statistics = smoothed_response_code_statistics(SEQUENCES, LABELS, NUM_CODES)
        support = statistics["normalized_log_response_support"]
        assert support.max() == pytest.approx(1.0)
        assert (support >= 0).all()

    def test_all_harmful_partition_is_finite(self):
        """Regression: base rate 1.0 previously returned NaN for every code."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            statistics = smoothed_response_code_statistics(SEQUENCES, [1, 1], NUM_CODES)
        assert statistics["base_harmful_response_rate"] == pytest.approx(1.0)
        assert not np.isnan(statistics["signed_harmfulness"]).any()
        assert np.array_equal(statistics["signed_harmfulness"], np.zeros(NUM_CODES))

    def test_all_safe_partition_is_finite(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            statistics = smoothed_response_code_statistics(SEQUENCES, [0, 0], NUM_CODES)
        assert np.array_equal(statistics["signed_harmfulness"], np.zeros(NUM_CODES))

    def test_empty_partition_has_zero_base_rate(self):
        statistics = smoothed_response_code_statistics([], [], NUM_CODES)
        assert statistics["base_harmful_response_rate"] == 0.0
        assert np.array_equal(statistics["signed_harmfulness"], np.zeros(NUM_CODES))
        # No response ever touched a code, so the prior keeps every probability at 0.
        assert np.array_equal(statistics["smoothed_harmful_probability"], np.zeros(NUM_CODES))


class TestResponseFrequencyStatistics:
    def test_unit_mass_per_response(self):
        statistics = smoothed_response_frequency_statistics(SEQUENCES, LABELS, NUM_CODES)
        assert statistics["response_frequency_mass"] == pytest.approx([1.0, 0.5, 0.5])
        assert statistics["harmful_response_frequency_mass"] == pytest.approx([0.5, 0.5, 0.0])

    def test_signed_score_divides_frequency_mass(self):
        statistics = smoothed_response_frequency_statistics(SEQUENCES, LABELS, NUM_CODES)
        assert statistics["smoothed_harmful_probability"] == pytest.approx([0.5, 5.5 / 10.5, 5 / 10.5])
        assert statistics["signed_harmfulness"] == pytest.approx([0.0, 1 / 21, -1 / 21])

    def test_empty_response_is_skipped(self):
        statistics = smoothed_response_frequency_statistics(
            [np.array([], dtype=np.int64), np.array([1])], [1, 0], NUM_CODES
        )
        assert statistics["response_counts"][1] == 1
        assert statistics["response_frequency_mass"][1] == pytest.approx(1.0)

    def test_all_harmful_partition_is_finite(self):
        """Regression: base rate 1.0 previously returned NaN for every code."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            statistics = smoothed_response_frequency_statistics(SEQUENCES, [1, 1], NUM_CODES)
        assert not np.isnan(statistics["signed_harmfulness"]).any()


class TestTokenOccurrenceStatistics:
    def test_counts_every_occurrence(self):
        statistics = token_occurrence_code_statistics(
            [np.array([0, 1, 1]), np.array([0, 2])], LABELS, NUM_CODES
        )
        assert np.array_equal(statistics["token_counts"], [2, 2, 1])
        assert np.array_equal(statistics["harmful_token_counts"], [1, 2, 0])
        assert statistics["n_tokens"] == 5
        assert statistics["n_harmful_tokens"] == 3

    def test_base_rate_is_per_token(self):
        statistics = token_occurrence_code_statistics(
            [np.array([0, 1, 1]), np.array([0, 2])], LABELS, NUM_CODES
        )
        assert statistics["base_harmful_token_rate"] == pytest.approx(0.6)
        assert statistics["signed_harmfulness"] == pytest.approx([-1 / 6, 1.0, -1.0])

    def test_empty_partition_does_not_divide_by_zero(self):
        """Regression: an empty split raised ZeroDivisionError instead of scoring."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            statistics = token_occurrence_code_statistics([], [], NUM_CODES)
        assert statistics["base_harmful_token_rate"] == 0.0
        assert statistics["n_tokens"] == 0
        assert np.array_equal(statistics["signed_harmfulness"], np.zeros(NUM_CODES))

    def test_responses_without_tokens_are_tolerated(self):
        statistics = token_occurrence_code_statistics(
            [np.array([], dtype=np.int64), np.array([1, 1])], [1, 0], NUM_CODES
        )
        assert statistics["n_tokens"] == 2
        assert statistics["base_harmful_token_rate"] == 0.0
        assert np.array_equal(statistics["harmful_probability"], [0.0, 0.0, 0.0])

    def test_unseen_code_falls_back_to_base_rate(self):
        statistics = token_occurrence_code_statistics([np.array([0, 0])], [1], NUM_CODES)
        assert statistics["harmful_probability"][1] == pytest.approx(1.0)
        assert statistics["harmful_probability"][0] == pytest.approx(1.0)


class TestDispatcher:
    @pytest.mark.parametrize(
        "score_method,probability_key",
        [
            ("response_presence", "smoothed_harmful_probability"),
            ("response_frequency", "smoothed_harmful_probability"),
            ("token_occurrence", "harmful_probability"),
        ],
    )
    def test_dispatches_to_the_matching_estimator(self, score_method, probability_key):
        statistics = code_harmfulness_statistics(score_method, SEQUENCES, LABELS, NUM_CODES)
        assert probability_key in statistics
        assert "signed_harmfulness" in statistics

    def test_matches_direct_calls(self):
        dispatched = code_harmfulness_statistics("response_presence", SEQUENCES, LABELS, NUM_CODES)
        direct = smoothed_response_code_statistics(SEQUENCES, LABELS, NUM_CODES)
        assert np.array_equal(dispatched["signed_harmfulness"], direct["signed_harmfulness"])

    def test_unknown_method_raises_instead_of_falling_back(self):
        """Regression: unknown methods silently used the token-occurrence estimator."""
        with pytest.raises(ValueError, match="unknown code score method"):
            code_harmfulness_statistics("response_presense", SEQUENCES, LABELS, NUM_CODES)

    def test_every_declared_method_is_accepted(self):
        for score_method in CODE_SCORE_METHODS:
            code_harmfulness_statistics(score_method, SEQUENCES, LABELS, NUM_CODES)


class TestRegionSource:
    def test_mapping_is_complete(self):
        for score_method in CODE_SCORE_METHODS:
            assert code_score_region_source(score_method).endswith("enrichment")

    def test_known_labels(self):
        assert code_score_region_source("response_presence") == "smoothed_response_enrichment"
        assert code_score_region_source("token_occurrence") == "token_occurrence_enrichment"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown code score method"):
            code_score_region_source("nope")


class TestSplitAssignments:
    def test_splits_by_response_length(self):
        sequences = [{"x": np.zeros((3, 2))}, {"x": np.zeros((2, 2))}]
        assignments = np.arange(5)
        split = split_assignments_by_response(sequences, assignments)
        assert len(split) == 2
        assert np.array_equal(split[0], [0, 1, 2])
        assert np.array_equal(split[1], [3, 4])

    def test_empty_input(self):
        assert split_assignments_by_response([], np.array([])) == []
