"""Tests for the VQ training diagnostics, including the end-to-end region summary."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vq.diagnostics import (
    _starts_word,
    append_diagnostic_record,
    gradient_norms_by_component,
    represent_firing,
    summarize_initial_codebook,
    summarize_region_changes,
    summarize_training_batch,
)
from vq.model import CrossLayerVQVAE

from conftest import FakeTokenizer, WordTokenizer

DIMENSION = 8
NUM_CODES = 8


def build_model():
    return CrossLayerVQVAE(NUM_CODES, DIMENSION, decoder_layers=1, nhead=2, ff_mult=2, dropout=0.0)


def build_sequences(labels, length=6, dimension=DIMENSION, seed=0):
    generator = np.random.RandomState(seed)
    sequences = []
    for index, label in enumerate(labels):
        sequences.append(
            {
                "idx": index,
                "label": label,
                "x": torch.tensor(generator.standard_normal((length, dimension)), dtype=torch.float32),
                "token_ids": [1 + ((index + position) % 3) for position in range(length)],
            }
        )
    return sequences


class TestRepresentFiring:
    def test_token_mode(self, word_tokenizer):
        assert represent_firing([0, 1, 2], 1, word_tokenizer, "token") == " cat"

    def test_word_mode_returns_the_whole_word(self, word_tokenizer):
        assert represent_firing([0, 1, 2], 1, word_tokenizer, "word") == "cat"

    def test_word_mode_walks_back_through_subword_tokens(self):
        tokenizer = FakeTokenizer({0: " hel", 1: "lo", 2: " wor", 3: "ld"})
        assert represent_firing([0, 1, 2, 3], 1, tokenizer, "word") == "hello"

    def test_phrase_mode_marks_the_firing_token(self, word_tokenizer):
        representation = represent_firing([0, 1, 2], 1, word_tokenizer, "phrase", context_len=1)
        assert representation == " the« cat» sat"

    def test_phrase_mode_clamps_to_the_sequence_bounds(self, word_tokenizer):
        assert represent_firing([0, 1, 2], 0, word_tokenizer, "phrase", context_len=4).startswith("« the»")

    def test_starts_word_detects_leading_space(self):
        assert _starts_word(WordTokenizer(["word"]), 0) is True
        assert _starts_word(FakeTokenizer({0: "tail"}), 0) is False


class TestSummarizeRegionChanges:
    def statistics(self, **extra):
        statistics = {"signed_harmfulness": np.array([0.5, -0.2, 0.0, 0.3])}
        statistics.update(extra)
        return statistics

    def test_counts_harmful_and_benign_codes(self):
        record = summarize_region_changes(self.statistics(), {0, 3})
        assert record["n_harmful_codes"] == 2
        assert record["n_benign_codes"] == 2
        assert record["median_absolute_score"] == pytest.approx(0.25)

    def test_without_previous_regions_nothing_changed(self):
        assert summarize_region_changes(self.statistics(), {0})["n_changed_codes"] == 0

    def test_reports_symmetric_difference(self):
        record = summarize_region_changes(self.statistics(), {0, 1}, previous_harmful_codes={0, 3})
        assert record["changed_codes"] == [1, 3]
        assert record["n_changed_codes"] == 2

    def test_counts_scores_near_the_decision_boundary(self):
        record = summarize_region_changes(self.statistics(), set())
        assert record["n_scores_within_0.01_of_boundary"] == 1  # the exact 0.0 score

    def test_includes_response_support_when_available(self):
        record = summarize_region_changes(
            self.statistics(response_counts=np.array([1, 12, 3, 40])), set()
        )
        assert record["response_support_min"] == 1
        assert record["response_support_max"] == 40
        assert record["n_codes_seen_in_fewer_than_10_responses"] == 2

    def test_omits_response_support_when_unavailable(self):
        assert "response_support_min" not in summarize_region_changes(self.statistics(), set())


class TestSummarizeTrainingBatch:
    def fake_output(self, batch_size=2, length=3):
        return {
            "reconstructed": torch.randn(batch_size, length, DIMENSION),
            "indices": torch.randint(0, NUM_CODES, (batch_size, length)),
            "z_e": torch.randn(batch_size, length, DIMENSION),
            "quantized": torch.randn(batch_size, length, DIMENSION),
            "min_distances": torch.rand(batch_size, length),
            "commit_loss": torch.tensor(0.1),
            "perplexity": torch.tensor(3.0),
        }

    def build_batch(self):
        return [
            {"idx": 1, "label": 1, "x": torch.zeros(3, DIMENSION)},
            {"idx": 2, "label": 0, "x": torch.zeros(2, DIMENSION)},
        ]

    def test_reports_batch_metadata(self):
        quantizer = SimpleNamespace(codebook=torch.randn(NUM_CODES, DIMENSION))
        record = summarize_training_batch(
            self.build_batch(),
            torch.randn(2, 3, DIMENSION),
            torch.randn(2, 3, DIMENSION),
            torch.tensor([[0, 1, 1], [0, 1, -1]]),
            self.fake_output(),
            quantizer,
            quantizer.codebook.clone(),
        )
        assert record["response_ids"] == [1, 2]
        assert record["labels"] == [1, 0]
        assert record["lengths"] == [3, 2]
        assert set(record["losses"]) == {"relative_reconstruction_mean",
                                         "relative_reconstruction_max",
                                         "commitment", "perplexity"}
        assert set(record["vector_norms"]) == {
            "input", "encoded", "quantized", "target", "reconstructed"
        }
        assert record["codebook_update"]["most_moved_code"] < NUM_CODES
        assert np.isfinite(record["codebook_update"]["decoder_assignment_distance_mean"])

    def test_ignores_padded_positions(self):
        quantizer = SimpleNamespace(codebook=torch.randn(NUM_CODES, DIMENSION))
        batch = self.build_batch()
        labels = torch.tensor([[0, 1, -1], [0, 1, 1]])
        record = summarize_training_batch(
            batch, torch.randn(2, 3, DIMENSION), torch.randn(2, 3, DIMENSION),
            labels, self.fake_output(), quantizer, quantizer.codebook.clone(),
        )
        largest_error = record["largest_error_token"]
        row = record["response_ids"].index(largest_error["response_id"])
        assert labels[row, largest_error["position"]] >= 0


class TestGradientNormsByComponent:
    def test_groups_parameters_by_component(self):
        model = build_model()
        model.train()
        activations = torch.randn(2, 4, DIMENSION)
        output = model(activations)
        (output["reconstructed"].pow(2).mean() + output["loss"]).backward()
        norms = gradient_norms_by_component(model)
        assert {"activation_encoder", "output_projection", "attention",
                "feed_forward", "decoder_norms"} <= set(norms)
        assert all(np.isfinite(value) and value > 0 for value in norms.values())

    def test_skips_parameters_without_gradients(self):
        model = build_model()
        assert gradient_norms_by_component(model) == {}


class TestAppendDiagnosticRecord:
    def test_appends_newline_delimited_json(self, tmp_path):
        path = tmp_path / "diagnostics.jsonl"
        append_diagnostic_record(path, {"epoch": 1, "loss": np.float32(0.5)})
        append_diagnostic_record(path, {"epoch": 2, "loss": 0.25})
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records == [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.25}]


class TestSummarizeInitialCodebook:
    def summarize(self, labels, score_method="response_presence", prior_strength=10.0):
        model = build_model()
        model.eval()
        sequences = build_sequences(labels, seed=1)
        return summarize_initial_codebook(
            model, sequences, {"init": "spherical"}, FakeTokenizer(), torch.device("cpu"),
            score_method, prior_strength,
        )

    def test_returns_statistics_text_and_regions(self):
        statistics, text, regions = self.summarize([1, 1, 0, 0])
        assert statistics["K"] == NUM_CODES
        assert statistics["n_train_tokens"] == 24
        assert statistics["base_rate_unit"] == "responses"
        assert regions["source"] == "smoothed_response_enrichment"
        assert "init clusters" in text
        assert "code" in text

    @pytest.mark.parametrize(
        "score_method,expected_source",
        [
            ("response_presence", "smoothed_response_enrichment"),
            ("response_frequency", "smoothed_response_frequency_enrichment"),
            ("token_occurrence", "token_occurrence_enrichment"),
        ],
    )
    def test_every_score_method_produces_a_region_split(self, score_method, expected_source):
        statistics, _, regions = self.summarize([1, 1, 0, 0], score_method)
        assert statistics["code_score_method"] == score_method
        assert regions["source"] == expected_source
        assert len(regions["harmful_codes"]) + len(regions["benign_codes"]) == NUM_CODES
        assert np.isfinite(regions["signed_harmfulness"]).all()

    def test_all_harmful_training_set_stays_finite(self):
        """Regression: a base rate of 1.0 used to make every region score NaN.

        The NaN then flowed into the checkpoint's ``signed_harmfulness`` and into the
        classifier's code features, and it collapsed the harmful/benign split.
        """
        statistics, _, regions = self.summarize([1, 1, 1, 1])
        assert statistics["base_rate"] == pytest.approx(1.0)
        assert np.isfinite(regions["signed_harmfulness"]).all()
        assert not np.isnan(regions["signed_harmfulness"]).any()
        assert np.isfinite(statistics["harm_enrichment_mean"])

    def test_unknown_score_method_raises(self):
        with pytest.raises(ValueError, match="unknown code score method"):
            self.summarize([1, 0], "response_presnce")

    def test_prior_strength_is_recorded_only_for_response_methods(self):
        assert self.summarize([1, 0], "response_presence")[0]["code_score_prior_strength"] == 10.0
        assert self.summarize([1, 0], "token_occurrence")[0]["code_score_prior_strength"] is None

    def test_details_only_carries_support_for_the_matching_method(self):
        _, _, presence_regions = self.summarize([1, 1, 0, 0], "response_presence")
        assert "response_counts" in presence_regions
        assert "token_counts" not in presence_regions
        _, _, frequency_regions = self.summarize([1, 1, 0, 0], "response_frequency")
        assert "response_frequency_mass" in frequency_regions
        _, _, token_regions = self.summarize([1, 1, 0, 0], "token_occurrence")
        assert "token_counts" in token_regions
