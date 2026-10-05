"""Tests for the vector quantizer and the cross-layer VQ-VAE (CPU only)."""

import numpy as np
import pytest
import torch

from vq.model import (
    AdaptiveResidualEncoder,
    CrossLayerVQVAE,
    PassThroughEncoder,
    VectorQuantizerEMA,
    _apply_rope,
    _rope_tables,
    causal_mask,
    load_steering_vq_checkpoint,
    load_vq_checkpoint,
)

DIMENSION = 8
NUM_CODES = 6


@pytest.fixture
def quantizer():
    return VectorQuantizerEMA(NUM_CODES, DIMENSION, use_sampling=False)


class TestVectorQuantizerEMA:
    def test_properties(self, quantizer):
        assert quantizer.num_codes == NUM_CODES
        assert quantizer.code_dimension == DIMENSION
        assert quantizer.codebook.shape == (NUM_CODES, DIMENSION)

    def test_top_k_is_clamped_to_num_codes(self):
        quantizer = VectorQuantizerEMA(3, DIMENSION, top_k=10)
        assert quantizer._top_k == 3

    def test_forward_shapes(self, quantizer):
        inputs = torch.randn(2, 5, DIMENSION)
        output = quantizer(inputs)
        assert output["quantized"].shape == (2, 5, DIMENSION)
        assert output["indices"].shape == (2, 5)
        assert output["min_distances"].shape == (2, 5)
        assert output["encoding_indices"].shape == (10,)

    def test_quantized_matches_codebook_snapshot(self, quantizer):
        """The decoder consumes the quantized vectors, not the encoder output."""
        quantizer.eval()
        output = quantizer(torch.randn(3, DIMENSION))
        expected = output["codebook_snapshot"][output["indices"]]
        assert torch.allclose(output["quantized"], expected, atol=1e-6)

    def test_straight_through_routes_gradient_to_the_encoder(self):
        quantizer = VectorQuantizerEMA(NUM_CODES, DIMENSION, use_sampling=False, dcr_enabled=False)
        quantizer.train()
        inputs = torch.randn(3, DIMENSION, requires_grad=True)
        quantizer(inputs)["quantized"].sum().backward()
        assert inputs.grad is not None
        assert torch.allclose(inputs.grad, torch.ones_like(inputs))

    def test_padding_rows_get_zero_index_and_distance(self, quantizer):
        inputs = torch.randn(1, 4, DIMENSION)
        inputs[0, 2:] = 0.0
        output = quantizer(inputs)
        assert torch.equal(output["indices"][0, 2:], torch.zeros(2, dtype=torch.long))
        assert torch.equal(output["min_distances"][0, 2:], torch.zeros(2))
        assert torch.equal(output["quantized"][0, 2:], torch.zeros(2, DIMENSION))

    def test_perplexity_within_bounds(self, quantizer):
        output = quantizer(torch.randn(8, DIMENSION))
        perplexity = float(output["perplexity"])
        assert 1.0 - 1e-6 <= perplexity <= NUM_CODES + 1e-6

    def test_eval_mode_does_not_update_the_codebook(self, quantizer):
        quantizer.eval()
        before = quantizer.codebook.detach().clone()
        quantizer(torch.randn(4, DIMENSION))
        assert torch.equal(quantizer.codebook, before)

    def test_usage_is_tracked_in_eval_mode_and_resettable(self, quantizer):
        """Usage accounting runs in every mode; the codebook geometry does not change."""
        quantizer.eval()
        quantizer(torch.randn(4, DIMENSION))
        assert float(quantizer._usage_count.sum()) == 4.0
        quantizer.reset_usage_stats()
        assert float(quantizer._usage_count.sum()) == 0.0

    def test_training_mode_updates_codebook_and_usage(self, quantizer):
        quantizer.train()
        before = quantizer.codebook.detach().clone()
        inputs = torch.randn(16, DIMENSION) * 3
        quantizer(inputs)
        assert not torch.equal(quantizer.codebook, before)
        assert float(quantizer._usage_count.sum()) > 0

    def test_codebook_vectors_stay_finite(self, quantizer):
        quantizer.train()
        for _ in range(3):
            quantizer(torch.randn(8, DIMENSION) * 5 + 100)
        assert torch.isfinite(quantizer.codebook).all()

    def test_assign_indices_matches_nearest_code(self, quantizer):
        codebook = quantizer.codebook.detach().clone()
        inputs = codebook[2].clone()
        assert int(quantizer.assign_indices(inputs.unsqueeze(0))[0]) == 2

    def test_assign_indices_does_not_update_state(self, quantizer):
        quantizer.train()
        before = quantizer.codebook.detach().clone()
        quantizer.assign_indices(torch.randn(4, DIMENSION))
        assert torch.equal(quantizer.codebook, before)
        assert float(quantizer._usage_count.sum()) == 0.0

    def test_assign_indices_zeroes_padding(self, quantizer):
        inputs = torch.randn(2, DIMENSION)
        inputs[1] = 0.0
        indices = quantizer.assign_indices(inputs)
        assert int(indices[1]) == 0

    def test_energy_distance_matches_brute_force(self, quantizer):
        inputs = torch.randn(5, DIMENSION)
        codebook = quantizer.codebook.detach()
        expected = ((inputs[:, None, :] - codebook[None, :, :]) ** 2).sum(-1).argmin(dim=1)
        assert torch.equal(quantizer.assign_indices(inputs), expected)

    def test_dead_code_reset_replaces_vectors(self):
        quantizer = VectorQuantizerEMA(
            NUM_CODES, DIMENSION, decay=0.0, dcr_enabled=True, dcr_count_threshold=1.0,
            dcr_patience_steps=1, dcr_max_resets_per_step=2,
        )
        quantizer.train()
        inputs = torch.ones(4, DIMENSION) * 7.0
        output = quantizer(inputs)
        assert 1 <= output["n_reset_codes"] <= 2
        assert output["n_dead_codes"] >= output["n_reset_codes"]

    def test_dead_code_reset_disabled(self):
        quantizer = VectorQuantizerEMA(
            NUM_CODES, DIMENSION, decay=0.0, dcr_enabled=False, dcr_count_threshold=1.0,
            dcr_patience_steps=1,
        )
        quantizer.train()
        output = quantizer(torch.ones(4, DIMENSION))
        assert output["n_reset_codes"] == 0

    def test_usage_stats_reflect_assignments(self, quantizer):
        quantizer.eval()
        codebook = quantizer.codebook.detach().clone()
        quantizer(torch.stack([codebook[0], codebook[1]]))
        statistics = quantizer.get_usage_stats()
        assert statistics["active_codes"] == 2
        assert statistics["total_codes"] == NUM_CODES
        assert statistics["usage_fraction"].sum() > 0

    def test_reset_usage_stats(self, quantizer):
        quantizer.eval()
        quantizer(torch.randn(3, DIMENSION))
        quantizer.reset_usage_stats()
        assert float(quantizer._usage_count.sum()) == 0.0

    def test_similarity_metrics_need_two_active_codes(self, quantizer):
        quantizer.eval()
        metrics = quantizer.compute_similarity_metrics()
        assert set(metrics) == {
            "cosine_mean_similarity", "cosine_min_similarity", "cosine_max_similarity",
            "euclidean_mean_distance", "euclidean_min_distance", "euclidean_max_distance",
        }
        assert all(value == 0.0 for value in metrics.values())

    def test_similarity_metrics_over_used_codes(self, quantizer):
        quantizer.eval()
        codebook = quantizer.codebook.detach().clone()
        quantizer(torch.stack([codebook[0], codebook[1], codebook[2]]))
        metrics = quantizer.compute_similarity_metrics()
        assert all(np.isfinite(value) for value in metrics.values())
        assert metrics["euclidean_min_distance"] <= metrics["euclidean_mean_distance"]


class TestAttentionHelpers:
    def test_causal_mask_semantics(self):
        mask = causal_mask(3)
        assert mask.shape == (3, 3)
        assert mask.dtype == torch.bool
        assert not mask[0, 0]
        assert mask[0, 1] and mask[1, 2]

    def test_rope_tables_shape(self):
        cosine, sine = _rope_tables(5, 4, torch.device("cpu"))
        assert cosine.shape == (5, 4)
        assert sine.shape == (5, 4)

    def test_apply_rope_preserves_norms(self):
        tensor = torch.randn(4, 1, 2, 8)
        cosine, sine = _rope_tables(4, 8, torch.device("cpu"))
        rotated = _apply_rope(tensor, cosine, sine)
        assert rotated.shape == tensor.shape
        assert torch.allclose(rotated.norm(dim=-1), tensor.norm(dim=-1), atol=1e-5)


class TestEncoders:
    def test_adaptive_encoder_blends_toward_projection(self):
        encoder = AdaptiveResidualEncoder(DIMENSION, fixed_alpha=1.0)
        activations = torch.randn(3, DIMENSION)
        transformed = encoder.layer_norm(encoder.linear(activations))
        expected = 0.5 * activations + 0.5 * transformed
        assert torch.allclose(encoder(activations), expected, atol=1e-6)

    def test_adaptive_encoder_learnable_mix_is_bounded(self):
        encoder = AdaptiveResidualEncoder(DIMENSION)
        assert encoder.is_fixed is False
        mix = torch.sigmoid(encoder.alpha) * 0.5
        assert 0.0 < float(mix) < 0.5

    def test_padding_mask_zeroes_encoder_output(self):
        encoder = PassThroughEncoder(DIMENSION)
        activations = torch.ones(2, 3, DIMENSION)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        mask[0, 1] = True
        output = encoder(activations, padding_mask=mask)
        assert torch.equal(output[0, 1], torch.zeros(DIMENSION))
        assert torch.equal(output[0, 0], torch.ones(DIMENSION))

    def test_pass_through_without_mask_is_identity(self):
        encoder = PassThroughEncoder(DIMENSION)
        activations = torch.randn(2, 3, DIMENSION)
        assert encoder(activations) is activations


def build_model(**overrides):
    arguments = {
        "num_embeddings": NUM_CODES,
        "embedding_dim": DIMENSION,
        "decoder_layers": 1,
        "nhead": 2,
        "ff_mult": 2,
        "dropout": 0.0,
    }
    arguments.update(overrides)
    return CrossLayerVQVAE(**arguments)


class TestCrossLayerVQVAE:
    @pytest.mark.parametrize("no_residual", [True, False])
    def test_forward_shapes(self, no_residual):
        model = build_model(no_residual=no_residual)
        model.eval()
        activations = torch.randn(2, 5, DIMENSION)
        output = model(activations)
        assert output["reconstructed"].shape == activations.shape
        assert output["quantized"].shape == activations.shape
        assert output["z_e"].shape == activations.shape
        assert output["indices"].shape == (2, 5)
        assert output["reconstructed"].dtype == activations.dtype

    def test_padding_rows_reconstruct_to_zero(self):
        model = build_model()
        model.eval()
        activations = torch.randn(1, 4, DIMENSION)
        activations[0, 2:] = 0.0
        output = model(activations)
        assert torch.equal(output["reconstructed"][0, 2:], torch.zeros(2, DIMENSION))
        assert torch.equal(output["z_e"][0, 2:], torch.zeros(2, DIMENSION))

    def test_encode_indices_runs_without_the_decoder(self):
        model = build_model()
        model.eval()
        indices = model.encode_indices(torch.randn(2, 5, DIMENSION))
        assert indices.shape == (2, 5)
        assert indices.dtype == torch.long
        assert int(indices.max()) < NUM_CODES

    def test_encode_indices_matches_forward(self):
        model = build_model()
        model.eval()
        activations = torch.randn(2, 5, DIMENSION)
        assert torch.equal(model.encode_indices(activations), model(activations)["indices"])

    def test_normalization_roundtrip(self):
        model = build_model(
            activation_normalization="std", read_activation_scale=2.0, target_activation_scale=4.0
        )
        activations = torch.randn(2, 3, DIMENSION)
        assert torch.allclose(model.normalize_read_activations(activations), activations / 2.0)
        assert torch.allclose(model.normalize_target_activations(activations), activations / 4.0)
        assert torch.allclose(model.denormalize_target_activations(activations), activations * 4.0)
        assert model.read_activation_scale == pytest.approx(2.0)
        assert model.target_activation_scale == pytest.approx(4.0)

    def test_normalization_none_is_identity(self):
        model = build_model()
        activations = torch.randn(2, 3, DIMENSION)
        assert model.normalize_read_activations(activations) is activations
        assert model.denormalize_target_activations(activations) is activations

    def test_activation_encoder_alias(self):
        model = build_model()
        assert model.activation_encoder is model._ContinuousEmbedding
        assert model.quantizer is model._VectorQuantizer

    def test_backward_populates_gradients(self):
        model = build_model(no_residual=True)
        model.train()
        activations = torch.randn(2, 4, DIMENSION)
        output = model(activations)
        loss = output["reconstructed"].pow(2).mean() + output["loss"]
        loss.backward()
        assert model.quantizer.codebook.grad is None  # EMA codebook is not optimized directly
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_get_codebook_usage_matches_quantizer(self):
        model = build_model()
        model.eval()
        usage = model.get_codebook_usage()
        assert usage["total_codes"] == NUM_CODES
        assert usage["active_codes"] == 0
        assert float(usage["usage_count"].sum()) == 0.0


class TestCheckpointLoading:
    def save_checkpoint(self, path, no_residual=True):
        model = build_model(no_residual=no_residual)
        checkpoint = {
            "model": model.state_dict(),
            "codebook": model.quantizer.codebook.detach().cpu(),
            "init_info": {"init": "spherical", "n_available_vectors": 10, "n_selected_vectors": 10},
            "config": {
                "decoder_layers": 1, "nhead": 2, "ff_mult": 2, "dropout": 0.0,
                "activation_normalization": "none", "read_activation_scale": 1.0,
                "target_activation_scale": 1.0,
            },
            "regions": {
                "benign_codes": [0, 1],
                "harmful_codes": [2],
                "source": "smoothed_response_enrichment",
                "score_method": "response_presence",
                "prior_strength": 10.0,
                "signed_harmfulness": [0.1, 0.2, 0.3],
            },
        }
        torch.save(checkpoint, path)
        return model

    def test_load_vq_checkpoint_roundtrip(self, tmp_path):
        source = self.save_checkpoint(tmp_path / "model.pt")
        model, checkpoint = load_vq_checkpoint(tmp_path / "model.pt", "cpu")
        assert checkpoint["regions"]["score_method"] == "response_presence"
        assert model.quantizer.num_codes == NUM_CODES
        for name, tensor in source.state_dict().items():
            assert torch.equal(model.state_dict()[name], tensor), name

    def test_load_vq_checkpoint_freezes_parameters(self, tmp_path):
        self.save_checkpoint(tmp_path / "model.pt")
        model, _ = load_vq_checkpoint(tmp_path / "model.pt", "cpu")
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())

    def test_load_steering_checkpoint_exposes_regions(self, tmp_path):
        self.save_checkpoint(tmp_path / "model.pt")
        model, num_codes = load_steering_vq_checkpoint(tmp_path / "model.pt", "cpu")
        assert num_codes == NUM_CODES
        assert model.checkpoint_config["nhead"] == 2
        assert model.checkpoint_regions["harmful_codes"] == [2]

    def test_load_steering_checkpoint_tolerates_missing_regions(self, tmp_path):
        self.save_checkpoint(tmp_path / "model.pt")
        checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu")
        del checkpoint["regions"]
        torch.save(checkpoint, tmp_path / "no_regions.pt")
        model, _ = load_steering_vq_checkpoint(tmp_path / "no_regions.pt", "cpu")
        assert model.checkpoint_regions is None
