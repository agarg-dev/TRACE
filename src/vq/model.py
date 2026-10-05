from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class VectorQuantizerEMA(nn.Module):
    """EMA-updated VQ codebook with top-k sampling and dead-code resets."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1,
                 perplexity_weight=0.01, decay=0.99, epsilon=1e-5,
                 top_k=10, temperature=1.0,
                 dcr_enabled=True, dcr_count_threshold=1.0,
                 dcr_patience_steps=100, dcr_max_resets_per_step=5):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._perplexity_weight = perplexity_weight
        self._top_k = min(top_k, num_embeddings)
        self._temperature = temperature
        self.dcr_enabled = dcr_enabled
        self.dcr_count_threshold = dcr_count_threshold
        self.dcr_patience_steps = dcr_patience_steps
        self.dcr_max_resets_per_step = dcr_max_resets_per_step

        # Registered names are frozen because they are stored in existing checkpoints.
        self.register_buffer("_decay", torch.tensor(float(decay)))
        self.register_buffer("_epsilon", torch.tensor(float(epsilon)))
        self.register_buffer("_ema_cluster_size", torch.ones(num_embeddings))
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.requires_grad_(False)
        self.register_buffer("_ema_w", self._embedding.weight.data.clone())
        self.register_buffer("_usage_count", torch.zeros(num_embeddings))
        self.register_buffer("_dead_steps", torch.zeros(num_embeddings, dtype=torch.long))

    @property
    def num_codes(self):
        return self._num_embeddings

    @property
    def code_dimension(self):
        return self._embedding_dim

    @property
    def codebook(self):
        return self._embedding.weight

    def get_usage_stats(self):
        counts = self._usage_count.detach().cpu()
        total = counts.sum().item()
        fractions = counts / total if total > 0 else torch.zeros_like(counts)
        active_codes = int((counts > 0).sum().item())
        return {
            "usage_count": counts, "usage_fraction": fractions,
            "active_codes": active_codes, "total_codes": self.num_codes,
            "code_utilization": active_codes / self.num_codes,
        }

    def reset_usage_stats(self):
        self._usage_count.zero_()

    def sample_from_distances(self, distances):
        nearest_distances, nearest_codes = torch.topk(distances, k=self._top_k, dim=1, largest=False)
        probabilities = F.softmax(-nearest_distances / self._temperature, dim=1)
        sampled_column = torch.multinomial(probabilities, 1)
        codes = nearest_codes.gather(1, sampled_column).squeeze(1)
        distances = nearest_distances.gather(1, sampled_column).squeeze(1)
        return codes, distances

    def _reset_dead_codes(self, valid_inputs):
        dead = self._ema_cluster_size < self.dcr_count_threshold
        self._dead_steps[dead] += 1
        self._dead_steps[~dead] = 0
        eligible = torch.where(self._dead_steps >= self.dcr_patience_steps)[0]
        num_resets = min(int(eligible.numel()), self.dcr_max_resets_per_step)
        if num_resets == 0 or valid_inputs.shape[0] == 0:
            return 0

        reset_codes = eligible[:num_resets]
        sampled_rows = torch.randint(0, valid_inputs.shape[0], (num_resets,), device=valid_inputs.device)
        replacement_vectors = valid_inputs[sampled_rows].detach()
        self.codebook.data[reset_codes] = replacement_vectors
        self._ema_w[reset_codes] = replacement_vectors
        self._ema_cluster_size[reset_codes] = 1.0
        self._dead_steps[reset_codes] = 0
        return num_resets

    def forward(self, inputs):
        inputs = inputs.to(self.codebook.device).contiguous()
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.code_dimension)
        valid_mask = torch.norm(flat_inputs, dim=1) > 1e-6
        valid_inputs = flat_inputs[valid_mask]

        # One immutable codebook geometry is used for the entire batch. The EMA update prepares the
        # codebook for the next batch without changing the vectors associated with current assignments.
        codebook = self.codebook.detach().clone()
        distances = (
            valid_inputs.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1) - 2 * valid_inputs @ codebook.t()
        )
        if self.training:
            code_indices, minimum_distances = self.sample_from_distances(distances)
        else:
            minimum_distances, code_indices = distances.min(dim=1)

        assignments = F.one_hot(code_indices, self.num_codes).float()
        self._usage_count += assignments.sum(dim=0)
        quantized_valid = F.embedding(code_indices, codebook)
        commitment_mse = F.mse_loss(valid_inputs, quantized_valid, reduction="mean")

        num_resets = 0
        if self.training:
            with torch.no_grad():
                batch_counts = assignments.sum(dim=0)
                self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * batch_counts
                total_count = self._ema_cluster_size.data.sum()
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (total_count + self.num_codes * self._epsilon) * total_count
                )
                weighted_sum = assignments.t() @ valid_inputs
                self._ema_w = self._ema_w * self._decay + (1 - self._decay) * weighted_sum
                self.codebook.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
                if self.dcr_enabled:
                    num_resets = self._reset_dead_codes(valid_inputs)

        quantized_flat = torch.zeros_like(flat_inputs)
        flat_indices = torch.zeros(flat_inputs.shape[0], dtype=torch.long, device=flat_inputs.device)
        flat_min_distances = torch.zeros(flat_inputs.shape[0], device=flat_inputs.device)
        quantized_flat[valid_mask] = quantized_valid
        flat_indices[valid_mask] = code_indices
        flat_min_distances[valid_mask] = minimum_distances
        quantized = quantized_flat.view(input_shape)
        indices = flat_indices.view(input_shape[:-1])
        minimum_distances = flat_min_distances.view(input_shape[:-1])

        straight_through = inputs + (quantized - inputs).detach()
        if self.training:
            code_probabilities = F.softmax(-distances / 0.2, dim=1).mean(dim=0)
        else:
            code_probabilities = assignments.mean(dim=0)
        perplexity = torch.exp(-torch.sum(code_probabilities * torch.log(code_probabilities + 1e-10)))
        commitment_loss = self._commitment_cost * commitment_mse
        perplexity_loss = self._perplexity_weight * torch.log(self.num_codes / (perplexity + 1e-10))

        return {
            "quantized": straight_through, "loss": commitment_loss + perplexity_loss,
            "commit_loss": commitment_loss, "encoding_indices": flat_indices,
            "indices": indices, "min_distances": minimum_distances,
            "codebook_snapshot": codebook,
            "perplexity": perplexity, "perplexity_loss": perplexity_loss,
            "n_dead_codes": int((self._ema_cluster_size < self.dcr_count_threshold).sum().item()),
            "n_reset_codes": num_resets,
        }

    @torch.no_grad()
    def assign_indices(self, inputs):
        """Assign nearest codes without sampling, usage accounting, EMA updates, or dead-code resets."""
        inputs = inputs.to(self.codebook.device).contiguous()
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.code_dimension)
        valid_mask = torch.norm(flat_inputs, dim=1) > 1e-6
        valid_inputs = flat_inputs[valid_mask]
        codebook = self.codebook.detach()
        distances = (
            valid_inputs.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1) - 2 * valid_inputs @ codebook.t()
        )
        valid_indices = distances.argmin(dim=1)
        flat_indices = torch.zeros(flat_inputs.shape[0], dtype=torch.long, device=flat_inputs.device)
        flat_indices[valid_mask] = valid_indices
        return flat_indices.view(input_shape[:-1])


class AdaptiveResidualEncoder(nn.Module):
    """Blend x with a learned normalized projection: (1-mix)x + mix*LN(Wx+b)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, activations, padding_mask=None):
        mix = torch.sigmoid(self.alpha) * 0.5
        transformed = self.layer_norm(self.linear(activations))
        encoded = (1 - mix) * activations + mix * transformed
        if padding_mask is not None:
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return encoded


def _rope_tables(sequence_length, head_dim, device, base=10000.0):
    frequencies = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inverse_frequency = 1.0 / (base ** frequencies)
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def _apply_rope(tensor, cosine, sine):
    first_half, second_half = tensor.chunk(2, dim=-1)
    rotated = torch.cat([-second_half, first_half], dim=-1)
    return tensor * cosine[:, None, None, :] + rotated * sine[:, None, None, :]


class RoPEAttention(nn.Module):
    """Bias-free causal self-attention with rotary position embeddings."""

    def __init__(self, model_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, hidden_states):
        sequence_length, batch_size, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(sequence_length, batch_size, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        cosine, sine = _rope_tables(sequence_length, self.head_dim, hidden_states.device)
        query = _apply_rope(query, cosine.to(hidden_states.dtype), sine.to(hidden_states.dtype))
        key = _apply_rope(key, cosine.to(hidden_states.dtype), sine.to(hidden_states.dtype))
        query, key, value = (tensor.permute(1, 2, 0, 3) for tensor in (query, key, value))
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.permute(2, 0, 1, 3).reshape(sequence_length, batch_size, -1)
        return self.out(attended)


class SwiGLU(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, model_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, hidden_states):
        return self.down(self.drop(F.silu(self.gate(hidden_states)) * self.up(hidden_states)))


class ModernDecoderLayer(nn.Module):
    """Pre-RMSNorm causal attention followed by a SwiGLU feed-forward block."""

    def __init__(self, model_dim, num_heads, feedforward_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(model_dim)
        self.attn = RoPEAttention(model_dim, num_heads, dropout)
        self.norm2 = nn.RMSNorm(model_dim)
        self.ffn = SwiGLU(model_dim, feedforward_dim, dropout)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ffn(self.norm2(hidden_states))


class CausalSelfAttnHead(nn.Module):
    """Causal decoder over quantized codes, with no final norm before regression."""

    def __init__(self, model_dim, output_dim, num_heads=8, num_layers=4, feedforward_dim=2048,
                 dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ModernDecoderLayer(model_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, source):
        hidden_states = source
        for layer in self.layers:
            hidden_states = (
                checkpoint(layer, hidden_states, use_reentrant=False)
                if self.training else layer(hidden_states)
            )
        return self.output_projection(hidden_states)


class CrossLayerVQVAE(nn.Module):
    """Quantize one residual layer and causally reconstruct a later layer.

    The unusual registered names are retained because they are part of existing
    checkpoint state-dict keys. Use the readable properties elsewhere in the code.
    """

    def __init__(self, num_embeddings, embedding_dim, decoder_layers=4,
                 perplexity_weight=0.01, top_k=10, temperature=1.0,
                 commitment_cost=0.1, nhead=8, ff_mult=3, dropout=0.0,
                 dcr_enabled=True, dcr_count_threshold=1.0,
                 dcr_patience_steps=100, dcr_max_resets_per_step=5):
        super().__init__()
        self._ContinuousEmbedding = AdaptiveResidualEncoder(embedding_dim)
        self._VectorQuantizer = VectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost=commitment_cost,
            perplexity_weight=perplexity_weight, top_k=top_k,
            temperature=temperature, dcr_enabled=dcr_enabled,
            dcr_count_threshold=dcr_count_threshold, dcr_patience_steps=dcr_patience_steps,
            dcr_max_resets_per_step=dcr_max_resets_per_step
        )
        feedforward_dim = int(ff_mult * embedding_dim)
        self._encoder = CausalSelfAttnHead(
            embedding_dim, embedding_dim, nhead, decoder_layers, feedforward_dim, dropout
        )

    @property
    def activation_encoder(self):
        return self._ContinuousEmbedding

    @property
    def quantizer(self):
        return self._VectorQuantizer

    def encode_activations(self, activations, padding_mask=None):
        return self.activation_encoder(activations, padding_mask=padding_mask)

    def forward(self, activations):
        activations = activations.contiguous()
        padding_mask = torch.norm(activations, dim=2) <= 1e-6
        encoded = self.encode_activations(activations, padding_mask=padding_mask)

        device_type = "cuda" if activations.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            vq_output = self.quantizer(encoded.float())
        quantized = vq_output["quantized"].to(encoded.dtype)

        quantized_sequence = quantized.transpose(0, 1)
        reconstructed = self._encoder(quantized_sequence).transpose(0, 1)
        reconstructed = reconstructed.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return {
            "loss": vq_output["loss"], "z_e": encoded, "reconstructed": reconstructed,
            "quantized": quantized, "encoding_indices": vq_output["encoding_indices"],
            "indices": vq_output["indices"], "min_distances": vq_output["min_distances"],
            "codebook_snapshot": vq_output["codebook_snapshot"],
            "perplexity": vq_output["perplexity"], "commit_loss": vq_output["commit_loss"],
            "perplexity_loss": vq_output["perplexity_loss"],
            "n_dead_codes": vq_output["n_dead_codes"], "n_reset_codes": vq_output["n_reset_codes"],
        }

    @torch.no_grad()
    def encode_indices(self, activations):
        """Assign codes without running the reconstruction decoder."""
        activations = activations.contiguous()
        padding_mask = torch.norm(activations, dim=2) <= 1e-6
        encoded = self.encode_activations(activations, padding_mask=padding_mask)
        device_type = "cuda" if activations.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return self.quantizer.assign_indices(encoded.float())

    def get_codebook_usage(self):
        return self.quantizer.get_usage_stats()


def load_vq_checkpoint(checkpoint_path, device, freeze=True):
    """Strictly load a VQ state dict and return its model and checkpoint metadata."""
    checkpoint_path = Path(checkpoint_path)
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_codes, activation_dim = saved_checkpoint["codebook"].shape
    config = saved_checkpoint["config"]
    model = CrossLayerVQVAE(
        num_codes,
        activation_dim,
        decoder_layers=config["decoder_layers"],
        nhead=config.get("nhead", 8),
        ff_mult=config.get("ff_mult", 3),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(saved_checkpoint["model"], strict=True)
    if freeze:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, saved_checkpoint


def load_steering_vq_checkpoint(checkpoint_path, device):
    """Load a VQ checkpoint with the metadata attributes used during steering."""
    model, saved_checkpoint = load_vq_checkpoint(checkpoint_path, device, freeze=False)
    model.eval()
    model.checkpoint_config = saved_checkpoint["config"]
    model.checkpoint_regions = saved_checkpoint.get("regions")
    return model, int(saved_checkpoint["codebook"].shape[0])
