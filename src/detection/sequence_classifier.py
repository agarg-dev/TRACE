"""Causal sequence detectors and their training and evaluation helpers."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data.dataset_splits import train_validation_split
from model_inputs import pad_activation_sequences
from vq.codebook import smoothed_response_code_statistics


def _run_causal_gru(gru, inputs, lengths, total_length, initial_state=None):
    packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
    # Keep the original call path unchanged when no prompt state is supplied.
    packed_outputs, _ = gru(packed) if initial_state is None else gru(packed, initial_state)
    outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=total_length)
    return outputs


class CodeSequenceClassifier(nn.Module):
    """Project frozen codebook vectors and classify their sequence with a causal GRU."""

    def __init__(
        self, codebook, code_features, projection_dim=128, hidden_dim=128, num_layers=1,
        dropout=0.1, hazard_bias=-7.0, harmfulness_score_weight=0.0, prompt_conditioning=False,
    ):
        super().__init__()
        codebook = torch.as_tensor(codebook, dtype=torch.float32).detach().clone()
        code_features = torch.as_tensor(code_features, dtype=torch.float32).detach().clone()
        self.num_codes, self.codebook_dim = codebook.shape
        self.code_feature_dim = code_features.shape[1]
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.harmfulness_score_weight = float(harmfulness_score_weight)
        self.prompt_conditioning = bool(prompt_conditioning)
        self.register_buffer("codebook", codebook, persistent=False)
        self.register_buffer("code_features", code_features, persistent=False)
        self.code_projection = nn.Linear(self.codebook_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        self.input_dropout = nn.Dropout(dropout)
        if self.prompt_conditioning:
            self.prompt_projection = nn.Linear(self.codebook_dim, projection_dim)
            self.prompt_norm = nn.LayerNorm(projection_dim)
            self.prompt_attention = nn.Linear(projection_dim, 1)
            self.prompt_state_projection = nn.Linear(projection_dim, num_layers * hidden_dim)
            nn.init.zeros_(self.prompt_attention.weight)
            nn.init.zeros_(self.prompt_attention.bias)
            nn.init.xavier_uniform_(self.prompt_state_projection.weight, gain=0.1)
            nn.init.zeros_(self.prompt_state_projection.bias)
        self.gru = nn.GRU(
            projection_dim + self.code_feature_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.response_head = nn.Linear(hidden_dim, 2)
        self.hazard_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.hazard_head.weight)
        nn.init.constant_(self.hazard_head.bias, hazard_bias)

    def prompt_initial_state(self, prompt_activations, prompt_lengths):
        """Attention-pool the known prompt into the GRU state used before response token one."""
        prompt_activations = prompt_activations.to(dtype=self.prompt_projection.weight.dtype)
        projected_prompt = self.prompt_norm(self.prompt_projection(prompt_activations))
        attention_logits = self.prompt_attention(projected_prompt).squeeze(-1).float()
        positions = torch.arange(prompt_activations.shape[1], device=prompt_activations.device)
        prompt_mask = positions.unsqueeze(0) < prompt_lengths.unsqueeze(1)
        attention_weights = attention_logits.masked_fill(~prompt_mask, -torch.inf).softmax(dim=1)
        pooled_prompt = torch.sum(attention_weights.unsqueeze(-1) * projected_prompt.float(), dim=1)
        initial_state = torch.tanh(self.prompt_state_projection(pooled_prompt))
        return initial_state.view(-1, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()

    def forward(self, inputs, lengths):
        """Return per-token logits and a valid-token mask for padded code-ID sequences."""
        if self.prompt_conditioning:
            code_ids, prompt_activations, prompt_lengths = inputs
        else:
            code_ids = inputs

        # Project the codebook once, then apply independent dropout after lookup.
        projected_codebook = self.input_norm(self.code_projection(self.codebook))
        projected_sequence = self.input_dropout(projected_codebook[code_ids])
        inputs = torch.cat([projected_sequence, self.code_features[code_ids]], dim=-1)
        initial_state = (
            self.prompt_initial_state(prompt_activations, prompt_lengths)
            if self.prompt_conditioning else None
        )
        outputs = _run_causal_gru(self.gru, inputs, lengths, code_ids.shape[1], initial_state)
        response_logits = self.response_head(outputs)
        hazard_logits = self.hazard_head(outputs).squeeze(-1)
        hazard_logits = hazard_logits + self.harmfulness_score_weight * self.code_features[code_ids, 0]
        positions = torch.arange(code_ids.shape[1], device=code_ids.device)
        valid_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        return response_logits, hazard_logits, valid_mask

    def forward_step(self, code_ids, hidden_state=None):
        """Classify one new code per response and return the updated causal GRU state."""
        projected_vectors = self.input_dropout(self.input_norm(self.code_projection(self.codebook[code_ids])))
        inputs = torch.cat([projected_vectors, self.code_features[code_ids]], dim=-1).unsqueeze(1)
        outputs, hidden_state = self.gru(inputs, hidden_state)
        response_logits = self.response_head(outputs[:, 0])
        hazard_logits = self.hazard_head(outputs[:, 0]).squeeze(-1)
        hazard_logits = hazard_logits + self.harmfulness_score_weight * self.code_features[code_ids, 0]
        return response_logits, hazard_logits, hidden_state


class RawActivationSequenceClassifier(nn.Module):
    """Project read-layer activations and classify the sequence with a causal GRU."""

    def __init__(
        self, activation_dim, projection_dim=128, hidden_dim=128, num_layers=1,
        dropout=0.1, hazard_bias=-7.0,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_projection = nn.Linear(activation_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            projection_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.response_head = nn.Linear(hidden_dim, 2)
        self.hazard_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.hazard_head.weight)
        nn.init.constant_(self.hazard_head.bias, hazard_bias)

    def _project(self, activations):
        activations = activations.to(dtype=self.activation_projection.weight.dtype)
        return self.input_dropout(self.input_norm(self.activation_projection(activations)))

    def forward(self, activations, lengths):
        """Return per-token logits and a valid-token mask for padded raw activations."""
        outputs = _run_causal_gru(self.gru, self._project(activations), lengths, activations.shape[1])
        response_logits = self.response_head(outputs)
        hazard_logits = self.hazard_head(outputs).squeeze(-1)
        positions = torch.arange(activations.shape[1], device=activations.device)
        valid_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        return response_logits, hazard_logits, valid_mask

    def forward_step(self, activations, hidden_state=None):
        """Classify one raw activation per response and return the updated causal GRU state."""
        outputs, hidden_state = self.gru(self._project(activations).unsqueeze(1), hidden_state)
        response_logits = self.response_head(outputs[:, 0])
        hazard_logits = self.hazard_head(outputs[:, 0]).squeeze(-1)
        return response_logits, hazard_logits, hidden_state


class HybridSequenceClassifier(nn.Module):
    """Classify raw activations together with their frozen VQ representation."""

    def __init__(
        self, activation_dim, codebook, code_features, raw_projection_dim=256,
        vq_projection_dim=256, hidden_dim=64, num_layers=1, dropout=0.1,
        hazard_bias=-7.0, harmfulness_score_weight=1.0, prompt_conditioning=False,
    ):
        super().__init__()
        codebook = torch.as_tensor(codebook, dtype=torch.float32).detach().clone()
        code_features = torch.as_tensor(code_features, dtype=torch.float32).detach().clone()
        self.activation_dim = activation_dim
        self.num_codes, self.codebook_dim = codebook.shape
        self.code_feature_dim = code_features.shape[1]
        self.raw_projection_dim = raw_projection_dim
        self.vq_projection_dim = vq_projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.harmfulness_score_weight = float(harmfulness_score_weight)
        self.prompt_conditioning = bool(prompt_conditioning)
        self.register_buffer("codebook", codebook, persistent=False)
        self.register_buffer("code_features", code_features, persistent=False)

        self.raw_projection = nn.Linear(activation_dim, raw_projection_dim)
        self.vq_projection = nn.Linear(self.codebook_dim, vq_projection_dim)
        self.raw_norm = nn.LayerNorm(raw_projection_dim)
        self.vq_norm = nn.LayerNorm(vq_projection_dim)
        self.input_dropout = nn.Dropout(dropout)
        if self.prompt_conditioning:
            self.prompt_attention = nn.Linear(raw_projection_dim, 1)
            self.prompt_state_projection = nn.Linear(raw_projection_dim, num_layers * hidden_dim)
            nn.init.zeros_(self.prompt_attention.weight)
            nn.init.zeros_(self.prompt_attention.bias)
            nn.init.xavier_uniform_(self.prompt_state_projection.weight, gain=0.1)
            nn.init.zeros_(self.prompt_state_projection.bias)
        gru_input_dim = raw_projection_dim + vq_projection_dim + self.code_feature_dim
        self.gru = nn.GRU(
            gru_input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.response_head = nn.Linear(hidden_dim, 2)
        self.hazard_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.hazard_head.weight)
        nn.init.constant_(self.hazard_head.bias, hazard_bias)

    def _project_raw(self, activations):
        activations = activations.to(dtype=self.raw_projection.weight.dtype)
        return self.input_dropout(self.raw_norm(self.raw_projection(activations)))

    def prompt_initial_state(self, prompt_activations, prompt_lengths):
        """Attention-pool the known prompt into the GRU state used before response token one."""
        prompt_activations = prompt_activations.to(dtype=self.raw_projection.weight.dtype)
        projected_prompt = self.raw_norm(self.raw_projection(prompt_activations))
        attention_logits = self.prompt_attention(projected_prompt).squeeze(-1).float()
        positions = torch.arange(prompt_activations.shape[1], device=prompt_activations.device)
        prompt_mask = positions.unsqueeze(0) < prompt_lengths.unsqueeze(1)
        attention_weights = attention_logits.masked_fill(~prompt_mask, -torch.inf).softmax(dim=1)
        pooled_prompt = torch.sum(attention_weights.unsqueeze(-1) * projected_prompt.float(), dim=1)
        initial_state = torch.tanh(self.prompt_state_projection(pooled_prompt))
        initial_state = initial_state.view(-1, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return initial_state

    def forward(self, inputs, lengths):
        """Return per-token logits for aligned raw-activation and VQ-code sequences."""
        activations, code_ids = inputs[:2]
        raw_sequence = self._project_raw(activations)
        projected_codebook = self.vq_norm(self.vq_projection(self.codebook))
        vq_sequence = self.input_dropout(projected_codebook[code_ids])
        gru_inputs = torch.cat([raw_sequence, vq_sequence, self.code_features[code_ids]], dim=-1)
        initial_state = self.prompt_initial_state(inputs[2], inputs[3]) if self.prompt_conditioning else None
        outputs = _run_causal_gru(self.gru, gru_inputs, lengths, code_ids.shape[1], initial_state)
        response_logits = self.response_head(outputs)
        hazard_logits = self.hazard_head(outputs).squeeze(-1)
        hazard_logits = hazard_logits + self.harmfulness_score_weight * self.code_features[code_ids, 0]
        positions = torch.arange(code_ids.shape[1], device=code_ids.device)
        valid_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        return response_logits, hazard_logits, valid_mask

    def forward_step(self, activations, code_ids, hidden_state=None):
        """Classify one aligned raw activation and VQ code per response."""
        raw_vectors = self._project_raw(activations)
        vq_vectors = self.input_dropout(self.vq_norm(self.vq_projection(self.codebook[code_ids])))
        inputs = torch.cat([raw_vectors, vq_vectors, self.code_features[code_ids]], dim=-1).unsqueeze(1)
        outputs, hidden_state = self.gru(inputs, hidden_state)
        response_logits = self.response_head(outputs[:, 0])
        hazard_logits = self.hazard_head(outputs[:, 0]).squeeze(-1)
        hazard_logits = hazard_logits + self.harmfulness_score_weight * self.code_features[code_ids, 0]
        return response_logits, hazard_logits, hidden_state


def final_token_logits(logits, lengths):
    """Select the classifier output after the final real token of each response."""
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, lengths - 1]


def cumulative_hazard_probabilities(hazard_logits, valid_mask):
    """Convert conditional per-token hazards into the risk that harm has appeared by each token."""
    log_survival = F.logsigmoid(-hazard_logits.float()).masked_fill(~valid_mask, 0.0)
    return -torch.expm1(log_survival.cumsum(dim=1))


def streaming_classification_loss(
    response_logits,
    hazard_logits,
    valid_mask,
    lengths,
    labels,
    class_weights=None,
    response_weight=1.0,
    streaming_weight=1.0,
):
    """Train final-response classification and cumulative streaming risk from response labels.

    Each hazard is conditional on harm not having appeared earlier. Their cumulative survival product
    gives the probability that harm has appeared by the end of the response, without choosing a fixed
    temporal window.
    """
    final_logits = final_token_logits(response_logits, lengths)
    response_loss = F.cross_entropy(final_logits, labels, weight=class_weights)
    log_survival_terms = F.logsigmoid(-hazard_logits.float()).masked_fill(~valid_mask, 0.0)
    log_survival = log_survival_terms.sum(dim=1)
    harmful_risk = (-torch.expm1(log_survival)).clamp_min(1e-12)
    sequence_losses = torch.where(labels.bool(), -torch.log(harmful_risk), -log_survival)
    if class_weights is None:
        sequence_weights = torch.ones_like(sequence_losses)
    else:
        sequence_weights = class_weights[labels]
    streaming_loss = (sequence_losses * sequence_weights).sum() / sequence_weights.sum()

    total = response_weight * response_loss + streaming_weight * streaming_loss
    return {"loss": total, "response": response_loss, "streaming": streaming_loss}


@torch.no_grad()
def assign_code_sequences(vq_model, activation_sequences, device, batch_size=16, include_token_ids=False):
    """Replace response activations with deterministic nearest-code sequences."""
    code_sequences = []
    for start in range(0, len(activation_sequences), batch_size):
        batch = activation_sequences[start:start + batch_size]
        padded_activations = pad_activation_sequences(batch, "x", device)
        batch_codes = vq_model.encode_indices(padded_activations)
        for row, sequence in enumerate(batch):
            length = sequence["x"].shape[0]
            assigned = {
                "idx": sequence["idx"],
                "label": sequence["label"],
                "codes": batch_codes[row, :length].cpu().long(),
            }
            if include_token_ids:
                assigned["token_ids"] = list(sequence.get("token_ids", []))
            code_sequences.append(assigned)
    return code_sequences


@torch.no_grad()
def assign_hybrid_sequences(vq_model, activation_sequences, device, batch_size=16, include_token_ids=False):
    """Attach deterministic VQ codes while retaining each raw activation sequence."""
    hybrid_sequences = []
    for start in range(0, len(activation_sequences), batch_size):
        batch = activation_sequences[start:start + batch_size]
        padded_activations = pad_activation_sequences(batch, "x", device)
        batch_codes = vq_model.encode_indices(padded_activations)
        for row, sequence in enumerate(batch):
            length = sequence["x"].shape[0]
            assigned = {
                "idx": sequence["idx"],
                "label": sequence["label"],
                "x": sequence["x"],
                "codes": batch_codes[row, :length].cpu().long(),
            }
            if include_token_ids:
                assigned["token_ids"] = list(sequence.get("token_ids", []))
            hybrid_sequences.append(assigned)
    return hybrid_sequences


def activation_mean_code_vectors(sequences, fallback_vectors):
    """Average training activations assigned to each code, falling back for unused codes."""
    fallback_vectors = torch.as_tensor(fallback_vectors, dtype=torch.float32).detach().cpu()
    num_codes, activation_dim = fallback_vectors.shape
    vector_sums = torch.zeros(num_codes, activation_dim, dtype=torch.float32)
    token_counts = torch.zeros(num_codes, dtype=torch.long)
    for sequence in sequences:
        activations = sequence["x"]
        codes = sequence["codes"].detach().cpu().long()
        vector_sums.index_add_(0, codes, activations.float())
        token_counts.index_add_(0, codes, torch.ones(len(codes), dtype=torch.long))

    code_vectors = vector_sums / token_counts.clamp(min=1).unsqueeze(1)
    unused_codes = token_counts == 0
    code_vectors[unused_codes] = fallback_vectors[unused_codes]
    return code_vectors, token_counts


def attach_prompt_activations(response_sequences, prompt_sequences):
    """Attach one checked prompt-activation sequence to every response sequence."""
    prompts_by_id = {prompt["idx"]: prompt for prompt in prompt_sequences}
    attached = []
    for sequence in response_sequences:
        prompt = prompts_by_id[sequence["idx"]]
        if prompt["label"] != sequence["label"]:
            raise ValueError(f"prompt and response labels differ for response {sequence['idx']}")
        combined = dict(sequence)
        combined["prompt_activations"] = prompt["x"]
        attached.append(combined)
    return attached


def sequence_length(sequence):
    """Return the token count for either a VQ-code or raw-activation sequence."""
    if "codes" in sequence:
        return len(sequence["codes"])
    return len(sequence["x"])


def code_feature_statistics(sequences, num_codes, prior_strength=10.0):
    """Estimate signed code harmfulness from response-level presence in the classifier training split.

    A repeated code contributes once per response when its global association is estimated. Its repeated
    occurrences remain in the code sequence seen by the GRU. An empirical-Bayes prior prevents codes seen
    in only a few responses from receiving extreme scores.
    """
    statistics = smoothed_response_code_statistics(
        (sequence["codes"].cpu().numpy() for sequence in sequences),
        (sequence["label"] for sequence in sequences), num_codes, prior_strength,
    )
    code_features = torch.tensor(
        np.stack([
            statistics["signed_harmfulness"], statistics["normalized_log_response_support"]
        ], axis=1), dtype=torch.float32
    )
    serialized_statistics = {
        "feature_names": ["signed_response_harmfulness", "normalized_log_response_support"],
        "source": "classifier training split only",
        "score_method": "smoothed response-presence enrichment relative to the harmful-response base rate",
        "prior_strength": statistics["prior_strength"],
        "base_harmful_response_rate": statistics["base_harmful_response_rate"],
        "n_training_responses": statistics["n_responses"],
        "n_harmful_responses": statistics["n_harmful_responses"],
        "n_safe_responses": statistics["n_safe_responses"],
        "n_unseen_codes": int((statistics["response_counts"] == 0).sum()),
        "response_counts": torch.from_numpy(statistics["response_counts"]),
        "harmful_response_counts": torch.from_numpy(statistics["harmful_response_counts"]),
        "smoothed_harmful_probability": torch.from_numpy(
            statistics["smoothed_harmful_probability"].astype(np.float32)
        ),
        "signed_harmfulness": torch.from_numpy(statistics["signed_harmfulness"].astype(np.float32)),
    }
    return code_features, serialized_statistics


def rescore_no_task_checkpoint_by_response_presence(sequences, num_codes, checkpoint_config,
                                                    prior_strength=10.0):
    """Derive presence scores from the exact VQ-training partition of a no-task checkpoint."""
    if float(checkpoint_config.get("task_weight", -1)) != 0:
        raise ValueError("response-presence rescoring is restricted to checkpoints trained without task loss")

    seed = int(checkpoint_config.get("seed", 42))
    detector_training, _ = train_validation_split(sequences, seed)
    vq_training, _ = train_validation_split(detector_training, seed)

    statistics = smoothed_response_code_statistics(
        (sequence["codes"].cpu().numpy() for sequence in vq_training),
        (sequence["label"] for sequence in vq_training), num_codes, prior_strength,
    )
    signed_harmfulness = statistics["signed_harmfulness"].astype(np.float32)
    harmful_codes = np.flatnonzero(signed_harmfulness > 0).tolist()
    harmful_code_set = set(harmful_codes)
    return {
        "benign_codes": [code for code in range(num_codes) if code not in harmful_code_set],
        "harmful_codes": harmful_codes,
        "source": "recomputed from the VQ training split",
        "score_source": "VQ training split",
        "score_method": "response_presence",
        "prior_strength": float(prior_strength),
        "signed_harmfulness": signed_harmfulness,
        "n_score_responses": len(vq_training),
    }


def checkpoint_code_feature_statistics(
    checkpoint_regions, sequences, num_codes, expected_prior_strength=10.0,
):
    """Build classifier features from the harmfulness score stored in a VQ checkpoint.

    Harmfulness is never re-estimated from classifier labels. The second feature is unlabeled response
    support in the classifier training split, which tells the GRU how frequently a code is observed without
    changing what that code's harmfulness score means.
    """
    score_method = checkpoint_regions.get("score_method")
    if score_method not in {"response_presence", "response_frequency"}:
        raise ValueError(
            "the classifier requires a VQ checkpoint with response-based code scores; "
            f"found {score_method!r}"
        )
    prior_strength = float(checkpoint_regions.get("prior_strength", float("nan")))
    if expected_prior_strength is not None and not np.isclose(prior_strength, expected_prior_strength):
        raise ValueError(
            f"the VQ checkpoint uses prior strength {prior_strength:g}, but "
            f"{expected_prior_strength:g} was requested"
        )

    signed_harmfulness = np.asarray(checkpoint_regions.get("signed_harmfulness"), dtype=np.float32)
    response_counts = np.zeros(num_codes, dtype=np.int64)
    for sequence in sequences:
        codes = np.asarray(sequence["codes"].cpu(), dtype=np.int64).reshape(-1)
        response_counts[np.unique(codes)] += 1
    max_count = max(1, int(response_counts.max()))
    normalized_support = np.log1p(response_counts) / np.log1p(max_count)
    code_features = torch.tensor(
        np.stack([signed_harmfulness, normalized_support], axis=1), dtype=torch.float32
    )
    serialized_statistics = {
        "feature_names": ["checkpoint_signed_response_harmfulness", "normalized_log_response_support"],
        "score_source": checkpoint_regions.get("score_source", "VQ checkpoint"),
        "support_source": "classifier training split without labels",
        "score_method": score_method,
        "prior_strength": prior_strength,
        "n_support_responses": len(sequences),
        "n_unseen_codes": int((response_counts == 0).sum()),
        "response_counts": torch.from_numpy(response_counts),
        "signed_harmfulness": torch.from_numpy(signed_harmfulness),
    }
    return code_features, serialized_statistics


def sequence_batches(sequences, batch_size, shuffle, seed):
    """Yield length-bucketed batches to minimize padding while preserving reproducibility."""
    ordered = sorted(sequences, key=sequence_length)
    batches = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
    if shuffle:
        np.random.RandomState(seed).shuffle(batches)
    yield from batches


def pad_code_batch(sequences, device):
    """Pad code IDs and return tensors plus the untouched sequence metadata."""
    lengths = torch.tensor(
        [len(sequence["codes"]) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    max_length = int(lengths.max().item())
    code_ids = torch.zeros(len(sequences), max_length, dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        code_ids[row, :len(sequence["codes"])] = sequence["codes"].to(device)
    labels = torch.tensor([sequence["label"] for sequence in sequences], dtype=torch.long, device=device)
    return code_ids, lengths, labels


def pad_activation_batch(sequences, device):
    """Pad cached activations without changing their stored bf16 precision."""
    lengths = torch.tensor([len(sequence["x"]) for sequence in sequences], dtype=torch.long, device=device)
    max_length = int(lengths.max().item())
    activation_dim = sequences[0]["x"].shape[1]
    activations = torch.zeros(
        len(sequences), max_length, activation_dim, dtype=sequences[0]["x"].dtype
    )
    for row, sequence in enumerate(sequences):
        activations[row, :len(sequence["x"])] = sequence["x"]
    labels = torch.tensor([sequence["label"] for sequence in sequences], dtype=torch.long, device=device)
    return activations.to(device), lengths, labels


def pad_hybrid_batch(sequences, device):
    """Pad aligned raw activations and VQ code IDs for the hybrid classifier."""
    activations, lengths, labels = pad_activation_batch(sequences, device)
    code_ids = torch.zeros(len(sequences), activations.shape[1], dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        code_ids[row, :len(sequence["codes"])] = sequence["codes"].to(device)
    return (activations, code_ids), lengths, labels


def pad_prompt_conditioned_code_batch(sequences, device):
    """Pad VQ response codes and their separately cached prompt activations."""
    code_ids, lengths, labels = pad_code_batch(sequences, device)
    prompt_lengths = torch.tensor(
        [len(sequence["prompt_activations"]) for sequence in sequences], dtype=torch.long, device=device
    )
    max_prompt_length = int(prompt_lengths.max().item())
    prompt_dim = sequences[0]["prompt_activations"].shape[1]
    prompt_activations = torch.zeros(
        len(sequences), max_prompt_length, prompt_dim, dtype=sequences[0]["prompt_activations"].dtype
    )
    for row, sequence in enumerate(sequences):
        prompt = sequence["prompt_activations"]
        prompt_activations[row, :len(prompt)] = prompt
    return (code_ids, prompt_activations.to(device), prompt_lengths), lengths, labels


def pad_prompt_conditioned_hybrid_batch(sequences, device):
    """Pad response and prompt activations independently for a prompt-conditioned hybrid batch."""
    (activations, code_ids), lengths, labels = pad_hybrid_batch(sequences, device)
    prompt_lengths = torch.tensor(
        [len(sequence["prompt_activations"]) for sequence in sequences], dtype=torch.long, device=device
    )
    max_prompt_length = int(prompt_lengths.max().item())
    prompt_dim = sequences[0]["prompt_activations"].shape[1]
    prompt_activations = torch.zeros(
        len(sequences), max_prompt_length, prompt_dim, dtype=sequences[0]["prompt_activations"].dtype
    )
    for row, sequence in enumerate(sequences):
        prompt = sequence["prompt_activations"]
        prompt_activations[row, :len(prompt)] = prompt
    return (activations, code_ids, prompt_activations.to(device), prompt_lengths), lengths, labels


def metrics_at_threshold(labels, scores, threshold):
    """Binary response metrics using one frozen operating threshold."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(1, len(labels))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def maximum_f1(labels, scores):
    """Return the best attainable F1 over all thresholds without a quadratic threshold sweep."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ordered_labels, ordered_scores = labels[order], scores[order]
    group_ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    true_positives = np.cumsum(ordered_labels)[group_ends]
    predicted_positives = group_ends + 1
    denominator = predicted_positives + int(labels.sum())
    f1_scores = np.divide(
        2 * true_positives, denominator,
        out=np.zeros_like(true_positives, dtype=np.float64), where=denominator != 0,
    )
    return float(f1_scores.max(initial=0.0))


def best_f1_threshold(labels, scores):
    """Select the validation threshold with maximum F1, preferring lower FPR on ties."""
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    records = [metrics_at_threshold(labels, scores, threshold) for threshold in candidates]
    best = max(records, key=lambda record: (record["f1"], -record["fpr"], record["threshold"]))
    return best["threshold"]


def threshold_at_fpr(labels, scores, target_fpr):
    """Choose the validation threshold with greatest recall while keeping FPR at or below the target."""
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    records = [metrics_at_threshold(labels, scores, threshold) for threshold in candidates]
    eligible = [record for record in records if record["fpr"] <= target_fpr + 1e-12]
    best = max(eligible, key=lambda record: (record["recall"], -record["fpr"], -record["threshold"]))
    return best["threshold"]


def select_thresholds(labels, scores):
    """Select all operating points from validation data only."""
    return {
        "argmax": 0.5,
        "best_f1": best_f1_threshold(labels, scores),
        "fpr05": threshold_at_fpr(labels, scores, 0.05),
        "fpr10": threshold_at_fpr(labels, scores, 0.10),
    }


def summarize_scores(labels, scores, thresholds):
    """Threshold-free ranking metrics plus metrics at named frozen thresholds."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "safe_score_mean": float(scores[labels == 0].mean()),
        "harmful_score_mean": float(scores[labels == 1].mean()),
        "operating_points": {
            name: metrics_at_threshold(labels, scores, threshold)
            for name, threshold in thresholds.items()
        },
    }


@torch.no_grad()
def _score_sequences(model, sequences, batch_size, device, pad_batch, include_token_details=False):
    """Score either code-ID or raw-activation sequences with the shared classifier outputs."""
    model.eval()
    labels, response_scores, streaming_scores, token_scores = [], [], [], []
    conditional_hazards, response_token_scores = [], []
    ordered_sequences = []
    for batch in sequence_batches(sequences, batch_size, shuffle=False, seed=0):
        inputs, lengths, batch_labels = pad_batch(batch, device)
        response_logits, hazard_logits, valid_mask = model(inputs, lengths)
        cumulative_risk = cumulative_hazard_probabilities(hazard_logits, valid_mask)
        final_probabilities = final_token_logits(response_logits, lengths).softmax(dim=-1)[:, 1]
        for row, sequence in enumerate(batch):
            valid_cumulative_risk = cumulative_risk[row, valid_mask[row]].cpu().numpy()
            labels.append(int(batch_labels[row].item()))
            response_scores.append(float(final_probabilities[row].item()))
            streaming_scores.append(float(valid_cumulative_risk[-1]))
            token_scores.append(valid_cumulative_risk)
            if include_token_details:
                valid_positions = valid_mask[row]
                conditional_hazards.append(
                    hazard_logits[row, valid_positions].float().sigmoid().cpu().numpy()
                )
                response_token_scores.append(
                    response_logits[row, valid_positions].softmax(dim=-1)[:, 1].float().cpu().numpy()
                )
            ordered_sequences.append(sequence)
    scored = {
        "labels": np.array(labels, dtype=np.int8),
        "response_scores": np.array(response_scores),
        "streaming_scores": np.array(streaming_scores),
        "token_scores": token_scores,
        "sequences": ordered_sequences,
    }
    if include_token_details:
        scored["conditional_hazards"] = conditional_hazards
        scored["response_token_scores"] = response_token_scores
    return scored


def score_code_sequences(model, sequences, batch_size, device, include_token_details=False):
    """Score VQ code-ID sequences."""
    pad_batch = pad_prompt_conditioned_code_batch if model.prompt_conditioning else pad_code_batch
    return _score_sequences(
        model, sequences, batch_size, device, pad_batch,
        include_token_details=include_token_details,
    )


def score_activation_sequences(model, sequences, batch_size, device, include_token_details=False):
    """Score raw layer-24 activation sequences."""
    return _score_sequences(
        model, sequences, batch_size, device, pad_activation_batch,
        include_token_details=include_token_details,
    )


def score_hybrid_sequences(model, sequences, batch_size, device, include_token_details=False):
    """Score aligned raw-activation and VQ-code sequences."""
    return _score_sequences(
        model, sequences, batch_size, device, pad_hybrid_batch,
        include_token_details=include_token_details,
    )


def score_prompt_conditioned_hybrid_sequences(
    model, sequences, batch_size, device, include_token_details=False,
):
    """Score hybrid response sequences initialized from their prompt activations."""
    return _score_sequences(
        model, sequences, batch_size, device, pad_prompt_conditioned_hybrid_batch,
        include_token_details=include_token_details,
    )


def evaluate_loss(model, sequences, args, class_weights, device, pad_batch):
    """Average classifier losses by response weight across the evaluation partition."""
    model.eval()
    totals = {"loss": 0.0, "response": 0.0, "streaming": 0.0}
    total_weight = 0.0
    with torch.no_grad():
        for batch in sequence_batches(sequences, args.batch_size, shuffle=False, seed=0):
            inputs, lengths, labels = pad_batch(batch, device)
            response_logits, hazard_logits, valid_mask = model(inputs, lengths)
            losses = streaming_classification_loss(
                response_logits, hazard_logits, valid_mask, lengths, labels, class_weights,
                args.response_weight, args.streaming_weight,
            )
            # Both heads return a mean normalized by this batch's response-class weights.
            if class_weights is None:
                batch_weight = len(batch)
            else:
                batch_weight = float(class_weights[labels].sum().item())
            for name in totals:
                totals[name] += float(losses[name].item()) * batch_weight
            total_weight += batch_weight
    return {name: value / total_weight if total_weight else 0.0 for name, value in totals.items()}


def threshold_report(scored, thresholds):
    """Summarize response and streaming scores at frozen operating thresholds."""
    return {
        "n_responses": int(len(scored["labels"])),
        "n_safe": int((scored["labels"] == 0).sum()),
        "n_harmful": int((scored["labels"] == 1).sum()),
        "response": summarize_scores(scored["labels"], scored["response_scores"], thresholds["response"]),
        "streaming": summarize_scores(scored["labels"], scored["streaming_scores"], thresholds["streaming"]),
    }


def initial_hazard_bias(sequences):
    """Set a token hazard whose mean-length cumulative risk matches the response base rate."""
    response_rate = float(np.mean([sequence["label"] for sequence in sequences]))
    mean_length = float(np.mean([sequence_length(sequence) for sequence in sequences]))
    token_hazard = 1 - (1 - response_rate) ** (1 / mean_length)
    token_hazard = float(np.clip(token_hazard, 1e-7, 1 - 1e-7))
    return float(np.log(token_hazard / (1 - token_hazard)))
