"""Prepare model prompts and variable-length activation batches."""

from typing import Optional, TypedDict

import numpy as np
import torch
import transformers.utils


@torch.no_grad()
def initialize_internlm3_rotary_embeddings(model):
    """Initialize InternLM3 RoPE buffers from the model's own RoPE function.

    Transformers 5 leaves these nonpersistent buffers empty after model loading.
    """
    if model.config.model_type != "internlm3":
        return
    initialized = 0
    for module in model.modules():
        if module.__class__.__name__ != "InternLM3RotaryEmbedding":
            continue
        inv_freq, attention_scaling = module.rope_init_fn(
            module.config, module.inv_freq.device, **module.rope_kwargs
        )
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = inv_freq
        module.attention_scaling = attention_scaling
        module.max_seq_len_cached = module.original_max_seq_len
        initialized += 1
    if initialized == 0:
        raise ValueError("InternLM3 model has no recognized rotary embeddings; check its remote code")
    print(f"[InternLM] initialized {initialized} RoPE buffers", flush=True)


def configure_transformers_compatibility():
    """Provide the type-only API expected by the InternLM3 model code.

    InternLM3's bundled model implementation targets Transformers 4.47 and imports
    ``LossKwargs`` from ``transformers.utils``. Transformers 5 removed that symbol,
    although the model uses it only to describe optional forward arguments.
    """
    if hasattr(transformers.utils, "LossKwargs"):
        return
    loss_kwargs = TypedDict("LossKwargs", {"num_items_in_batch": Optional[int]}, total=False)
    loss_kwargs.__module__ = "transformers.utils"
    transformers.utils.LossKwargs = loss_kwargs


def format_prompt_token_ids(tokenizer, prompt):
    """Apply the chat template and open an assistant generation turn."""
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=True
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded


def pad_activation_sequences(sequences, key, device):
    """Right-pad ``[T_i, D]`` tensors into one ``[B, T_max, D]`` batch.

    Left padding would shift causal/RoPE positions and let real tokens attend across the padding.
    """
    max_length = max(sequence[key].shape[0] for sequence in sequences)
    padded = torch.zeros(len(sequences), max_length, sequences[0][key].shape[1])
    for row, sequence in enumerate(sequences):
        padded[row, :sequence[key].shape[0]] = sequence[key].float()
    return padded.to(device)


def iter_activation_batches(sequences, batch_size, shuffle, seed):
    """Yield sequential batches for evaluation or shuffled length buckets for training."""
    if not shuffle:
        for start in range(0, len(sequences), batch_size):
            yield sequences[start:start + batch_size]
        return
    by_length = sorted(range(len(sequences)), key=lambda index: sequences[index]["x"].shape[0])
    batch_starts = np.arange(0, len(by_length), batch_size)
    np.random.RandomState(seed).shuffle(batch_starts)
    for start in batch_starts:
        yield [sequences[index] for index in by_length[start:start + batch_size]]
