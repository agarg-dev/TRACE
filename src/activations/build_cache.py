#!/usr/bin/env python
"""Cache prompt or response activations for training and evaluation."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.dataset_splits import DEFAULT_DATASET, dataset_split
from project_config import ACTIVATION_CACHE_DIR, DEFAULT_BASE_MODEL, MODEL_DIR
from project_config import READ_LAYER, TARGET_LAYER
from model_inputs import (
    restore_transformers_loss_kwargs, format_prompt_token_ids,
    repair_internlm3_rotary_embeddings,
)

TRANSCODER_LAYERS = [READ_LAYER, TARGET_LAYER]
TOKENS_PER_SHARD = 150_000
TOKEN_SCOPES = ("response", "prompt")


def load_tokenizer(base_model=DEFAULT_BASE_MODEL):
    return AutoTokenizer.from_pretrained(str(MODEL_DIR / base_model), trust_remote_code=True)


def load_jsonl(source):
    with source.open() as input_file:
        return [json.loads(line) for line in input_file]


def model_input_and_cache_slice(prompt_token_ids, response_token_ids, token_scope):
    """Return the model input, hidden-state slice, and token IDs for one cache scope."""
    if token_scope == "prompt":
        return prompt_token_ids, slice(0, len(prompt_token_ids)), prompt_token_ids
    prompt_length = len(prompt_token_ids)
    full_token_ids = prompt_token_ids + response_token_ids
    return full_token_ids, slice(prompt_length, prompt_length + len(response_token_ids)), response_token_ids


def cache(
    data_set_names,
    dataset,
    base_model,
    cache_tag,
    requested_layers=None,
    token_scope="response",
    overwrite=False,
):
    # Load the generator once and reuse it for every requested dataset split.
    tokenizer = load_tokenizer(base_model)
    special_token_ids = set(tokenizer.all_special_ids)
    restore_transformers_loss_kwargs()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR / base_model), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    repair_internlm3_rotary_embeddings(model)
    base = getattr(model, "model", model)  # base transformer: skip the unused 152k-vocab LM head

    def save_shard(out_dir, shard_number, activation_buffer, layers):
        """Write the buffered activations and return the next shard number."""
        if not activation_buffer[layers[0]]:
            return shard_number
        for layer in layers:
            shard_path = out_dir / f"layer{layer}.part{shard_number:04d}.pt"
            torch.save(torch.cat(activation_buffer[layer], dim=0), shard_path)
            activation_buffer[layer].clear()
        return shard_number + 1

    if requested_layers is not None:
        requested_layers = list(dict.fromkeys(requested_layers))

    # Resolve the source rows and hidden-state layers for each cache.
    jobs = []
    for name in data_set_names:
        selection = dataset_split(name, dataset)
        if selection.kind == "train":
            layers = requested_layers or TRANSCODER_LAYERS
        else:
            layers = requested_layers or ([READ_LAYER] if selection.name == "test" else TRANSCODER_LAYERS)
        if selection.source.suffix == ".json":
            rows = json.loads(selection.source.read_text())
        else:
            rows = load_jsonl(selection.source)
        jobs.append((selection, rows, layers))

    for selection, rows, layers in jobs:
        # Prepare a separate cache directory for this dataset split.
        out_dir = ACTIVATION_CACHE_DIR / cache_tag / dataset / selection.name
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"activation cache is not empty: {out_dir}; pass --overwrite to rebuild it")
        split = out_dir.relative_to(ACTIVATION_CACHE_DIR).as_posix()
        max_response_tokens = selection.max_response_tokens
        id_key = selection.id_key
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("layer*.pt"):
            stale.unlink()
        if overwrite:
            for generated_file in (out_dir / "meta.jsonl", out_dir / "info.json"):
                generated_file.unlink(missing_ok=True)

        metadata_file = open(out_dir / "meta.jsonl", "w")
        activation_buffer = {layer: [] for layer in layers}
        shard_number = token_count = hidden_size = cached_examples = max_prompt_tokens = 0

        # Run the generator on each saved response and buffer the requested hidden states.
        for row in rows:
            prompt_token_ids = format_prompt_token_ids(tokenizer, row["prompt"])
            response_token_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
            response_token_ids = response_token_ids[:max_response_tokens]
            if not response_token_ids:
                continue
            model_token_ids, activation_slice, cached_token_ids = model_input_and_cache_slice(
                prompt_token_ids, response_token_ids, token_scope
            )
            input_ids = torch.tensor([model_token_ids], device="cuda")
            with torch.no_grad():
                hidden_states = base(input_ids, output_hidden_states=True, use_cache=False).hidden_states
            if token_count == 0:
                print(
                    f"[{split}] scope={token_scope} first idx={row[id_key]}: "
                    f"prompt_tok={len(prompt_token_ids)} resp_tok={len(response_token_ids)} "
                    f"forward_seq={input_ids.shape[1]} n_hs={len(hidden_states)} "
                    f"hidden={hidden_states[layers[0]].shape[-1]}",
                    flush=True,
                )
            for layer in layers:
                block = hidden_states[layer][0, activation_slice, :].to(torch.bfloat16).cpu()
                activation_buffer[layer].append(block)
            hidden_size = hidden_states[layers[0]].shape[-1]
            for position, token_id in enumerate(cached_token_ids):
                token_metadata = {
                    "split": split,
                    "idx": int(row[id_key]),
                    "label": int(row["label"]),
                    "pos": position,
                    "token_id": int(token_id),
                    "is_special": int(token_id) in special_token_ids,
                    "token_scope": token_scope,
                }
                metadata_file.write(json.dumps(token_metadata) + "\n")
            cached_examples += 1
            max_prompt_tokens = max(max_prompt_tokens, len(prompt_token_ids))
            token_count += len(cached_token_ids)
            if sum(block.shape[0] for block in activation_buffer[layers[0]]) >= TOKENS_PER_SHARD:
                shard_number = save_shard(out_dir, shard_number, activation_buffer, layers)

        # Flush the final partial shard and record the cache layout.
        shard_number = save_shard(out_dir, shard_number, activation_buffer, layers)
        metadata_file.close()
        json.dump(
            {"schema_version": 3, "layers": layers, "read": layers[0],
             "target": layers[-1] if len(layers) > 1 else None, "n_tokens": token_count,
             "hidden": hidden_size, "max_resp": max_response_tokens, "n_examples": len(rows),
             "n_cached_examples": cached_examples, "max_prompt_tokens": max_prompt_tokens,
             "dtype": "bfloat16", "n_parts": shard_number, "dataset": dataset,
             "split": selection.name, "data_set": selection.name,
             "cache_tag": cache_tag, "source": str(selection.source),
             "base_model": base_model, "token_scope": token_scope,
             "prefix_definition": "full_generation_prefix" if token_scope == "prompt" else None,
             "enable_thinking": False},
            open(out_dir / "info.json", "w"), indent=2,
        )
        print(
            f"[{split}] {token_count} {token_scope} tokens from {cached_examples} examples "
            f"in {shard_number} shards -> {out_dir}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--data-set", dest="data_sets", nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+",
                    help="hidden-state indices to cache instead of the standard layer set")
    ap.add_argument("--cache-tag", help="write under output/activations/<tag>/<dataset>/<set>")
    ap.add_argument("--token-scope", choices=TOKEN_SCOPES, default="response",
                    help="cache response tokens (default) or the complete formatted generation prefix")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow rebuilding an existing prompt cache destination")
    args = ap.parse_args()
    if args.cache_tag is None:
        ap.error("--cache-tag is required")
    cache(
        args.data_sets,
        args.dataset,
        args.base_model,
        args.cache_tag,
        args.layers,
        args.token_scope,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
