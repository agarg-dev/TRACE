#!/usr/bin/env python
"""Cache generator response activations for TRACE."""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activations.activation_cache import (
    activation_cache_output_path,
)
from data.dataset_splits import DETECTOR_TRAIN_SET_BY_DATASET, TEST_SETS, TRAIN_SETS
from data.dataset_splits import evaluation_split, training_split
from project_config import ACTIVATION_CACHE_DIR, DEFAULT_BASE_MODEL, DEFAULT_DATASET
from project_config import READ_LAYER, TARGET_LAYER, base_model_path
from model_inputs import (
    configure_transformers_compatibility, format_prompt_token_ids,
    initialize_internlm3_rotary_embeddings,
)

TRANSCODER_LAYERS = [READ_LAYER, TARGET_LAYER]
TOKENS_PER_SHARD = 150_000


def expand_split_names(names, dataset):
    """Expand ``both`` to the training pool and official test set."""
    expanded = []
    for name in names:
        expanded.extend([DETECTOR_TRAIN_SET_BY_DATASET[dataset], "test"] if name == "both" else [name])
    return expanded


def cache(
    split_names,
    dataset,
    base_model,
    requested_layers=None,
    cache_tag=None,
    overwrite=False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path(base_model)), trust_remote_code=True
    )
    special_token_ids = set(tokenizer.all_special_ids)
    configure_transformers_compatibility()
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path(base_model)), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    initialize_internlm3_rotary_embeddings(model)
    base = model.model

    def save_shard(out_dir, shard_number, activation_buffer, layers):
        """Write and clear one activation shard."""
        if not activation_buffer[layers[0]]:
            return shard_number
        for layer in layers:
            shard_path = out_dir / f"layer{layer}.part{shard_number:04d}.pt"
            torch.save(torch.cat(activation_buffer[layer], dim=0), shard_path)
            activation_buffer[layer].clear()
        return shard_number + 1

    if requested_layers is not None:
        requested_layers = list(dict.fromkeys(requested_layers))
        if any(layer < 0 for layer in requested_layers):
            raise ValueError("--layers must contain non-negative hidden-state indices")

    for name in expand_split_names(split_names, dataset):
        if name in TRAIN_SETS:
            selection = training_split(name, dataset)
            layers = requested_layers or TRANSCODER_LAYERS
        elif name in TEST_SETS:
            selection = evaluation_split(name, dataset)
            layers = requested_layers or ([READ_LAYER] if selection.name == "test" else TRANSCODER_LAYERS)
        else:
            choices = ", ".join((*TRAIN_SETS, *TEST_SETS, "both"))
            raise ValueError(f"unknown data set {name!r}; choose {choices}")
        rows = (json.load(open(selection.source)) if selection.source.suffix == ".json"
                else [json.loads(line) for line in open(selection.source)])

        out_dir = activation_cache_output_path(dataset, selection, tag=cache_tag)
        cache_exists = out_dir.exists() and any(out_dir.iterdir())
        if cache_exists and not overwrite:
            raise FileExistsError(f"activation cache exists: {out_dir}; pass --overwrite to replace it")
        split = out_dir.relative_to(ACTIVATION_CACHE_DIR).as_posix()
        max_response_tokens = selection.max_response_tokens
        id_key = selection.id_key
        out_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for stale in out_dir.glob("layer*.pt"):
                stale.unlink()
            for generated_file in (out_dir / "meta.jsonl", out_dir / "info.json"):
                generated_file.unlink(missing_ok=True)

        metadata_file = open(out_dir / "meta.jsonl", "w")
        activation_buffer = {layer: [] for layer in layers}
        shard_number = token_count = hidden_size = cached_examples = 0

        for row in rows:
            prompt_token_ids = format_prompt_token_ids(tokenizer, row["prompt"])
            response_token_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
            response_token_ids = response_token_ids[:max_response_tokens]
            if not response_token_ids:
                continue
            prompt_length = len(prompt_token_ids)
            model_token_ids = prompt_token_ids + response_token_ids
            activation_slice = slice(prompt_length, prompt_length + len(response_token_ids))
            input_ids = torch.tensor([model_token_ids], device="cuda")
            with torch.no_grad():
                hidden_states = base(input_ids, output_hidden_states=True, use_cache=False).hidden_states
            if token_count == 0:
                print(
                    f"[{split}] first idx={row[id_key]}: "
                    f"prompt_tok={len(prompt_token_ids)} resp_tok={len(response_token_ids)} "
                    f"forward_seq={input_ids.shape[1]} n_hs={len(hidden_states)} "
                    f"hidden={hidden_states[layers[0]].shape[-1]}",
                    flush=True,
                )
            for layer in layers:
                block = hidden_states[layer][0, activation_slice, :].to(torch.bfloat16).cpu()
                activation_buffer[layer].append(block)
            hidden_size = hidden_states[layers[0]].shape[-1]
            for position, token_id in enumerate(response_token_ids):
                token_metadata = {
                    "split": split, "idx": int(row[id_key]), "label": int(row["label"]),
                    "pos": position, "token_id": int(token_id), "is_special": int(token_id) in special_token_ids,
                }
                metadata_file.write(json.dumps(token_metadata) + "\n")
            cached_examples += 1
            token_count += len(response_token_ids)
            if sum(block.shape[0] for block in activation_buffer[layers[0]]) >= TOKENS_PER_SHARD:
                shard_number = save_shard(out_dir, shard_number, activation_buffer, layers)

        shard_number = save_shard(out_dir, shard_number, activation_buffer, layers)
        metadata_file.close()
        cache_info = {
            "layers": layers, "n_tokens": token_count,
            "hidden": hidden_size, "max_resp": max_response_tokens, "n_examples": len(rows),
            "n_cached_examples": cached_examples,
            "dtype": "bfloat16", "n_parts": shard_number, "dataset": dataset,
            "split": selection.name, "source": str(selection.source), "base_model": base_model,
        }
        with (out_dir / "info.json").open("w") as info_file:
            json.dump(cache_info, info_file, indent=2)
        print(
            f"[{split}] {token_count} response tokens from {cached_examples} examples "
            f"in {shard_number} shards -> {out_dir}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-set", "--split", dest="split_names", nargs="+", default=["both"],
                        help="training set, test set, or both")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="hidden-state indices to cache instead of the standard layer set")
    parser.add_argument("--cache-tag",
                        help="write under output/activations/<tag>/<dataset>/<set> instead of the main cache")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing cache")
    args = parser.parse_args()
    cache(
        args.split_names,
        args.dataset,
        args.base_model,
        args.layers,
        args.cache_tag,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
