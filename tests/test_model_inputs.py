"""Tests for prompt formatting and activation batching helpers."""

import torch

from model_inputs import (
    format_prompt_token_ids,
    iter_activation_batches,
    pad_activation_sequences,
    restore_transformers_loss_kwargs,
)


class ChatTokenizer:
    """Records apply_chat_template calls and can emulate the legacy Transformers signature."""

    def __init__(self, shape="dict", supports_thinking=True):
        self.shape = shape
        self.supports_thinking = supports_thinking
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False,
                            enable_thinking=None):
        self.calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "enable_thinking": enable_thinking,
            }
        )
        if enable_thinking is not None and not self.supports_thinking:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
        if self.shape == "dict":
            return {"input_ids": [10, 11, 12]}
        return [10, 11, 12]


class TestFormatPromptTokenIds:
    def test_unwraps_mapping_result(self):
        tokenizer = ChatTokenizer(shape="dict")
        assert format_prompt_token_ids(tokenizer, "hello") == [10, 11, 12]

    def test_accepts_plain_list_result(self):
        tokenizer = ChatTokenizer(shape="list")
        assert format_prompt_token_ids(tokenizer, "hello") == [10, 11, 12]

    def test_wraps_prompt_in_a_user_message_and_opens_generation(self):
        tokenizer = ChatTokenizer()
        format_prompt_token_ids(tokenizer, "hello")
        call = tokenizer.calls[0]
        assert call["messages"] == [{"role": "user", "content": "hello"}]
        assert call["add_generation_prompt"] is True
        assert call["tokenize"] is True

    def test_disables_thinking_when_supported(self):
        tokenizer = ChatTokenizer()
        format_prompt_token_ids(tokenizer, "hello")
        assert tokenizer.calls[0]["enable_thinking"] is False

    def test_retries_without_enable_thinking(self):
        tokenizer = ChatTokenizer(supports_thinking=False)
        assert format_prompt_token_ids(tokenizer, "hello") == [10, 11, 12]
        assert len(tokenizer.calls) == 2
        assert tokenizer.calls[0]["enable_thinking"] is False
        assert tokenizer.calls[1]["enable_thinking"] is None


class TestPadActivationSequences:
    def test_shapes_dtype_and_right_padding(self):
        sequences = [{"x": torch.ones(3, 4)}, {"x": torch.full((5, 4), 2.0)}]
        padded = pad_activation_sequences(sequences, "x", torch.device("cpu"))
        assert padded.shape == (2, 5, 4)
        assert padded.dtype == torch.float32
        assert torch.equal(padded[0, :3], torch.ones(3, 4))
        assert torch.equal(padded[0, 3:], torch.zeros(2, 4))
        assert torch.equal(padded[1], torch.full((5, 4), 2.0))

    def test_single_sequence_is_unchanged(self):
        sequences = [{"x": torch.randn(4, 3)}]
        padded = pad_activation_sequences(sequences, "x", torch.device("cpu"))
        assert torch.allclose(padded[0], sequences[0]["x"])

    def test_does_not_mutate_the_inputs(self):
        sequences = [{"x": torch.ones(3, 2)}, {"x": torch.ones(5, 2)}]
        before = [sequence["x"].clone() for sequence in sequences]
        pad_activation_sequences(sequences, "x", torch.device("cpu"))
        for sequence, original in zip(sequences, before):
            assert torch.equal(sequence["x"], original)

    def test_cast_to_float32(self):
        sequences = [{"x": torch.ones(2, 2, dtype=torch.float64)}]
        assert pad_activation_sequences(sequences, "x", torch.device("cpu")).dtype == torch.float32


def make_batch_sequences(lengths):
    return [{"idx": index, "x": torch.zeros(length, 2)} for index, length in enumerate(lengths)]


class TestIterActivationBatches:
    def test_sequential_batches_cover_every_sequence_in_order(self):
        sequences = make_batch_sequences([3, 1, 4, 1, 5])
        batches = list(iter_activation_batches(sequences, batch_size=2, shuffle=False, seed=0))
        assert [len(batch) for batch in batches] == [2, 2, 1]
        assert [record["idx"] for batch in batches for record in batch] == [0, 1, 2, 3, 4]

    def test_batch_size_exceeding_the_population(self):
        sequences = make_batch_sequences([2, 2])
        batches = list(iter_activation_batches(sequences, batch_size=10, shuffle=False, seed=0))
        assert len(batches) == 1 and len(batches[0]) == 2

    def test_shuffled_batches_cover_every_sequence_exactly_once(self):
        sequences = make_batch_sequences([3, 1, 4, 1, 5, 9, 2, 6])
        batches = list(iter_activation_batches(sequences, batch_size=3, shuffle=True, seed=42))
        indexes = sorted(record["idx"] for batch in batches for record in batch)
        assert indexes == list(range(8))

    def test_shuffled_batches_are_length_bucketed(self):
        sequences = make_batch_sequences([1, 9, 2, 8, 3, 7])
        batches = list(iter_activation_batches(sequences, batch_size=2, shuffle=True, seed=1))
        for batch in batches:
            lengths = [record["x"].shape[0] for record in batch]
            assert lengths == sorted(lengths)

    def test_shuffled_order_is_reproducible(self):
        sequences = make_batch_sequences([3, 1, 4, 1, 5, 9, 2, 6])
        first = list(iter_activation_batches(sequences, batch_size=3, shuffle=True, seed=5))
        second = list(iter_activation_batches(sequences, batch_size=3, shuffle=True, seed=5))
        assert [[record["idx"] for record in batch] for batch in first] == [
            [record["idx"] for record in batch] for batch in second
        ]

    def test_batch_order_depends_on_the_seed(self):
        sequences = make_batch_sequences(list(range(1, 13)))
        orders = {
            tuple(tuple(record["idx"] for record in batch)
                  for batch in iter_activation_batches(sequences, batch_size=3, shuffle=True, seed=seed))
            for seed in range(5)
        }
        assert len(orders) > 1


class TestRestoreTransformersLossKwargs:
    def test_installs_the_legacy_symbol(self):
        import transformers.utils as transformers_utils

        had_symbol = hasattr(transformers_utils, "LossKwargs")
        saved = getattr(transformers_utils, "LossKwargs", None)
        if had_symbol:
            del transformers_utils.LossKwargs
        try:
            restore_transformers_loss_kwargs()
            assert hasattr(transformers_utils, "LossKwargs")
            assert transformers_utils.LossKwargs.__module__ == "transformers.utils"
            assert "num_items_in_batch" in transformers_utils.LossKwargs.__annotations__
            # Calling again must not replace the installed symbol.
            installed = transformers_utils.LossKwargs
            restore_transformers_loss_kwargs()
            assert transformers_utils.LossKwargs is installed
        finally:
            if had_symbol:
                transformers_utils.LossKwargs = saved
            elif hasattr(transformers_utils, "LossKwargs"):
                del transformers_utils.LossKwargs

    def test_is_a_no_op_when_already_present(self):
        import transformers.utils as transformers_utils

        sentinel = object()
        had_symbol = hasattr(transformers_utils, "LossKwargs")
        saved = getattr(transformers_utils, "LossKwargs", None)
        transformers_utils.LossKwargs = sentinel
        try:
            restore_transformers_loss_kwargs()
            assert transformers_utils.LossKwargs is sentinel
        finally:
            if had_symbol:
                transformers_utils.LossKwargs = saved
            else:
                del transformers_utils.LossKwargs
