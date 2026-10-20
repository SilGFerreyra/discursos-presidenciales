"""Query-encoding regressions using deterministic tensors, without model downloads."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runtime import configure_runtime
configure_runtime()

import numpy as np
import torch
import embedding_logic as embedding


class Tokenizer:
    def __call__(self, texts, return_tensors=None, **kwargs):
        ids = [[0, 10, 2] for _ in texts]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones((len(texts), 3), dtype=torch.long)}
        return {"input_ids": ids}


class Model:
    def __init__(self, zero=False):
        self.zero = zero

    def __call__(self, **batch):
        hidden = torch.full((len(batch["input_ids"]), 3, 4), 100.0)
        hidden[:, 0, :] = 0
        if not self.zero:
            hidden[:, 0, :2] = torch.tensor([3.0, 4.0])
        return SimpleNamespace(last_hidden_state=hidden)


class EmbeddingTests(unittest.TestCase):
    def test_cls_pooling_excludes_other_tokens_and_normalizes(self):
        vectors, raw, normalized, tokens, flags = embedding.encode_batch(
            ["uno", "dos"], Tokenizer(), Model(), torch.device("cpu"), 8192)
        np.testing.assert_allclose(vectors, [[0.6, 0.8, 0, 0]] * 2)
        np.testing.assert_allclose(raw, 5)
        np.testing.assert_allclose(normalized, 1)
        np.testing.assert_array_equal(tokens, 3)
        np.testing.assert_array_equal(flags, "ok")

    def test_zero_vector_is_invalid_instead_of_silently_searchable(self):
        vectors, _, norms, _, flags = embedding.encode_batch(
            ["uno"], Tokenizer(), Model(zero=True), torch.device("cpu"), 8192)
        self.assertTrue(np.isnan(vectors).all() and np.isnan(norms).all())
        np.testing.assert_array_equal(flags, "error")

    def test_oversized_query_is_rejected_before_model_inference(self):
        model = Mock()
        with self.assertRaises(ValueError):
            embedding.encode_batch(["uno"], Tokenizer(), model, torch.device("cpu"), 2)
        model.assert_not_called()

    def test_model_load_remains_pinned_local_and_restricted(self):
        tokenizer = SimpleNamespace(cls_token_id=0, model_max_length=8192)
        model = Mock(config=SimpleNamespace(hidden_size=1024))
        with patch.object(embedding.AutoTokenizer, "from_pretrained", return_value=tokenizer) as token_load, \
                patch.object(embedding.AutoModel, "from_pretrained", return_value=model) as model_load:
            self.assertEqual(embedding.load_model(torch.device("cpu"), "fp32", 8192), (tokenizer, model))
        for call in (token_load.call_args, model_load.call_args):
            self.assertEqual(call.args[0], embedding.MODEL_ID)
            self.assertEqual(call.kwargs["revision"], embedding.MODEL_REVISION)
            self.assertTrue(call.kwargs["local_files_only"])
            self.assertFalse(call.kwargs["trust_remote_code"])
        self.assertTrue(model_load.call_args.kwargs["weights_only"])
        self.assertFalse(model_load.call_args.kwargs["use_safetensors"])

    def test_online_loading_is_rejected(self):
        with patch.object(embedding.AutoModel, "from_pretrained") as load, self.assertRaises(ValueError):
            embedding.load_model(torch.device("cpu"), "fp32", 8192, local_files_only=False)
        load.assert_not_called()

    def test_fingerprint_ignores_dictionary_order(self):
        self.assertEqual(embedding.config_fingerprint({"a": 1, "b": 2}),
                         embedding.config_fingerprint({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
