"""Mocked classifier/tokenizer tests. No model load or network allowed."""
from runtime import configure_runtime
configure_runtime()

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rerank_logic import RerankConfig, RerankerResource, load_reranker, score_candidates, RERANK_REVISION
from test_support import FakeTokenizer, FakeModel


class RerankTests(unittest.TestCase):
    def resource(self, **config):
        return RerankerResource(FakeTokenizer(), FakeModel(), RerankConfig(**config))

    def test_all_four_pairs_candidate_major_raw_logits(self):
        resource = self.resource()
        output = score_candidates(resource, ["one", "two", "three", "four"], [42, 7], ["passage a", "passage b"])
        self.assertEqual(output.candidate_ids, (42, 7))
        np.testing.assert_array_equal(output.logits, [[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertEqual(output.metrics["pair_evaluations"], 8)
        self.assertEqual(output.metrics["evaluated_candidates"], 2)
        self.assertEqual(resource.model.calls, 2)
        self.assertEqual(resource.model.grad_flags, [False, False])
        self.assertFalse(output.logits.flags.writeable)

    def test_preserves_carriers_truncates_passage_only_counts_and_dynamic_padding(self):
        resource = self.resource(max_length=8, batch_size=1)
        carriers = ["one", "one two", "one", "one two"]
        output = score_candidates(resource, carriers, [0], ["a b c d e f g h i"])
        self.assertEqual(output.metrics["truncated_pairs"], 4)
        expected_carriers = resource.tokenizer(carriers)["input_ids"]
        for index, (query, passage) in enumerate(resource.tokenizer.prepared):
            self.assertEqual(query, expected_carriers[index])
            self.assertEqual(len(passage), 8 - len(query) - 3)
        small = self.resource(max_length=512)
        score_candidates(small, ["one"] * 4, [0], ["two"])
        self.assertEqual(small.tokenizer.padded_widths, [5])

    def test_rejects_carrier_without_room(self):
        resource = self.resource(max_length=8)
        with self.assertRaises(ValueError):
            score_candidates(resource, ["a b c d e"] * 4, [0], ["passage"])
        self.assertEqual(resource.model.calls, 0)

    def test_invalid_inputs_and_configuration(self):
        resource = self.resource()
        for carriers, ids, passages in ((["a"] * 3, [0], ["b"]), ([""] * 4, [0], ["b"]),
                                        (["a"] * 4, [0, 0], ["b", "c"]), (["a"] * 4, [0], []),
                                        (["a"] * 4, [-1], ["b"]), (["a"] * 4, [0], [""])):
            with self.assertRaises(ValueError):
                score_candidates(resource, carriers, ids, passages)
        for kwargs in ({"batch_size": 0}, {"max_length": 513}, {"revision": "main"}, {"device": "mps"}, {"batch_size": True}):
            with self.assertRaises(ValueError):
                RerankConfig(**kwargs)

    def test_empty_candidates_no_model_forward(self):
        resource = self.resource()
        result = score_candidates(resource, ["a"] * 4, [], [])
        self.assertEqual(result.logits.shape, (0, 4))
        self.assertEqual(resource.model.calls, 0)

    def test_mid_batch_failure_no_output_or_input_mutation(self):
        resource = self.resource()
        resource.model.fail_batch = 2
        ids, passages = [0, 1], ["a", "b"]
        with self.assertRaises(RuntimeError):
            score_candidates(resource, ["query"] * 4, ids, passages)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(passages, ["a", "b"])

    def test_nonfinite_or_incomplete_logits_rejected(self):
        for values in ([1, 2, float("nan"), 4], [1, 2, float("inf"), 4], [1, 2, 3]):
            resource = self.resource()
            resource.model = FakeModel(values)
            with self.assertRaises(ValueError):
                score_candidates(resource, ["q"] * 4, [0], ["p"])

    def test_pair_validation_rejects_corrupt_tokenizer(self):
        resource = self.resource()
        original = resource.tokenizer.prepare_for_model
        def corrupt(*args, **kwargs):
            item = original(*args, **kwargs)
            item["input_ids"][1] = 999
            return item
        resource.tokenizer.prepare_for_model = corrupt
        with self.assertRaises(ValueError):
            score_candidates(resource, ["q"] * 4, [0], ["p"])

    def test_loader_pinned_safetensors_cpu_eval_offline(self):
        tokenizer, model = FakeTokenizer(), FakeModel()
        with patch("rerank_logic.AutoTokenizer.from_pretrained", return_value=tokenizer) as token_load, \
             patch("rerank_logic.AutoModelForSequenceClassification.from_pretrained", return_value=model) as model_load:
            resource = load_reranker(RerankConfig(), local_files_only=True, cache_dir="app-owned-cache")
        for call in (token_load.call_args, model_load.call_args):
            self.assertEqual(call.kwargs["revision"], RERANK_REVISION)
            self.assertFalse(call.kwargs["trust_remote_code"])
            self.assertTrue(call.kwargs["local_files_only"])
        self.assertTrue(model_load.call_args.kwargs["use_safetensors"])
        self.assertEqual(model_load.call_args.kwargs["dtype"], torch.float32)
        model.eval.assert_called_once()
        self.assertEqual(resource.config.precision, "fp32")

    def test_unavailable_cuda_rejected_before_loading(self):
        with patch("rerank_logic.torch.cuda.is_available", return_value=False), \
             patch("rerank_logic.AutoTokenizer.from_pretrained") as loader:
            with self.assertRaises(RuntimeError):
                load_reranker(RerankConfig(device="cuda"))
        loader.assert_not_called()

    def test_shared_resource_serializes_inference(self):
        resource = self.resource()
        active = {"count": 0, "maximum": 0}
        base = resource.model
        def forward(**batch):
            active["count"] += 1
            active["maximum"] = max(active["maximum"], active["count"])
            time.sleep(0.01)
            result = base(**batch)
            active["count"] -= 1
            return result
        resource.model = forward
        errors = []
        def work():
            try:
                score_candidates(resource, ["q"] * 4, [0], ["p"])
            except Exception as error:
                errors.append(error)
        workers = [threading.Thread(target=work) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(errors, [])
        self.assertEqual(active["maximum"], 1)


if __name__ == "__main__":
    unittest.main()
