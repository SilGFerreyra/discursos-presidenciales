import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from generar_embeddings_bge_m3_colab import (
    EMBEDDING_DIM,
    compatibility_config,
    consolidate_shards,
    config_fingerprint,
    encode_batch,
    main,
    prepare_output,
    process_shards,
    summarize_shard_provenance,
    validate_canonical_input,
    validate_final_artifacts,
    validate_input,
    validate_shard,
)


class DummyTokenizer:
    cls_token_id = 0
    model_max_length = 8192

    def __call__(self, texts, return_tensors=None, **kwargs):
        input_ids = [[0, 10, 2] for _ in texts]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
            }
        return {"input_ids": input_ids}


class DummyModel:
    def __init__(self, zero_cls=False, dimension=4):
        self.zero_cls = zero_cls
        self.dimension = dimension

    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        hidden = torch.full((batch, 3, self.dimension), 100.0)
        hidden[:, 0, :] = 0.0
        if not self.zero_cls:
            hidden[:, 0, 0] = 3.0
            hidden[:, 0, 1] = 4.0
        return SimpleNamespace(last_hidden_state=hidden)


class EmbeddingPipelineTests(unittest.TestCase):
    @staticmethod
    def make_compatibility_config(**overrides):
        values = {
            "input_sha256": "input-sha",
            "source_rows": 2,
            "row_count": 2,
            "max_length": 8192,
            "precision": "fp32",
            "shard_size": 2,
            "official_pooling_config": {
                "word_embedding_dimension": EMBEDDING_DIM,
                "pooling_mode_cls_token": True,
                "pooling_mode_mean_tokens": False,
            },
        }
        values.update(overrides)
        return compatibility_config(**values)

    @staticmethod
    def make_provenance(name="runtime-a", requested_batch_size=2):
        return {
            "input_file": "chunks.csv",
            "requested_batch_size": requested_batch_size,
            "runtime": {
                "device_type": "cuda",
                "hardware": {"gpu_name": name},
                "python_version": "3.12",
                "torch_version": name,
                "transformers_version": name,
                "script_sha256": name,
            },
        }

    @staticmethod
    def make_args(input_path, output_dir, **overrides):
        values = {
            "input": Path(input_path),
            "output_dir": Path(output_dir),
            "batch_size": 2,
            "shard_size": 2,
            "max_length": 8192,
            "device": "cpu",
            "precision": "fp32",
            "limit": None,
            "allow_cpu": True,
            "overwrite": False,
            "accept_other_input": True,
            "local_files_only": True,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_encode_batch_uses_cls_and_normalizes(self):
        embeddings, raw_norms, normalized_norms, token_counts, flags = encode_batch(
            ["uno", "dos"],
            DummyTokenizer(),
            DummyModel(),
            torch.device("cpu"),
            8192,
        )

        np.testing.assert_allclose(embeddings[:, :2], [[0.6, 0.8], [0.6, 0.8]])
        np.testing.assert_allclose(embeddings[:, 2:], 0.0)
        np.testing.assert_allclose(raw_norms, 5.0)
        np.testing.assert_allclose(normalized_norms, 1.0)
        np.testing.assert_array_equal(token_counts, 3)
        np.testing.assert_array_equal(flags, "ok")

    def test_encode_batch_flags_zero_cls(self):
        embeddings, raw_norms, normalized_norms, _, flags = encode_batch(
            ["texto"],
            DummyTokenizer(),
            DummyModel(zero_cls=True),
            torch.device("cpu"),
            8192,
        )

        self.assertTrue(np.isnan(embeddings).all())
        np.testing.assert_allclose(raw_norms, 0.0)
        self.assertTrue(np.isnan(normalized_norms).all())
        np.testing.assert_array_equal(flags, "error")

    def test_validate_input_rejects_invalid_identity(self):
        invalid = pd.DataFrame({"doc_id": [None], "chunk_id": [1], "texto_chunk": ["texto"]})
        with self.assertRaises(ValueError):
            validate_input(invalid)

    def test_config_fingerprint_is_order_independent(self):
        self.assertEqual(config_fingerprint({"a": 1, "b": 2}), config_fingerprint({"b": 2, "a": 1}))

    def test_runtime_and_batch_are_excluded_from_compatibility_config(self):
        config = self.make_compatibility_config()
        first = self.make_provenance("t4", requested_batch_size=32)
        second = self.make_provenance("a100", requested_batch_size=16)

        self.assertNotEqual(first, second)
        self.assertTrue(
            {"device_type", "hardware", "batch_size", "torch_version", "transformers_version", "script_sha256"}
            .isdisjoint(config)
        )

    def test_prepare_output_rejects_semantic_configuration_changes(self):
        provenance = self.make_provenance()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            config = self.make_compatibility_config()
            prepare_output(output_dir, config, provenance, overwrite=False)

            for key, value in {
                "input_sha256": "other-input",
                "model_revision": "other-revision",
                "pooling": "mean",
                "precision": "fp16",
                "max_length": 4096,
                "shard_size": 4,
            }.items():
                changed = config.copy()
                changed[key] = value
                with self.subTest(key=key):
                    with self.assertRaises(RuntimeError):
                        prepare_output(output_dir, changed, provenance, overwrite=False)

    def test_shard_provenance_detects_mixed_runtimes(self):
        first = self.make_provenance("t4")
        first["effective_batch_size"] = 2
        second = self.make_provenance("a100")
        second["effective_batch_size"] = 2

        summary = summarize_shard_provenance([first, second])

        self.assertTrue(summary["mixed_runtimes"])
        self.assertEqual(summary["unique_runtime_count"], 2)
        self.assertEqual(sum(item["shard_count"] for item in summary["runtimes"]), 2)

    def test_canonical_input_is_required_by_default(self):
        with self.assertRaises(RuntimeError):
            validate_canonical_input(2, "wrong", accept_other_input=False)
        validate_canonical_input(2, "wrong", accept_other_input=True)

    def test_shard_identity_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            embeddings = np.zeros((2, EMBEDDING_DIM), dtype=np.float32)
            embeddings[:, 0] = 1.0
            provenance_payload = self.make_provenance()
            provenance_payload["effective_batch_size"] = 2
            np.savez(
                path,
                embeddings=embeddings,
                raw_norms=np.ones(2, dtype=np.float32),
                normalized_norms=np.ones(2, dtype=np.float32),
                token_counts=np.full(2, 10, dtype=np.int32),
                flags=np.asarray(["ok", "ok"]),
                shard_start=np.int64(0),
                shard_end=np.int64(2),
                config_fingerprint=np.asarray("fingerprint"),
                generation_provenance=np.asarray(json.dumps(provenance_payload)),
            )
            provenance = validate_shard(path, 0, 2, "fingerprint", 8192)
            self.assertEqual(provenance["runtime"]["hardware"]["gpu_name"], "runtime-a")
            with self.assertRaises(RuntimeError):
                validate_shard(path, 0, 2, "different", 8192)

    def test_final_artifacts_preserve_order_and_vectors(self):
        source = pd.DataFrame(
            {
                "doc_id": ["a.pdf", "a.pdf"],
                "chunk_id": [1, 2],
                "texto_chunk": ["primer texto", "segundo texto"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            shard_dir = output_dir / "shards"
            shard_dir.mkdir()
            embeddings = np.zeros((2, EMBEDDING_DIM), dtype=np.float32)
            embeddings[0, 0] = 1.0
            embeddings[1, 1] = 1.0
            np.savez(
                shard_dir / "shard-000000-000002.npz",
                embeddings=embeddings,
                raw_norms=np.asarray([5.0, 6.0], dtype=np.float32),
                normalized_norms=np.ones(2, dtype=np.float32),
                token_counts=np.asarray([4, 5], dtype=np.int32),
                flags=np.asarray(["ok", "ok"]),
            )

            df_with_embeddings, matrix = consolidate_shards(source, output_dir, shard_size=2)
            result = validate_final_artifacts(source, output_dir, matrix, max_length=8192)

            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["npy_parquet_vectors_identical"])
            del df_with_embeddings, matrix

    def test_process_shards_aborts_on_invalid_embedding(self):
        invalid_result = (
            np.full((1, EMBEDDING_DIM), np.nan, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.full(1, np.nan, dtype=np.float32),
            np.ones(1, dtype=np.int32),
            np.asarray(["error"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "shards").mkdir()
            with patch("generar_embeddings_bge_m3_colab.encode_batch", return_value=invalid_result):
                with self.assertRaises(RuntimeError):
                    process_shards(
                        ["texto"], DummyTokenizer(), DummyModel(), torch.device("cpu"),
                        8192, 1, 1, output_dir, "fingerprint", self.make_provenance()
                    )

    def test_cuda_oom_reduces_batch_for_following_shards(self):
        calls = []

        def fake_encode(texts, *args):
            calls.append(len(texts))
            if len(texts) > 2:
                raise torch.cuda.OutOfMemoryError("simulated")
            rows = len(texts)
            embeddings = np.zeros((rows, EMBEDDING_DIM), dtype=np.float32)
            embeddings[:, 0] = 1.0
            return (
                embeddings,
                np.ones(rows, dtype=np.float32),
                np.ones(rows, dtype=np.float32),
                np.full(rows, 10, dtype=np.int32),
                np.full(rows, "ok"),
            )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "shards").mkdir()
            with patch("generar_embeddings_bge_m3_colab.encode_batch", side_effect=fake_encode):
                with patch("torch.cuda.empty_cache"):
                    stats = process_shards(
                        ["texto"] * 8, DummyTokenizer(), DummyModel(), torch.device("cuda"),
                        8192, 4, 4, output_dir, "fingerprint", self.make_provenance()
                    )

            self.assertEqual(calls, [4, 2, 2, 2, 2])
            self.assertEqual(stats["oom_retries"], 1)
            self.assertEqual(stats["minimum_batch_size"], 2)
            self.assertEqual(stats["shard_provenance"]["unique_runtime_count"], 1)

    def test_main_rejects_noncanonical_input_without_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "chunks.csv"
            pd.DataFrame(
                {"doc_id": ["a.pdf"], "chunk_id": [1], "texto_chunk": ["texto"]}
            ).to_csv(input_path, index=False)
            args = self.make_args(
                input_path,
                Path(directory) / "output",
                accept_other_input=False,
            )
            with self.assertRaises(RuntimeError):
                main(args)

    def test_main_runs_end_to_end_with_explicit_custom_input(self):
        official = {
            "pooling": {
                "word_embedding_dimension": EMBEDDING_DIM,
                "pooling_mode_cls_token": True,
                "pooling_mode_mean_tokens": False,
            },
            "modules": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "chunks.csv"
            output_dir = Path(directory) / "output"
            pd.DataFrame(
                {
                    "doc_id": ["a.pdf", "a.pdf"],
                    "chunk_id": [1, 2],
                    "texto_chunk": ["primer texto", "segundo texto"],
                }
            ).to_csv(input_path, index=False)
            args = self.make_args(input_path, output_dir)

            with patch("generar_embeddings_bge_m3_colab.verify_official_pooling", return_value=official):
                with patch(
                    "generar_embeddings_bge_m3_colab.load_model",
                    return_value=(DummyTokenizer(), DummyModel(dimension=EMBEDDING_DIM)),
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        df_chunks_emb, report = main(args)

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["artifact_validation"]["status"], "ok")
            self.assertFalse(report["shard_provenance"]["mixed_runtimes"])
            self.assertEqual(len(df_chunks_emb), 2)
            shard_path = output_dir / "shards" / "shard-000000-000002.npz"
            with np.load(shard_path, allow_pickle=False) as shard:
                provenance = json.loads(str(shard["generation_provenance"]))
            self.assertEqual(provenance["effective_batch_size"], 2)
            del df_chunks_emb

    def test_main_resumes_when_execution_runtime_changes(self):
        official = {
            "pooling": {
                "word_embedding_dimension": EMBEDDING_DIM,
                "pooling_mode_cls_token": True,
                "pooling_mode_mean_tokens": False,
            },
            "modules": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "chunks.csv"
            output_dir = Path(directory) / "output"
            pd.DataFrame(
                {"doc_id": ["a.pdf", "a.pdf"], "chunk_id": [1, 2], "texto_chunk": ["uno", "dos"]}
            ).to_csv(input_path, index=False)
            args = self.make_args(input_path, output_dir)

            with patch("generar_embeddings_bge_m3_colab.verify_official_pooling", return_value=official):
                with patch(
                    "generar_embeddings_bge_m3_colab.load_model",
                    return_value=(DummyTokenizer(), DummyModel(dimension=EMBEDDING_DIM)),
                ):
                    with patch(
                        "generar_embeddings_bge_m3_colab.execution_provenance",
                        side_effect=[self.make_provenance("t4"), self.make_provenance("a100")],
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            first_df, first_report = main(args)
                            del first_df
                            second_df, second_report = main(args)
                            del second_df

            self.assertEqual(first_report["execution"]["reused_shards"], 0)
            self.assertEqual(second_report["execution"]["reused_shards"], 1)
            manifest = json.loads((output_dir / "manifest-embeddings-bge-m3.json").read_text())
            self.assertEqual(len(manifest["execution_history"]), 2)
            self.assertEqual(manifest["execution_history"][0]["runtime"]["hardware"]["gpu_name"], "t4")
            self.assertEqual(manifest["execution_history"][1]["runtime"]["hardware"]["gpu_name"], "a100")


if __name__ == "__main__":
    unittest.main()
