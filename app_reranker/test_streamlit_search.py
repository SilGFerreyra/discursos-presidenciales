"""Adapted original search tests plus independent candidate and export contracts."""
from runtime import configure_runtime
configure_runtime()

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

import search_logic as search
from test_support import make_metadata, make_scores, make_embedding, make_bundle
from rerank_logic import RerankOutput, RerankConfig
from rerank_state import reranker_metadata


class SearchTests(unittest.TestCase):
    def test_initial_cpu_candidate_formula_preserves_limits_and_n_ge_k(self):
        for k, expected_n in ((10, 25), (20, 40), (100, 200), (500, 1000)):
            self.assertEqual(search.initial_candidate_budget(k), expected_n)
            self.assertGreaterEqual(expected_n, k)
        self.assertEqual(search.DEFAULT_TOP_K, 10)
    def candidates(self, n=4, k=2):
        return search.select_candidates(make_metadata(), make_scores(), [], (1900, 1903), [], (-1, 1), n, k)

    def test_centroid_averages_and_renormalizes(self):
        embedding = make_embedding()
        np.testing.assert_allclose(embedding.centroid[:4], 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(embedding.centroid)), 1)
        self.assertAlmostEqual(embedding.diagnostics["centroid_norm_before_l2"], 0.5)

    def test_centroid_rejects_shapes_norms_and_flags(self):
        for vectors, flags in ((np.zeros((3, 1024)), None), (np.ones((4, 1024)), None),
                               (make_embedding().vectors, ["ok", "ok", "ok", "zero"])):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                search.build_carrier_embedding(vectors, flags)

    def test_canonical_encoder_patch_targets_new_module(self):
        embedding = make_embedding()
        values = (embedding.vectors, np.ones(4) * 5, np.ones(4), np.ones(4, dtype=int) * 20, ["ok"] * 4)
        with patch("search_logic.encode_batch", return_value=values) as encoder:
            result = search.embed_carriers(["uno", "dos", "tres", "cuatro"], "tokenizer", "model", "cpu")
        encoder.assert_called_once_with(["uno", "dos", "tres", "cuatro"], "tokenizer", "model", "cpu", 8192)
        np.testing.assert_array_equal(result.token_counts, [20] * 4)

    def test_scores_keep_centroid_separate(self):
        embedding = make_embedding()
        scores = search.compute_score_matrix(embedding.centroid[None, :], embedding)
        np.testing.assert_allclose(scores, [[1, 0.5, 0.5, 0.5, 0.5]])

    def test_filters_before_candidate_budget(self):
        result = search.select_candidates(make_metadata(), make_scores(), ["arg"], (1900, 1902), [], (0.6, 1), 2, 1)
        self.assertEqual(result.metadata_match_count, 2)
        self.assertEqual(result.similarity_match_count, 2)
        self.assertEqual(result.rows.embedding_row.tolist(), [0, 2])
        result = search.rank_results(make_metadata(), make_scores(), [], (1900, 1903), ["Presidente 1"], (-1, 1), 1)
        self.assertEqual(result.rows.embedding_row.tolist(), [1])

    def test_empty_selections_mean_all(self):
        self.assertEqual(self.candidates().similarity_match_count, 4)

    def test_equal_cosine_ties_use_original_year_president_chunk_row_order(self):
        metadata = make_metadata()
        metadata["anio"] = 1900
        metadata["presidente"] = "same"
        metadata["chunk_id"] = 1
        metadata = metadata.iloc[::-1].reset_index(drop=True)
        result = search.rank_results(metadata, np.ones((4, 5)) * 0.8, [], (1900, 1900), [], (-1, 1), 2)
        self.assertEqual(result.rows.embedding_row.tolist(), [0, 1])

    def test_outside_cosine_top_k_can_enter_final(self):
        candidates = self.candidates().rows
        original = candidates.copy(deep=True)
        output = RerankOutput(tuple(candidates.embedding_row), np.asarray([[0] * 4, [1] * 4, [4] * 4, [3] * 4]), {})
        final = search.apply_rerank(candidates, output, 2)
        self.assertEqual(final.embedding_row.tolist(), [2, 3])
        self.assertEqual(final.retrieval_rank.tolist(), [3, 4])
        self.assertEqual(final.final_rank.tolist(), [1, 2])
        np.testing.assert_array_equal(final[search.SCORE_COLUMNS], candidates.iloc[[2, 3]][search.SCORE_COLUMNS])
        pd.testing.assert_frame_equal(original, candidates)

    def test_mean_four_raw_logits_and_stable_rerank_ties(self):
        candidates = self.candidates().rows
        output = RerankOutput(tuple(candidates.embedding_row), np.asarray([[1, 2, 3, 4]] * 4), {})
        final = search.apply_rerank(candidates, output, 2)
        self.assertEqual(final.embedding_row.tolist(), [0, 1])
        np.testing.assert_array_equal(final.rerank_score, [2.5, 2.5])

    def test_empty_and_fewer_than_k(self):
        empty = search.select_candidates(make_metadata(), make_scores(), [], (1800, 1801), [], (-1, 1), 10, 5)
        self.assertTrue(empty.rows.empty)
        final = search.apply_rerank(empty.rows, RerankOutput((), np.empty((0, 4)), {}), 5)
        self.assertTrue(final.empty)
        result = search.select_candidates(make_metadata(), make_scores(), [], (1900, 1903), [], (-1, 1), 100, 50)
        self.assertEqual(len(result.rows), 4)

    def test_invalid_budgets(self):
        for n, k in ((1, 2), (1001, 1), (0, 1), (100, 501), (100, 0), (2.5, 1), (True, 1)):
            with self.subTest(n=n, k=k), self.assertRaises(ValueError):
                self.candidates(n, k)

    def test_invalid_cosine_or_matrix(self):
        for cosine in ((0.7, 0.2), (-1.1, 1), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                search.rank_results(make_metadata(), make_scores(), [], (1900, 1903), [], cosine, 1)
        with self.assertRaises(ValueError):
            search.rank_results(make_metadata(), np.zeros((4, 4)), [], (1900, 1903), [], (-1, 1), 1)

    def test_missing_nonfinite_misaligned_scores_rejected(self):
        candidates = self.candidates().rows
        for ids, values in (((1, 0, 2, 3), np.ones((4, 4))), ((0, 1, 2, 3), np.ones((3, 4))),
                            ((0, 1, 2, 3), np.full((4, 4), float("nan")))):
            with self.assertRaises(ValueError):
                search.apply_rerank(candidates, RerankOutput(ids, values, {}), 2)

    def test_duplicate_candidate_identity_rejected(self):
        metadata = make_metadata()
        metadata.loc[1, "embedding_row"] = 0
        with self.assertRaises(ValueError):
            search.rank_results(metadata, make_scores(), [], (1900, 1903), [], (-1, 1), 1)

    def test_signature_ignores_k_page_and_draft_but_tracks_scores_filters_config(self):
        candidates = self.candidates().rows
        filters = dict(countries=[], presidents=[], year_range=[1900, 1903], cosine_range=[-1, 1], top_k=2, page_size=25)
        key = search.candidate_signature("corpus", ["a"] * 4, candidates, filters, 4, {})
        changed = {**filters, "top_k": 3, "page_size": 50}
        self.assertEqual(key, search.candidate_signature("corpus", ["a"] * 4, candidates, changed, 4, {}))
        for field, value in (("countries", ["arg"]), ("cosine_range", [0, 1]), ("year_range", [1901, 1903])):
            self.assertNotEqual(key, search.candidate_signature("corpus", ["a"] * 4, candidates, {**filters, field: value}, 4, {}))
        modified = candidates.copy()
        modified.loc[0, "similitud_carrier"] = 0.1
        self.assertNotEqual(key, search.candidate_signature("corpus", ["a"] * 4, modified, filters, 4, {}))
        self.assertNotEqual(key, search.candidate_signature("corpus2", ["a"] * 4, candidates, filters, 4, {}))
        self.assertNotEqual(key, search.candidate_signature("corpus", ["b"] * 4, candidates, filters, 4, {}))
        self.assertNotEqual(key, search.candidate_signature("corpus", ["a"] * 4, candidates, filters, 4, {"max_length": 128}))

    def test_exports_applied_pending_error_disabled_all_rows_same_order(self):
        candidates = self.candidates().rows
        output = RerankOutput(tuple(candidates.embedding_row), np.asarray([[0] * 4, [1] * 4, [4] * 4, [3] * 4]),
                              {"evaluated_candidates": 4, "pair_evaluations": 16, "truncated_pairs": 0, "elapsed_seconds": 0.1})
        results = search.apply_rerank(candidates, output, 3)
        fallback = search.rank_results(make_metadata(), make_scores(), [], (1900, 1903), [], (-1, 1), 3).rows
        for state in ("applied", "pending", "error", "disabled"):
            with self.subTest(state=state):
                info = reranker_metadata(RerankConfig().metadata(), state, 4, 4, 3, output if state == "applied" else None)
                metadata = search.build_export_metadata(["a", "b", "c", "d"], make_embedding(), make_bundle(4).manifest,
                                                        {}, "cpu", "fp32", 4, 3, info)
                data = search.build_exports(results, metadata, cosine_results=fallback)
                import io
                csv = pd.read_csv(io.BytesIO(data["csv"]))
                expected_ids = [2, 3, 1] if state == "applied" else [0, 1, 2]
                self.assertEqual(csv.embedding_row.tolist(), expected_ids)
                text = data["txt"].decode()
                positions = [text.index(f"embedding_row: {i}\n") for i in expected_ids]
                self.assertEqual(positions, sorted(positions))
                self.assertEqual(len(csv), 3)
                self.assertEqual("rerank_score" in csv, state == "applied")
                self.assertEqual("Logit promedio:" in data["txt"].decode(), state == "applied")
                exported = json.loads(data["json"])
                self.assertEqual(exported["reranker"]["effective_method"], "reranker" if state == "applied" else "cosine")
                self.assertEqual(exported["model"]["id"], "BAAI/bge-m3")
                self.assertEqual(exported["reranker"]["model_id"], "BAAI/bge-reranker-v2-m3")

    def test_fallback_exports_reject_missing_or_reranked_provenance(self):
        candidates = self.candidates().rows
        output = RerankOutput(tuple(candidates.embedding_row), np.asarray([[0] * 4, [1] * 4, [4] * 4, [3] * 4]), {})
        results = search.apply_rerank(candidates, output, 3)
        metadata = {"reranker": {"state": "pending", "effective_method": "cosine"}, "returned_count": 3}
        with self.assertRaises(ValueError):
            search.build_exports(results, metadata)
        with self.assertRaises(ValueError):
            search.build_exports(results, metadata, cosine_results=results)

    def test_year_range_rejects_fractional_nonfinite_boolean_or_reversed_endpoints(self):
        for years in ((1900.9, 1900.1), (1900, 1901.5), (True, 1900), (1900, np.bool_(True)),
                      (float("nan"), 1900), (1900, float("inf")), (1901, 1900), ("1900", 1901), (1900,)):
            with self.subTest(years=years), self.assertRaises(ValueError):
                search.validate_search_controls((-1, 1), 1, years)

    def test_year_range_accepts_integral_numeric_boundaries_including_numpy(self):
        for years in ((1900, 1900), (1900.0, 1903.0), (np.int64(1900), np.int32(1903)),
                      (np.float64(1900), np.float32(1903))):
            with self.subTest(years=years):
                control = search.validate_search_controls((-1, 1), 1, years)
                self.assertEqual(control[-2:], tuple(map(int, years)))
                rows = search.rank_results(make_metadata(), make_scores(), [], years, [], (-1, 1), 4).rows
                self.assertEqual(rows.embedding_row.tolist(), [0] if years[0] == years[1] else [0, 1, 2, 3])

    def test_synthetic_artifact_contract_is_readonly_and_rejects_changed_fingerprint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent / ".cache/tmp") as folder:
            root = Path(folder)
            metadata = make_metadata()
            metadata["embedding_norm"] = 1.0
            metadata["embedding_norm_normalized"] = 1.0
            metadata["n_tokens_chunk"] = 5
            metadata["embedding_flag"] = "ok"
            matrix = make_embedding().vectors
            metadata.to_csv(root / search.METADATA_NAME, index=False)
            np.save(root / search.MATRIX_NAME, matrix)
            config = {"model_id": search.MODEL_ID, "model_revision": search.MODEL_REVISION,
                      "embedding_dimension": search.EMBEDDING_DIM,
                      "pooling": "CLS (last_hidden_state[:, 0, :])", "prefixes": "ninguno",
                      "max_length": search.MAX_LENGTH, "precision": "fp16", "rows": 4,
                      "input_sha256": search.EXPECTED_INPUT_SHA256, "source_rows": search.EXPECTED_INPUT_ROWS,
                      "official_pooling_config": {"word_embedding_dimension": search.EMBEDDING_DIM,
                                                  "pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}}
            manifest = {"status": "complete", "compatibility_config": config,
                        "compatibility_fingerprint": search.config_fingerprint(config),
                        "artifact_checksums_sha256": {n: search.sha256_file(root / n)
                                                      for n in (search.MATRIX_NAME, search.METADATA_NAME)}}
            (root / search.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
            bundle = search.load_artifacts(root, require_full=False)
            try:
                self.assertFalse(bundle.embeddings.flags.writeable)
                manifest["compatibility_fingerprint"] = "wrong"
                with self.assertRaises(search.ArtifactError):
                    search.validate_manifest(manifest, require_full=False)
                with self.assertRaises(search.ArtifactError):
                    search.validate_manifest(bundle.manifest, require_full=True)
            finally:
                # Windows cannot remove the temporary NPY while its mapping is open.
                bundle.embeddings._mmap.close()

    def test_metadata_norms_and_matrix_norms_are_validated(self):
        matrix = make_embedding().vectors.copy()
        matrix[0, 0] = 2.0
        with self.assertRaises(search.ArtifactError):
            search._validate_matrix(matrix, 4)
        with self.assertRaises(search.ArtifactError):
            search._validate_metadata(make_metadata(), 4, 8192)


if __name__ == "__main__":
    unittest.main()
