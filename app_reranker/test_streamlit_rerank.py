"""AppTest interaction coverage. Browser dataframe row selection is not simulated."""
from runtime import APP_ROOT, configure_runtime
configure_runtime()

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from rerank_logic import RerankOutput
from test_support import make_bundle, make_embedding, make_scores


class StreamlitRerankTests(unittest.TestCase):
    def setUp(self):
        self.bundle = make_bundle()
        self.query_resource = SimpleNamespace(tokenizer="mock-tokenizer", model="mock-model", lock=threading.RLock())
        self.patchers = [
            patch("search_logic.manifest_signature", return_value="synthetic-stat"),
            patch("app_resources.load_artifacts_cached", return_value=self.bundle),
            patch("app_resources.load_query_model_cached", return_value=self.query_resource),
            patch("search_logic.embed_carriers", return_value=make_embedding()),
            patch("search_logic.compute_score_matrix", return_value=make_scores(120)),
            patch("app_resources.load_reranker_cached", return_value=object()),
            patch("rerank_logic.score_candidates", side_effect=self.score),
            patch("rerank_logic.AutoTokenizer.from_pretrained", side_effect=AssertionError("Network/model load forbidden in tests")),
            patch("rerank_logic.AutoModelForSequenceClassification.from_pretrained", side_effect=AssertionError("Model load forbidden in tests")),
        ]
        self.mocks = [p.start() for p in self.patchers]
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patchers)])
        self.at = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=30)
        self.at.run()
        self.assert_clean()

    def score(self, resource, carriers, ids, passages):
        values = np.repeat(np.arange(len(ids), dtype=np.float64)[:, None], 4, axis=1)
        return RerankOutput(tuple(ids), values, {"evaluated_candidates": len(ids),
                            "pair_evaluations": 4 * len(ids), "truncated_pairs": 0, "elapsed_seconds": 0.01})

    def assert_clean(self):
        self.assertEqual(len(self.at.exception), 0, [str(e) for e in self.at.exception])

    def submit(self):
        next(button for button in self.at.button if button.label == "Analizar cuatro carriers").click().run()
        self.assert_clean()

    def enable_apply(self):
        self.at.toggle(key="reranker_enabled").set_value(True).run()
        self.at.button(key="apply_reranker").click().run()
        self.assert_clean()

    def use_large_budget(self):
        """Pagination/export stress tests opt in to K100/N200; not the default preset."""
        self.at.slider(key="top_k").set_value(100)
        self.at.number_input(key="candidate_n").set_value(200)
        self.at.run()
        self.assert_clean()

    def assert_rendered_page_and_detail(self, expected_ids, applied, candidate_count=120):
        """Check emitted AppTest elements, not the session result snapshots.

        This covers the default first-row detail on each page, not browser clicks.
        """
        tables = [node.value for node in self.at.dataframe if "embedding_row" in node.value.columns]
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.embedding_row.tolist(), expected_ids)
        expected_metadata = self.bundle.metadata.iloc[expected_ids]
        self.assertEqual(table["Año"].tolist(), expected_metadata.anio.tolist())
        self.assertEqual(table["Presidente"].tolist(), expected_metadata.presidente.tolist())
        np.testing.assert_allclose(table["Coseno del centroide"], make_scores(120)[expected_ids, 0])
        self.assertEqual("Logit promedio" in table, applied)
        self.assertEqual("Puesto original" in table, applied)
        self.assertEqual("Puesto final" in table, applied)
        first_id = expected_ids[0]
        self.assertTrue(any(node.value == f"Texto sintético {first_id}" for node in self.at.markdown))
        self.assertTrue(any(node.value.endswith(f"embedding_row {first_id}") for node in self.at.caption))
        if applied:
            self.assertEqual(table["Logit promedio"].tolist(), expected_ids)
            self.assertEqual(table["Puesto original"].tolist(), [i + 1 for i in expected_ids])
            self.assertEqual(table["Puesto final"].tolist(), [candidate_count - i for i in expected_ids])
            self.assertTrue(any(f"**Logit promedio:** {first_id:.6f}" in node.value for node in self.at.markdown))
            self.assertTrue(any(f"**Puesto original:** {first_id + 1}" in node.value and
                                f"**Puesto final:** {candidate_count - first_id}" in node.value for node in self.at.markdown))
            profiles = [node.value for node in self.at.dataframe if list(node.value.columns) == ["Carrier", "Logit"]]
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0]["Logit"].tolist(), [first_id] * 4)
        else:
            self.assertFalse(any("**Logit promedio:**" in node.value for node in self.at.markdown))

    def test_no_models_before_submit_or_apply_and_toggle_filters_do_not_infer(self):
        self.mocks[2].assert_not_called()
        self.mocks[5].assert_not_called()
        self.mocks[6].assert_not_called()
        self.at.toggle(key="reranker_enabled").set_value(True).run()
        self.mocks[5].assert_not_called()
        self.submit()
        self.assertEqual(self.mocks[2].call_count, 1)
        self.mocks[6].assert_not_called()
        self.at.multiselect(key="countries").set_value(["arg"]).run()
        self.at.selectbox(key="page_size").set_value(10).run()
        self.assert_clean()
        self.assertEqual(self.mocks[2].call_count, 1)
        self.mocks[5].assert_not_called()
        self.mocks[6].assert_not_called()

    def test_apply_all_candidates_then_k_and_exports_same_order(self):
        self.use_large_budget()
        self.submit()
        self.enable_apply()
        self.assertEqual(self.mocks[6].call_count, 1)
        results = self.at.session_state["final_results"]
        self.assertEqual(len(results), 100)
        self.assertEqual(results.embedding_row.iloc[0], 119)
        self.assertIn("rerank_score", results)
        metadata = self.at.session_state["export_metadata"]
        self.assertEqual(metadata["reranker"]["effective_n"], 120)
        self.assertEqual(metadata["reranker"]["pair_evaluations"], 480)
        import io
        exported = pd.read_csv(io.BytesIO(self.at.session_state["exports"]["csv"]))
        self.assertEqual(exported.embedding_row.tolist(), results.embedding_row.tolist())
        self.assertEqual(len(exported), 100)
        self.assert_rendered_page_and_detail(list(range(119, 94, -1)), True)

    def test_page_size_pagination_k_toggle_and_rerun_reuse_scores(self):
        self.use_large_budget()
        self.submit()
        self.enable_apply()
        self.at.number_input(key="results_page").set_value(2).run()
        self.assert_clean()
        self.assert_rendered_page_and_detail(list(range(94, 69, -1)), True)
        self.at.selectbox(key="page_size").set_value(10).run()
        self.assert_clean()
        self.assert_rendered_page_and_detail(list(range(109, 99, -1)), True)
        self.at.slider(key="top_k").set_value(50).run()
        self.assert_clean()
        self.assertEqual(self.at.session_state["export_metadata"]["reranker"]["state"], "applied")
        self.assertEqual(len(self.at.session_state["final_results"]), 50)
        self.assert_rendered_page_and_detail(list(range(119, 109, -1)), True)
        self.at.toggle(key="reranker_enabled").set_value(False).run()
        self.assert_clean()
        self.assertNotIn("rerank_score", self.at.session_state["final_results"])
        self.assertEqual(self.at.session_state["export_metadata"]["reranker"]["effective_method"], "cosine")
        self.assert_rendered_page_and_detail(list(range(10)), False)
        self.at.toggle(key="reranker_enabled").set_value(True).run()
        self.at.run()
        self.assert_clean()
        self.assertEqual(self.mocks[6].call_count, 1)
        self.assertEqual(self.mocks[2].call_count, 1)
        self.assert_rendered_page_and_detail(list(range(119, 109, -1)), True)

    def test_rendered_cosine_table_and_detail_before_explicit_apply(self):
        self.submit()
        self.assert_rendered_page_and_detail(list(range(10)), False)
        self.at.toggle(key="reranker_enabled").set_value(True).run()
        self.assert_clean()
        self.assert_rendered_page_and_detail(list(range(10)), False)
        self.mocks[6].assert_not_called()

    def test_filters_invalidate_and_reset_navigation_without_inference(self):
        self.use_large_budget()
        self.submit()
        self.enable_apply()
        self.at.number_input(key="results_page").set_value(2).run()
        self.at.multiselect(key="countries").set_value(["arg"]).run()
        self.assert_clean()
        self.assertEqual(self.at.session_state["results_page"], 1)
        self.assertNotIn("rerank_score", self.at.session_state["final_results"])
        self.assertEqual(self.at.session_state["export_metadata"]["reranker"]["state"], "pending")
        self.assertEqual(self.mocks[6].call_count, 1)

    def test_cosine_budget_and_config_changes_invalidate(self):
        for kind in ("cosine", "candidate_n", "max_length"):
            with self.subTest(kind=kind):
                self.submit()
                if not self.at.toggle(key="reranker_enabled").value:
                    self.at.toggle(key="reranker_enabled").set_value(True).run()
                self.at.button(key="apply_reranker").click().run()
                count = self.mocks[6].call_count
                if kind == "cosine":
                    self.at.slider(key="cosine").set_value((0.55, 1.0)).run()
                elif kind == "candidate_n":
                    self.at.number_input(key="candidate_n").set_value(210).run()
                elif kind == "max_length":
                    self.at.selectbox(key="max_length").set_value(256).run()
                self.assert_clean()
                self.assertEqual(self.mocks[6].call_count, count)
                self.assertEqual(self.at.session_state["export_metadata"]["reranker"]["state"], "pending")
                self.assertNotIn("rerank_score", self.at.session_state["final_results"])

    def test_conservative_defaults_are_local_explicit_and_evaluate_four_pairs_per_candidate(self):
        self.assertEqual(self.at.slider(key="top_k").value, 10)
        self.assertEqual(self.at.number_input(key="candidate_n").value, 25)
        self.assertFalse(any(toggle.key == "offline" for toggle in self.at.toggle))
        self.assertTrue(any("13 min" in caption.value and "No es una ETA" in caption.value for caption in self.at.caption))
        self.mocks[2].assert_not_called()
        self.mocks[6].assert_not_called()
        self.submit()
        self.assertEqual(len(self.at.session_state["final_results"]), 10)
        self.enable_apply()
        self.assert_rendered_page_and_detail(list(range(24, 14, -1)), True, candidate_count=25)
        info = self.at.session_state["export_metadata"]["reranker"]
        self.assertEqual((info["requested_n"], info["evaluated_candidates"], info["pair_evaluations"]), (25, 25, 100))
        self.assertTrue(info["local_files_only"])
        self.assertEqual(self.mocks[6].call_count, 1)

    def test_manual_n_below_k_errors_without_clamping_or_inference(self):
        self.submit()
        self.at.toggle(key="reranker_enabled").set_value(True)
        self.at.slider(key="top_k").set_value(50)
        self.at.number_input(key="candidate_n").set_value(25)
        self.at.run()
        self.assert_clean()
        self.assertEqual(self.at.number_input(key="candidate_n").value, 25)
        self.assertTrue(self.at.button(key="apply_reranker").disabled)
        self.assertTrue(any("N debe ser mayor o igual" in error.value for error in self.at.error))
        self.assertEqual(self.at.session_state["export_metadata"]["reranker"]["effective_method"], "cosine")
        self.mocks[6].assert_not_called()

    def test_draft_does_not_replace_analyzed_carriers_new_analysis_invalidates(self):
        self.submit()
        self.enable_apply()
        active = self.at.session_state["analysis"]["carriers"]
        self.at.text_area(key="carrier_1").set_value("Un borrador diferente.").run()
        self.assert_clean()
        self.assertEqual(self.at.session_state["analysis"]["carriers"], active)
        self.assertEqual(self.at.session_state["export_metadata"]["reranker"]["state"], "applied")
        self.submit()
        self.assertNotEqual(self.at.session_state["analysis"]["carriers"], active)
        self.assertEqual(self.at.session_state["export_metadata"]["reranker"]["state"], "pending")
        self.assertEqual(self.mocks[6].call_count, 1)

    def test_failed_analysis_retains_labelled_previous_query_without_scores(self):
        self.submit()
        self.enable_apply()
        active = self.at.session_state["analysis"]["carriers"]
        self.mocks[3].side_effect = RuntimeError("synthetic failed new analysis")
        self.submit()
        self.assertEqual(self.at.session_state["analysis"]["carriers"], active)
        self.assertTrue(self.at.session_state["export_metadata"]["analysis"]["using_previous_after_failure"])
        self.assertNotIn("rerank_score", self.at.session_state["final_results"])
        self.assertTrue(any("consulta anterior" in warning.value for warning in self.at.warning))

    def test_inference_error_no_stale_or_partial_scores_and_no_auto_retry(self):
        self.submit()
        self.enable_apply()
        self.mocks[6].side_effect = RuntimeError("synthetic rerank failure")
        self.at.button(key="apply_reranker").click().run()
        self.assert_clean()
        info = self.at.session_state["export_metadata"]["reranker"]
        self.assertEqual(info["state"], "error")
        self.assertEqual(info["effective_method"], "cosine")
        self.assertEqual(info["evaluated_candidates"], 0)
        self.assertNotIn("rerank_score", self.at.session_state["final_results"])
        self.at.selectbox(key="page_size").set_value(10).run()
        self.at.run()
        self.assert_clean()
        self.assertEqual(self.mocks[6].call_count, 2)
        self.assertNotIn(b"rerank_score", self.at.session_state["exports"]["csv"])

    def test_loader_error_no_inference_and_truthful_error_exports(self):
        self.submit()
        self.mocks[5].side_effect = OSError("synthetic missing offline model")
        self.enable_apply()
        self.mocks[6].assert_not_called()
        exported = json.loads(self.at.session_state["exports"]["json"])
        self.assertEqual(exported["reranker"]["state"], "error")
        self.assertEqual(exported["reranker"]["effective_method"], "cosine")

    def test_no_matches_no_apply_or_stale_scores(self):
        self.submit()
        self.enable_apply()
        self.at.slider(key="cosine").set_value((-1.0, 0.0)).run()
        self.assert_clean()
        self.assertTrue(self.at.button(key="apply_reranker").disabled)
        self.assertEqual(len(self.at.session_state["final_results"]), 0)
        self.assertEqual(self.at.session_state["export_metadata"]["returned_count"], 0)
        self.assertEqual(self.mocks[6].call_count, 1)


if __name__ == "__main__":
    unittest.main()
