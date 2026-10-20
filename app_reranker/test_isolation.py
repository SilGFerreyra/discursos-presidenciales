"""Portable runtime, launcher and session-state regressions."""
import importlib
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import runtime
import launch
from rerank_state import invalidate, reconcile, publish, fail, reranker_metadata


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        (runtime.APP_ROOT / ".cache/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=runtime.APP_ROOT / ".cache/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.tempdir = patch.object(tempfile, "tempdir", tempfile.tempdir)
        self.tempdir.start()
        self.addCleanup(self.tempdir.stop)

    def test_import_has_no_directory_creation(self):
        with patch.object(Path, "mkdir", side_effect=AssertionError("Unexpected import-time mkdir")):
            importlib.reload(runtime)

    def test_paths_idempotence_and_bytecode(self):
        with patch.object(runtime, "APP_ROOT", self.root):
            first = runtime.configure_runtime()
            second = runtime.configure_runtime()
        self.assertEqual(first, second)
        for variable, path in first.items():
            self.assertTrue(path.is_relative_to(self.root))
            self.assertTrue(path.is_dir())
            self.assertEqual(os.environ[variable], str(path))
        self.assertTrue(sys.dont_write_bytecode)
        self.assertEqual(os.environ["DISABLE_SAFETENSORS_CONVERSION"], "true")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(tempfile.gettempdir(), str(first["TEMP"]))

    def test_permission_errors_propagate_without_env_publication(self):
        os.environ["HF_HOME"] = "old-value"
        with patch.object(runtime, "APP_ROOT", self.root), patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                runtime.configure_runtime()
        self.assertEqual(os.environ["HF_HOME"], "old-value")

    def test_linked_cache_paths_rejected(self):
        with patch.object(runtime, "APP_ROOT", self.root), patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(RuntimeError):
                runtime.configure_runtime()

    def test_inherited_transformers_cache_removed(self):
        os.environ["TRANSFORMERS_CACHE"] = "outside"
        with patch.object(runtime, "APP_ROOT", self.root):
            runtime.configure_runtime()
        self.assertNotIn("TRANSFORMERS_CACHE", os.environ)

    def test_late_bootstrap_rejects_import_cached_online_mode_without_mutation(self):
        cached = SimpleNamespace(HF_HUB_OFFLINE=False)
        with patch.dict(sys.modules, {"huggingface_hub.constants": cached}):
            with self.assertRaises(RuntimeError):
                runtime.configure_network_mode(offline=True)
        self.assertFalse(cached.HF_HUB_OFFLINE)

    def test_preimport_bootstrap_overrides_inherited_online_flags(self):
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        runtime.configure_network_mode(offline=True)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")


class StateTests(unittest.TestCase):
    def test_signature_change_discards_old_output(self):
        state = {}
        publish(state, "old", object())
        self.assertEqual(reconcile(state, "new", True), ("pending", None))
        self.assertNotIn("rerank_record", state)

    def test_toggle_reuses_matching_complete_result(self):
        state, output = {}, object()
        publish(state, "same", output)
        self.assertEqual(reconcile(state, "same", False), ("disabled", None))
        status, record = reconcile(state, "same", True)
        self.assertEqual(status, "applied")
        self.assertIs(record["output"], output)

    def test_failure_invalidates_output_without_retry(self):
        state = {}
        publish(state, "key", object())
        fail(state, "key", ValueError("failed"))
        status, record = reconcile(state, "key", True)
        self.assertEqual(status, "error")
        self.assertNotIn("output", record)
        self.assertEqual(reconcile(state, "key", True), (status, record))

    def test_invalidate_resets_navigation_and_scores(self):
        state = {"results_page": 3, "selected_embedding_row": 42, "display_signature": "old",
                 "exports": "old", "final_results": "old", "export_metadata": "old"}
        publish(state, "key", object())
        invalidate(state)
        self.assertEqual(state["results_page"], 1)
        self.assertIsNone(state["selected_embedding_row"])
        for key in ("rerank_record", "display_signature", "exports", "final_results", "export_metadata"):
            self.assertNotIn(key, state)

    def test_fallback_metadata_never_reports_previous_evaluations(self):
        for status in ("pending", "error", "disabled"):
            result = reranker_metadata({"model_id": "pinned"}, status, 100, 50, 10)
            self.assertEqual(result["effective_method"], "cosine")
            self.assertEqual(result["evaluated_candidates"], 0)
            self.assertIsNone(result["elapsed_seconds"])


class LauncherTests(unittest.TestCase):
    def setUp(self):
        (runtime.APP_ROOT / ".cache/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=runtime.APP_ROOT / ".cache/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / ".venv"
        self.root.mkdir()
        (self.root / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")

    def isolated_interpreter(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(runtime, "APP_ROOT", self.root.parent))
        executable = self.root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        stack.enter_context(patch.object(sys, "executable", str(executable)))
        stack.enter_context(patch.object(sys, "prefix", str(self.root)))
        stack.enter_context(patch.object(sys, "base_prefix", str(self.root.parent / "system-python")))
        stack.enter_context(patch.object(sys, "path", [str(self.root / "Lib/site-packages")]))
        stack.enter_context(patch("launch.site.ENABLE_USER_SITE", False))
        return stack

    def test_app_local_environment_accepted(self):
        with self.isolated_interpreter():
            self.assertEqual(launch.validate_interpreter(), self.root)

    def test_environment_variable_cannot_redirect_interpreter(self):
        with patch.dict(os.environ, {"APP_RERANKER_VENV": str(self.root / "foreign")}):
            self.assertEqual(runtime.get_venv_root(), runtime.APP_ROOT / ".venv")

    def test_linked_virtual_environment_rejected(self):
        with patch.object(Path, "is_symlink", return_value=True), self.assertRaises(RuntimeError):
            runtime.get_venv_root()

    def test_foreign_prefix_user_site_and_package_paths_rejected(self):
        with self.isolated_interpreter():
            for target, value in (("prefix", str(self.root.parent)),
                                  ("path", [str(self.root.parent / "foreign/site-packages")])):
                with patch.object(sys, target, value), self.assertRaises(RuntimeError):
                    launch.validate_interpreter()
            with patch("launch.site.ENABLE_USER_SITE", True), self.assertRaises(RuntimeError):
                launch.validate_interpreter()

    def test_skips_occupied_port_without_stopping_processes(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            if port == 65535:
                self.skipTest("Ephemeral port cannot be incremented")
            self.assertNotEqual(launch.find_unused_port(port, min(port + 20, 65536)), port)

    def test_system_interpreter_rejected(self):
        with patch.object(sys, "executable", str(self.root.parent / "python")), self.assertRaises(RuntimeError):
            launch.validate_interpreter()

    def test_no_free_port_error(self):
        with self.assertRaises(RuntimeError):
            launch.find_unused_port(8502, 8502)


if __name__ == "__main__":
    unittest.main()
