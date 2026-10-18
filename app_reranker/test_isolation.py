"""Standard-library-only isolation, inventory, launcher and session-state tests."""
import importlib
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import inventory
import runtime
import launch
from rerank_state import invalidate, reconcile, publish, fail, reranker_metadata


class RuntimeTests(unittest.TestCase):
    def setUp(self):
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
        self.assertEqual(os.environ["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(os.environ["DISABLE_SAFETENSORS_CONVERSION"], "true")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(tempfile.gettempdir(), str(first["TEMP"]))

    def test_permission_errors_propagate_no_fallback_or_env_publication(self):
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

    def test_conversion_disabled_overrides_inherited_flag_and_is_idempotent(self):
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "false"
        with patch.object(runtime, "APP_ROOT", self.root):
            first = runtime.configure_runtime()
            second = runtime.configure_runtime()
        self.assertEqual(first, second)
        self.assertEqual(os.environ["DISABLE_SAFETENSORS_CONVERSION"], "true")

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


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runtime.APP_ROOT / ".cache/tmp")
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.excluded = self.parent / "app_reranker"
        self.excluded.mkdir()

    def test_excludes_only_new_app_and_hashes_original_sources(self):
        (self.parent / "original.py").write_text("original", encoding="utf-8")
        (self.excluded / "not-covered.py").write_text("new", encoding="utf-8")
        entries = inventory.snapshot(self.parent, self.excluded)
        self.assertEqual(list(entries), ["original.py"])
        self.assertIn("sha256", entries["original.py"])

    def test_walk_error_is_fail_closed(self):
        def broken_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("traversal denied"))
            return iter(())
        with patch("inventory.os.walk", side_effect=broken_walk):
            with self.assertRaises(PermissionError):
                inventory.snapshot(self.parent, self.excluded)

    def test_stat_failure_is_not_silently_skipped(self):
        (self.parent / "file.py").write_text("test", encoding="utf-8")
        with patch.object(Path, "stat", side_effect=PermissionError("stat denied")):
            with self.assertRaises(PermissionError):
                inventory.snapshot(self.parent, self.excluded)

    def test_protected_links_rejected_not_followed(self):
        (self.parent / "file.py").write_text("test", encoding="utf-8")
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(RuntimeError):
                inventory.snapshot(self.parent, self.excluded)

    def test_baseline_refusal_occurs_before_traversal(self):
        with patch.object(sys, "argv", ["inventory.py", "before"]), patch("inventory.snapshot") as snapshot:
            with self.assertRaises(RuntimeError):
                inventory.main()
        snapshot.assert_not_called()


class StateTests(unittest.TestCase):
    def test_signature_change_discards_old_output(self):
        state = {}
        publish(state, "old", object())
        self.assertEqual(reconcile(state, "new", True), ("pending", None))
        self.assertNotIn("rerank_record", state)

    def test_toggle_does_not_infer_or_discard_matching_complete_result(self):
        state = {}
        output = object()
        publish(state, "same", output)
        self.assertEqual(reconcile(state, "same", False), ("disabled", None))
        status, record = reconcile(state, "same", True)
        self.assertEqual(status, "applied")
        self.assertIs(record["output"], output)

    def test_failure_replaces_complete_output_and_persists_without_retry(self):
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
        self.assertNotIn("rerank_record", state)
        self.assertNotIn("display_signature", state)
        self.assertNotIn("exports", state)
        self.assertNotIn("final_results", state)
        self.assertNotIn("export_metadata", state)

    def test_fallback_export_metadata_never_reports_previous_evaluations(self):
        for status in ("pending", "error", "disabled"):
            result = reranker_metadata({"model_id": "pinned"}, status, 100, 50, 10)
            self.assertEqual(result["effective_method"], "cosine")
            self.assertEqual(result["evaluated_candidates"], 0)
            self.assertIsNone(result["elapsed_seconds"])


class LauncherTests(unittest.TestCase):
    def isolated_interpreter(self, root):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, {"APP_RERANKER_VENV": str(root)}))
        stack.enter_context(patch.object(sys, "executable", str(root / "Scripts/python.exe")))
        stack.enter_context(patch.object(sys, "prefix", str(root)))
        stack.enter_context(patch.object(sys, "base_prefix", r"C:\Users\Usuario\AppData\Local\Programs\Python\Python312"))
        stack.enter_context(patch.object(sys, "path", [str(root / "Lib/site-packages")]))
        stack.enter_context(patch("launch.site.ENABLE_USER_SITE", False))
        return stack

    def test_authorized_external_own_environment_accepted(self):
        root = runtime.LOCAL_ENV_ROOT / ".venv"
        with self.isolated_interpreter(root):
            self.assertEqual(launch.validate_interpreter(), root)

    def test_app_local_default_own_environment_still_accepted(self):
        root = runtime.APP_ROOT / ".venv"
        with self.isolated_interpreter(root):
            self.assertEqual(launch.validate_interpreter(), root)

    def test_original_and_system_environment_configuration_rejected(self):
        for root in (runtime.APP_ROOT.parent / "app/.venv",
                     Path(r"C:\Users\Usuario\AppData\Local\Programs\Python\Python312")):
            with patch.dict(os.environ, {"APP_RERANKER_VENV": str(root)}):
                with self.assertRaises(RuntimeError):
                    runtime.get_venv_root()

    def test_wrong_prefix_user_site_and_foreign_package_paths_rejected(self):
        root = runtime.LOCAL_ENV_ROOT / ".venv"
        with self.isolated_interpreter(root):
            with patch.object(sys, "prefix", str(runtime.APP_ROOT.parent / "app/.venv")):
                with self.assertRaises(RuntimeError):
                    launch.validate_interpreter()
            with patch("launch.site.ENABLE_USER_SITE", True):
                with self.assertRaises(RuntimeError):
                    launch.validate_interpreter()
            with patch.object(sys, "path", [str(runtime.APP_ROOT.parent / "app/.venv/Lib/site-packages")]):
                with self.assertRaises(RuntimeError):
                    launch.validate_interpreter()

    def test_batch_rejects_original_environment_before_execution(self):
        for root in (runtime.APP_ROOT.parent / "app/.venv",
                     Path(r"C:\Users\Usuario\AppData\Local\Programs\Python\Python312")):
            with self.subTest(root=root):
                environment = {**os.environ, "APP_RERANKER_VENV": str(root)}
                result = subprocess.run(["cmd.exe", "/d", "/c", str(runtime.APP_ROOT / "ejecutar_app.bat"), "--check"],
                                        env=environment, cwd=runtime.APP_ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 1)
                self.assertIn("no es una ubicacion propia autorizada", result.stdout)

    def test_skips_occupied_port_without_terminating_anything(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            if port == 65535:
                self.skipTest("Ephemeral port cannot be incremented")
            self.assertNotEqual(launch.find_unused_port(port, min(port + 20, 65536)), port)

    def test_requires_own_python(self):
        with patch.object(sys, "executable", "C:/outside/python.exe"):
            with self.assertRaises(RuntimeError):
                launch.validate_interpreter()

    def test_no_free_port_error(self):
        with self.assertRaises(RuntimeError):
            launch.find_unused_port(8502, 8502)


if __name__ == "__main__":
    unittest.main()
