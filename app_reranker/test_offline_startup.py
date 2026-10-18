"""Fresh subprocess regressions; no real weights, server, downloads or benchmark."""
import json
import os
import subprocess
import sys
import unittest

from runtime import APP_ROOT


class OfflineStartupTests(unittest.TestCase):
    def probe(self, entry):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                       "HF_HUB_OFFLINE": "1" if entry == "provision" else "0",
                       "TRANSFORMERS_OFFLINE": "1" if entry == "provision" else "0"}
        result = subprocess.run([sys.executable, "-B", str(APP_ROOT / "offline_startup_probe.py"), "--entry", entry],
                                cwd=APP_ROOT, env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [line[len("OFFLINE_PROBE="):] for line in result.stdout.splitlines() if line.startswith("OFFLINE_PROBE=")]
        self.assertEqual(len(records), 1)
        report = json.loads(records[0])
        self.assertEqual(report["network_attempts"], [])
        self.assertFalse(report["model_weights_loaded"])
        self.assertFalse(report["carrier_inference_executed"])
        self.assertFalse(report["server_started"])
        self.assertGreater(len(report["early_imports"]), 0)
        return report

    def test_direct_resource_bootstrap_precedes_imports_and_query_network_guard(self):
        report = self.probe("resources")
        self.assertTrue(report["hub_offline_cached"] and report["transformers_offline_cached"])
        self.assertTrue(report["online_resource_request_rejected"])

    def test_launcher_check_precedes_hf_transformers_imports(self):
        report = self.probe("launcher")
        self.assertTrue(report["hub_offline_cached"] and report["transformers_offline_cached"])

    def test_direct_app_script_precedes_imports_and_keeps_new_defaults(self):
        report = self.probe("script")
        self.assertEqual(report["script_defaults"], {"k": 10, "n": 25})
        self.assertTrue(report["hub_offline_cached"])

    def test_provision_cli_is_explicitly_online_before_imports_without_conversion(self):
        self.assertTrue(self.probe("provision")["explicit_online_provision_bootstrap"])

    def test_late_offline_switch_is_rejected_without_patching_library_globals(self):
        self.assertTrue(self.probe("late")["late_import_rejected"])


if __name__ == "__main__":
    unittest.main()
