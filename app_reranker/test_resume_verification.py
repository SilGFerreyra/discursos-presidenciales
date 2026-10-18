"""Guard tests only; no model weights, downloads or app inference."""
import unittest
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from unittest.mock import patch
from runtime import APP_ROOT, configure_runtime
configure_runtime()
import resume_verification as verify


class ResumeGuardTests(unittest.TestCase):
    def test_offline_flags_are_active_at_library_import(self):
        from huggingface_hub import constants
        from transformers.utils import is_offline_mode
        self.assertTrue(constants.HF_HUB_OFFLINE)
        self.assertTrue(is_offline_mode())

    def test_owned_output_rejects_original_sibling(self):
        with self.assertRaises(ValueError):
            verify.owned(APP_ROOT.parent / "app" / "forbidden.json")

    def test_offline_observer_rejects_network(self):
        import requests
        report = {}
        with self.assertRaises(AssertionError):
            with verify.offline_observers(report):
                requests.get("https://huggingface.co/")
        self.assertEqual(len(report["network_attempts"]), 1)

    def test_bin_observer_rejects_unrestricted_load_before_delegate(self):
        import torch
        path = APP_ROOT / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots" / verify.BGE_REVISION / "pytorch_model.bin"
        with patch.object(torch, "load") as delegate:
            with self.assertRaises(AssertionError):
                with verify.offline_observers({}):
                    torch.load(path, weights_only=False)
            delegate.assert_not_called()

    def test_asyncio_loopback_exception_does_not_allow_external_sockets(self):
        import socket
        report = {}
        with patch.object(socket.socket, "connect", return_value=None) as delegate:
            with verify.offline_observers(report, allow_asyncio_loopback=True):
                socket.socket.connect(object(), ("127.0.0.1", 12345))
            delegate.assert_called_once()
        self.assertEqual(len(report["asyncio_loopback_connects"]), 1)
        with self.assertRaises(AssertionError):
            with verify.offline_observers({}, allow_asyncio_loopback=True):
                socket.socket.connect(object(), ("192.0.2.1", 443))


if __name__ == "__main__":
    unittest.main()
