"""Fresh-process offline regression probe; fake weights unless --real-model is explicit.

No server or reranker inference is started. --real-tokenizer reads only cached
tokenizer files; --real-model is reserved for the next focused verifier.
"""
import argparse
import builtins
import json
import os
import socket
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runtime import APP_ROOT, configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=["resources", "launcher", "script", "provision", "late"], default="resources")
    parser.add_argument("--real-tokenizer", action="store_true")
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.real_model and args.entry in {"provision", "late"}:
        raise ValueError("A real-model probe requires a production offline entry point")
    report = {"entry": args.entry, "network_attempts": [], "early_imports": [],
              "asyncio_loopback_connects": [],
              "real_tokenizer": args.real_tokenizer or args.real_model,
              "model_weights_loaded": args.real_model, "carrier_inference_executed": False,
              "benchmark_executed": False, "server_started": False}
    import requests

    def forbid(*items, **kwargs):
        report["network_attempts"].append(str(items[1:] if len(items) > 1 else items))
        raise AssertionError("Probe forbids networking before any traffic")

    original_import = builtins.__import__
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    patched_corpus = False

    def connection_guard(original):
        def connect(sock, address):
            # AppTest's Windows asyncio socketpair uses numeric loopback, never model HTTP.
            if (args.entry == "script" and isinstance(address, tuple)
                    and address[0] in {"127.0.0.1", "::1"}):
                report["asyncio_loopback_connects"].append(str(address))
                return original(sock, address)
            return forbid(sock, address)
        return connect

    def observe_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal patched_corpus
        if name.split(".")[0] in {"huggingface_hub", "transformers"} and name not in sys.modules:
            expected = "0" if args.entry in {"provision", "late"} else "1"
            if len(report["early_imports"]) < 12:
                report["early_imports"].append({"module": name, "hf": os.environ.get("HF_HUB_OFFLINE"),
                                                "tf": os.environ.get("TRANSFORMERS_OFFLINE")})
            assert os.environ.get("HF_HUB_OFFLINE") == expected
            assert os.environ.get("TRANSFORMERS_OFFLINE") == expected
        module = original_import(name, globals, locals, fromlist, level)
        if args.entry == "script" and name == "app_resources" and not patched_corpus:
            patched_corpus = True
            from test_support import make_bundle
            module.load_artifacts_cached = Mock(return_value=make_bundle())
        return module

    with ExitStack() as stack:
        stack.enter_context(patch.object(requests.sessions.Session, "request", side_effect=forbid))
        stack.enter_context(patch.object(socket.socket, "connect", new=connection_guard(original_connect)))
        stack.enter_context(patch.object(socket.socket, "connect_ex", new=connection_guard(original_connect_ex)))
        stack.enter_context(patch.object(builtins, "__import__", side_effect=observe_import))
        from launch import validate_interpreter
        validate_interpreter()
        if args.entry == "late":
            import transformers
            from huggingface_hub import constants
            from transformers.utils import is_offline_mode
            assert not constants.HF_HUB_OFFLINE and not is_offline_mode()
            try:
                configure_runtime()
            except RuntimeError:
                report["late_import_rejected"] = True
            else:
                raise AssertionError("Late offline bootstrap must fail closed")
            assert not constants.HF_HUB_OFFLINE and not is_offline_mode()
        elif args.entry == "provision":
            import verify_real_runtime
            def inspect_provision_mode():
                import transformers
                from huggingface_hub import constants
                from transformers.utils import is_offline_mode
                assert not constants.HF_HUB_OFFLINE and not is_offline_mode()
                assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "true"
                report["explicit_online_provision_bootstrap"] = True
            stack.enter_context(patch.object(sys, "argv", ["verify_real_runtime.py", "provision"]))
            stack.enter_context(patch.object(verify_real_runtime, "provision", side_effect=inspect_provision_mode))
            verify_real_runtime.main()
        else:
            if args.entry == "launcher":
                import launch
                stack.enter_context(patch.object(sys, "argv", ["launch.py", "--check"]))
                assert launch.main() == 0
            elif args.entry == "script":
                from streamlit.testing.v1 import AppTest
                app = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=60).run()
                assert not app.exception and not app.error
                assert app.slider(key="top_k").value == 10
                assert app.number_input(key="candidate_n").value == 25
                report["script_defaults"] = {"k": 10, "n": 25}
            import app_resources  # Its own pre-import bootstrap must work without the launcher.
            import transformers
            from huggingface_hub import constants
            from transformers.utils import is_offline_mode
            from search_logic import MODEL_ID, MODEL_REVISION, MAX_LENGTH, embed_carriers
            assert constants.HF_HUB_OFFLINE and is_offline_mode()
            report["hub_offline_cached"] = constants.HF_HUB_OFFLINE
            report["transformers_offline_cached"] = is_offline_mode()
            if not args.real_model:
                dummy = Mock(config=SimpleNamespace(hidden_size=1024))
                dummy.eval.return_value = dummy
                dummy.to.return_value = dummy
                model_load = stack.enter_context(patch.object(transformers.AutoModel, "from_pretrained", return_value=dummy))
            if not args.real_tokenizer and not args.real_model:
                def metadata_sensitive_tokenizer(*items, **kwargs):
                    # Reproduce the known metadata bypass if import-cached mode is not offline.
                    if not is_offline_mode():
                        requests.get("https://huggingface.co/api/models/BAAI/bge-m3")
                    return SimpleNamespace(model_max_length=8192, cls_token_id=0)
                stack.enter_context(patch.object(transformers.AutoTokenizer, "from_pretrained", side_effect=metadata_sensitive_tokenizer))
            resource = app_resources.load_query_model_cached(MODEL_ID, MODEL_REVISION, "cpu", "fp32", MAX_LENGTH)
            if args.real_model:
                import torch
                from benchmark_reranker import DEFAULT_CARRIERS
                with resource.lock:
                    encoded = embed_carriers(DEFAULT_CARRIERS, resource.tokenizer, resource.model, torch.device("cpu"), MAX_LENGTH)
                assert encoded.vectors.shape == (4, 1024)
                report["carrier_inference_executed"] = True
                report["carrier_flags"] = encoded.flags.tolist()
            else:
                assert model_load.call_count == 1
                assert model_load.call_args.kwargs["local_files_only"] is True
                assert model_load.call_args.kwargs["revision"] == MODEL_REVISION
            # The cache must not hide attempts to opt the production loader into online mode.
            try:
                app_resources.load_query_model_cached(MODEL_ID, MODEL_REVISION, "cpu", "fp32", MAX_LENGTH,
                                                      _local_files_only=False)
            except ValueError:
                report["online_resource_request_rejected"] = True
            else:
                raise AssertionError("Production resource must remain offline even on a cache hit")
        assert not report["network_attempts"]
    report["status"] = "passed"
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(APP_ROOT):
            raise ValueError("Probe output must stay under NEWAPP")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("OFFLINE_PROBE=" + json.dumps(report))


if __name__ == "__main__":
    main()
