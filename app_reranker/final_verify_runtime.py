"""Focused real default-startup proof. Observers never set offline flags or fake models."""
from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from runtime import APP_ROOT  # Importing the constants does not bootstrap network mode.

EVIDENCE = APP_ROOT / ".cache/evaluation"
BGE_PIN = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_PIN = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def save(name, value):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def inherited():
    flags = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    assert all(value in (None, "0") for value in flags.values()), "Fresh proof must not inherit offline=1"
    assert not any(name.startswith(("huggingface_hub", "transformers")) for name in sys.modules)
    return flags


def ui():
    report = {"status": "started", "inherited_flags": inherited(), "verifier_sets_offline_flags": False,
              "network_attempts": [], "asyncio_loopback": [], "early_imports": [], "torch_bin_loads": [],
              "steps": [], "embedding_calls": 0, "scoring_calls": [], "cpu_threads_in_test": 4}
    import requests
    original_import = builtins.__import__
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def reject(*args, **kwargs):
        report["network_attempts"].append(str(args[1:]))
        raise AssertionError("External network or HTTP attempt forbidden")

    def socket_guard(original):
        def guarded(sock, address):
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                report["asyncio_loopback"].append(list(address))
                return original(sock, address)
            return reject(sock, address)
        return guarded

    def import_observer(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in {"huggingface_hub", "transformers"} and name not in sys.modules:
            assert os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            if len(report["early_imports"]) < 12:
                report["early_imports"].append({"module": name, "hf": "1", "tf": "1"})
        return original_import(name, globals, locals, fromlist, level)

    started = time.perf_counter()
    resources = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(requests.sessions.Session, "request", reject))
            stack.enter_context(patch.object(socket.socket, "connect", socket_guard(connect)))
            stack.enter_context(patch.object(socket.socket, "connect_ex", socket_guard(connect_ex)))
            stack.enter_context(patch.object(builtins, "__import__", import_observer))
            # This is an ordinary production entry, not an independent configure_runtime call.
            import app_resources as resources
            from launch import validate_interpreter
            validate_interpreter()
            import numpy as np
            import pandas as pd
            import torch
            from huggingface_hub import constants
            from transformers.utils import is_offline_mode
            from streamlit.testing.v1 import AppTest
            import search_logic as search
            import rerank_logic as rerank
            assert constants.HF_HUB_OFFLINE and is_offline_mode()
            report["production_flags"] = {key: os.environ[key] for key in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DISABLE_SAFETENSORS_CONVERSION")}
            report["cached_offline_modes"] = {"hub": constants.HF_HUB_OFFLINE, "transformers": is_offline_mode()}
            torch.set_num_threads(4)  # Declared test measurement setting; no production default change.
            original_load = torch.load
            embed_real, score_real = search.embed_carriers, rerank.score_candidates

            def bin_observer(*args, **kwargs):
                path = Path(args[0]).resolve()
                expected = APP_ROOT / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots" / BGE_PIN / "pytorch_model.bin"
                assert path == expected.resolve() and path.stat().st_size == 2271145830
                assert kwargs.get("weights_only") is True
                report["torch_bin_loads"].append({"path": str(path), "weights_only": True})
                return original_load(*args, **kwargs)

            def embed_observer(texts, tokenizer, model, device, *args, **kwargs):
                assert model.config._commit_hash == BGE_PIN and not model.training
                assert next(model.parameters()).dtype == torch.float32 and str(device) == "cpu"
                start = time.perf_counter()
                encoded = embed_real(texts, tokenizer, model, device, *args, **kwargs)
                assert np.isfinite(encoded.vectors).all() and encoded.flags.tolist() == ["ok"] * 4
                report["embedding_calls"] += 1
                report["embedding"] = {"pin": model.config._commit_hash, "dtype": "fp32", "vectors_shape": list(encoded.vectors.shape),
                                        "flags": encoded.flags.tolist(), "inference_seconds": time.perf_counter() - start,
                                        "carriers": list(texts)}
                return encoded

            def score_observer(resource, carriers, ids, passages):
                assert resource.model.config._commit_hash == RERANK_PIN and not resource.model.training
                assert next(resource.model.parameters()).dtype == torch.float32
                assert resource.config.batch_size == 4 and resource.config.max_length == 512
                evaluated = score_real(resource, carriers, ids, passages)
                assert evaluated.logits.shape == (25, 4) and np.isfinite(evaluated.logits).all()
                report["scoring_calls"].append({**evaluated.metrics, "candidate_ids": list(evaluated.candidate_ids),
                                                "four_logits": evaluated.logits.tolist()})
                save("final-default-ui.json", report)
                return evaluated

            stack.enter_context(patch.object(torch, "load", bin_observer))
            stack.enter_context(patch.object(search, "embed_carriers", embed_observer))
            stack.enter_context(patch.object(rerank, "score_candidates", score_observer))
            at = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=900).run()
            assert not at.exception and not at.error
            assert at.slider(key="top_k").value == 10
            assert at.number_input(key="candidate_n").value == 25
            assert not at.toggle(key="reranker_enabled").value
            assert "offline" not in [node.key for node in at.toggle]
            report["displayed_defaults"] = {"k": 10, "n": 25, "page_size": at.selectbox(key="page_size").value,
                                             "batch_size": at.selectbox(key="batch_size").value,
                                             "max_length": at.selectbox(key="max_length").value,
                                             "device": at.selectbox(key="rerank_device").value,
                                             "offline_checkbox_present": False}
            assert report["embedding_calls"] == 0 and not report["scoring_calls"]

            def snapshot(label, state, score_count):
                assert not at.exception, [str(e) for e in at.exception]
                assert not at.error, [e.value for e in at.error]
                results = at.session_state["final_results"]
                metadata = at.session_state["export_metadata"]
                analysis = at.session_state["analysis"]
                matrix = analysis["score_matrix"]
                assert matrix.shape == (87546, 5) and np.isfinite(matrix).all()
                assert np.all(np.abs(matrix) <= 1.0001)
                assert metadata["reranker"]["state"] == state
                applied = state == "applied"
                assert metadata["reranker"]["effective_method"] == ("reranker" if applied else "cosine")
                assert ("rerank_score" in results) == applied
                assert report["embedding_calls"] == 1 and len(report["scoring_calls"]) == score_count
                if applied:
                    logits = results[[f"rerank_score_carrier_{i}" for i in range(1, 5)]].to_numpy()
                    assert np.isfinite(logits).all()
                    np.testing.assert_allclose(results.rerank_score, logits.mean(axis=1))
                    assert results.rerank_score.is_monotonic_decreasing
                else:
                    assert results.similitud_carrier.is_monotonic_decreasing
                table = next(node.value for node in at.dataframe if "embedding_row" in node.value)
                page, page_size = int(at.session_state["results_page"]), int(at.session_state["page_size"])
                expected = results.iloc[(page - 1) * page_size:page * page_size]
                assert table.embedding_row.tolist() == expected.embedding_row.tolist()
                assert any(node.value == expected.iloc[0].texto_chunk for node in at.markdown)
                np.testing.assert_allclose(table["Coseno del centroide"], expected.similitud_carrier)
                for row in results.itertuples():
                    np.testing.assert_allclose([getattr(row, c) for c in search.SCORE_COLUMNS], matrix[row.embedding_row])
                exports = at.session_state["exports"]
                csv = pd.read_csv(io.BytesIO(exports["csv"]))
                cfg = json.loads(exports["json"])
                txt = exports["txt"].decode("utf-8")
                assert csv.embedding_row.tolist() == results.embedding_row.tolist()
                assert [int(v) for v in re.findall(r"embedding_row: (\d+)", txt)] == results.embedding_row.tolist()
                assert cfg["reranker"]["state"] == state and cfg["returned_count"] == len(results)
                assert ("rerank_score" in csv) == applied
                for kind, data in exports.items():
                    (EVIDENCE / f"final-ui-{label}.{kind}").write_bytes(data)
                report["steps"].append({"label": label, "state": state, "page": page, "ids": results.embedding_row.tolist(),
                                        "rendered_ids": table.embedding_row.tolist(), "metadata": metadata,
                                        "score_matrix_shape": list(matrix.shape), "embedding_calls": 1, "scoring_calls": score_count,
                                        "rendered_page_default_detail_cosine_logits_and_full_exports_checked": True})
                save("final-default-ui.json", report)
                print(f"Default UI {label}: {state}; query calls=1; scoring calls={score_count}", flush=True)

            next(node for node in at.button if node.label == "Analizar cuatro carriers").click().run()
            snapshot("analysis", "disabled", 0)
            at.toggle(key="reranker_enabled").set_value(True).run()
            snapshot("pending", "pending", 0)
            at.button(key="apply_reranker").click().run()
            snapshot("applied", "applied", 1)
            at.selectbox(key="page_size").set_value(10).run()
            at.slider(key="top_k").set_value(20).run()
            assert at.number_input(key="candidate_n").value == 25
            snapshot("k20-reuse", "applied", 1)
            at.number_input(key="results_page").set_value(2).run()
            snapshot("page2", "applied", 1)
            at.multiselect(key="countries").set_value(["arg"]).run()
            snapshot("filter-pending", "pending", 1)
            assert at.session_state["results_page"] == 1
            at.slider(key="top_k").set_value(10).run()
            at.button(key="apply_reranker").click().run()
            snapshot("reapplied", "applied", 2)
            at.toggle(key="reranker_enabled").set_value(False).run()
            snapshot("disabled", "disabled", 2)
            at.run()
            snapshot("cheap-rerun", "disabled", 2)
            assert report["torch_bin_loads"] and not report["network_attempts"]
        report.update(status="passed", total_seconds=time.perf_counter() - started,
                      browser_row_click="manual pending", actual_browser_download="manual pending")
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        save("final-default-ui.json", report)
        if resources is not None:
            resources.load_query_model_cached.clear()
            resources.load_reranker_cached.clear()
    print(json.dumps({"status": report["status"], "defaults": report["displayed_defaults"],
                      "network_attempts": len(report["network_attempts"]), "inference_seconds": [r["elapsed_seconds"] for r in report["scoring_calls"]],
                      "total_seconds": report["total_seconds"]}), flush=True)


def server():
    """No verifier bootstrap/flags: launch the ordinary production command from online=0."""
    flags = inherited()
    import psutil
    import urllib.request
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    log_path = EVIDENCE / "final-default-server.log"
    report = {"status": "started", "inherited_flags": flags, "verifier_sets_offline_flags": False, "log": str(log_path)}
    env = {**os.environ, "STREAMLIT_SERVER_HEADLESS": "true"}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(APP_ROOT / "launch.py")], cwd=APP_ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        report["launcher_pid"] = proc.pid
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                assert proc.poll() is None, "Created launcher exited before health check"
                match = re.search(r"http://localhost:(\d+)", log_path.read_text(encoding="utf-8"))
                if match:
                    port = int(match.group(1))
                    assert port >= 8502
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as r:
                            health = {"status": r.status, "body": r.read().decode()}
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as r:
                            root = {"status": r.status, "bytes": len(r.read())}
                        assert health == {"status": 200, "body": "ok"} and root["status"] == 200
                        break
                    except OSError:
                        pass
                time.sleep(1)
            else:
                raise TimeoutError("Own ordinary launcher health exceeded 90 s")
            report.update(status="passed", port=port, url=f"http://127.0.0.1:{port}", health=health, root_http=root,
                          children=[{"pid": p.pid, "cmdline": p.cmdline()} for p in psutil.Process(proc.pid).children(recursive=True)])
        finally:
            children = psutil.Process(proc.pid).children(recursive=True) if proc.poll() is None else []
            for child in reversed(children):
                command = child.cmdline()
                assert str(APP_ROOT / "launch.py") in command or str(APP_ROOT / "streamlit_app.py") in command
                child.terminate()
            if children:
                _, alive = psutil.wait_procs(children, timeout=15)
                assert not alive, "Own descendants still running; no escalation attempted"
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=15)
            report.update(state_at_return="stopped explicitly", created_and_stopped_pids=[proc.pid] + [p.pid for p in children])
            save("final-default-server.json", report)
    print(json.dumps(report), flush=True)


def check():
    """Consistency assertions only; no flags, imports of ML, or repeat inference."""
    status = json.loads((APP_ROOT / "verification-status.json").read_text(encoding="utf-8"))
    ui_report = json.loads((EVIDENCE / "final-default-ui.json").read_text(encoding="utf-8"))
    query = json.loads((EVIDENCE / "offline-startup-real-query-probe.json").read_text(encoding="utf-8"))
    server_report = json.loads((EVIDENCE / "final-default-server.json").read_text(encoding="utf-8"))
    hashes = json.loads((APP_ROOT / "protected-hashes-report.json").read_text(encoding="utf-8"))
    assert ui_report["status"] == server_report["status"] == query["status"] == "passed"
    assert not ui_report["network_attempts"] and not query["network_attempts"]
    assert query["model_weights_loaded"] and query["carrier_inference_executed"]
    assert query["online_resource_request_rejected"]
    assert not ui_report["verifier_sets_offline_flags"] and not server_report["verifier_sets_offline_flags"]
    assert all(v == "0" for v in ui_report["inherited_flags"].values())
    assert all(v == "0" for v in server_report["inherited_flags"].values())
    assert ui_report["production_flags"]["HF_HUB_OFFLINE"] == ui_report["production_flags"]["TRANSFORMERS_OFFLINE"] == "1"
    assert ui_report["displayed_defaults"]["k"] == 10 and ui_report["displayed_defaults"]["n"] == 25
    assert not ui_report["displayed_defaults"]["offline_checkbox_present"]
    assert ui_report["embedding_calls"] == 1 and len(ui_report["scoring_calls"]) == 2 and len(ui_report["steps"]) == 9
    assert ui_report["embedding"]["pin"] == BGE_PIN and ui_report["torch_bin_loads"]
    for score in ui_report["scoring_calls"]:
        assert score["revision"] == RERANK_PIN and score["pair_evaluations"] == 100 and score["truncated_pairs"] == 0
        assert len(score["four_logits"]) == 25
        assert all(len(row) == 4 and all(math.isfinite(v) for v in row) for row in score["four_logits"])
    assert server_report["health"] == {"status": 200, "body": "ok"}
    assert server_report["state_at_return"] == "stopped explicitly"
    assert status["independent_verification"]["server"]["created_and_stopped_pids"] == server_report["created_and_stopped_pids"]
    assert status["independent_verification"]["automated_core_verdict"] == "PASS"
    assert status["test_counts"]["total"] == 97
    assert hashes["unchanged"] and hashes["checked_hashes"] == 27 and not hashes["changes"]
    assert hashes["baseline_sha256"] == status["invariance"]["original_baseline_sha256"]
    for name in ("README.md", "real-verification-report.md"):
        text = (APP_ROOT / name).read_text(encoding="utf-8")
        assert "PASS" in text and "not measured" in text and ("117.97" in text or "117.9667" in text)
    result = {"status": "passed", "automated_core": "PASS", "baseline_sha256": hashes["baseline_sha256"],
              "scope": "Fresh final model/UI/server/defaults/status/docs and27originalhash evidence agree; no repeated benchmark/inference/full-tree traversal"}
    save("final-evidence-consistency.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["ui", "server", "check"])
    args = parser.parse_args()
    {"ui": ui, "server": server, "check": check}[args.phase]()


if __name__ == "__main__":
    main()
