"""Opt-in real verification observers; never replace computations or edit app logic."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import gc
import io
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
from unittest.mock import patch
import weakref

from runtime import APP_ROOT, configure_runtime
configure_runtime()
from verify_real_runtime import BGE_REVISION, RERANK_REVISION, EVIDENCE, save


def owned(path):
    path = Path(path).resolve()
    if not path.is_relative_to(APP_ROOT):
        raise ValueError("Verification output must remain inside NEWAPP")
    return path


def prepare():
    history = EVIDENCE / "before-resume"
    history.mkdir(parents=True, exist_ok=True)
    for name in ("provisioning.json", "server-check.json", "server.log"):
        source = EVIDENCE / name
        target = history / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    for name in ("README.md", "verification-status.json", "real-verification-report.md"):
        target = history / name
        if not target.exists():
            shutil.copy2(APP_ROOT / name, target)
    print(str(history), flush=True)


@contextmanager
def offline_observers(report, allow_asyncio_loopback=False):
    """Reject networking and observe restricted real bin loads without changing kwargs."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import requests
    import torch
    from huggingface_hub import constants
    from transformers.utils import is_offline_mode
    assert constants.HF_HUB_OFFLINE and is_offline_mode(), "Offline flags must be set before ML/Hub imports"
    original_load = torch.load
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    report["network_attempts"] = []
    report["asyncio_loopback_connects"] = []
    report["torch_bin_loads"] = []

    def reject(*args, **kwargs):
        report["network_attempts"].append(str(args[1:]))
        raise AssertionError("Networking forbidden for real offline verification")

    def observe_load(*args, **kwargs):
        path = Path(args[0]).resolve()
        expected = APP_ROOT / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots" / BGE_REVISION / "pytorch_model.bin"
        assert path == expected.resolve() and kwargs.get("weights_only") is True
        report["torch_bin_loads"].append({"path": str(path), "weights_only": kwargs["weights_only"]})
        return original_load(*args, **kwargs)

    def socket_observer(original):
        def observed(sock, address):
            if allow_asyncio_loopback and isinstance(address, tuple):
                try:
                    internal = ipaddress.ip_address(address[0]).is_loopback
                except ValueError:
                    internal = False
                if internal:
                    report["asyncio_loopback_connects"].append(list(address))
                    return original(sock, address)
            return reject(sock, address)
        return observed

    with ExitStack() as stack:
        stack.enter_context(patch.object(requests.sessions.Session, "request", reject))
        stack.enter_context(patch.object(socket.socket, "connect", socket_observer(original_connect)))
        stack.enter_context(patch.object(socket.socket, "connect_ex", socket_observer(original_connect_ex)))
        stack.enter_context(patch.object(torch, "load", observe_load))
        yield
    assert not report["network_attempts"]
    report["offline_assertions"] = {"local_only": True, "HF_HUB_OFFLINE": True, "TRANSFORMERS_OFFLINE": True,
                                     "DISABLE_SAFETENSORS_CONVERSION": os.environ["DISABLE_SAFETENSORS_CONVERSION"],
                                     "external_http_and_socket_rejected": True, "attempts": len(report["network_attempts"]),
                                     "asyncio_loopback_allowed": allow_asyncio_loopback,
                                     "asyncio_loopback_connect_count": len(report["asyncio_loopback_connects"])}


def embedding():
    import torch
    import verify_real_runtime as verifier
    report = {"status": "started", "scope": "unchanged canonical helper, verified pinned local bin"}
    torch.set_num_threads(4)
    with offline_observers(report):
        verifier.embedding_load()
    assert report["torch_bin_loads"]
    local = json.loads((EVIDENCE / "embedding-local-load.json").read_text(encoding="utf-8"))
    report.update(local)
    save("embedding-offline-observers.json", report)
    print(json.dumps(report), flush=True)


def benchmark(args):
    """Run the existing helper unchanged, forwarding its CLI and checkpoint real results."""
    import numpy as np
    import psutil
    import torch
    import benchmark_reranker as helper
    import search_logic as search
    import generar_embeddings_bge_m3_colab as canonical
    import rerank_logic as rerank

    output = owned(args.output)
    checkpoint = output.with_name(output.stem + "-checkpoint.json")
    progress = output.with_name(output.stem + "-progress.json")
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        if previous.get("status") == "failed":
            preserved = checkpoint.with_name(checkpoint.stem + "-failed-before-resume.json")
            if preserved.exists():
                raise RuntimeError("Prior failed attempt already preserved; no automatic repeat loop")
            shutil.copy2(checkpoint, preserved)
    report = {"status": "started", "output": str(output), "completed_runs": [], "model_loads": [],
              "requested_budgets": args.budgets, "candidate_selection": "real cosine retrieval over complete read-only corpus"}
    state = {"query": None, "last_output": None}
    load_query, load_rank = canonical.load_model, rerank.load_reranker
    score_real, apply_real = rerank.score_candidates, search.apply_rerank
    started_total = time.perf_counter()

    def query_observer(*values, **kwargs):
        assert (kwargs.get("local_files_only", values[3] if len(values) > 3 else None)) is True
        started = time.perf_counter()
        tokenizer, model = load_query(*values, **kwargs)
        assert model.config._commit_hash == BGE_REVISION and not model.training
        assert next(model.parameters()).dtype == torch.float32
        state["query"] = weakref.ref(model)
        report["model_loads"].append({"role": "embedding", "revision": model.config._commit_hash,
                                     "dtype": str(next(model.parameters()).dtype), "seconds": time.perf_counter() - started})
        save(checkpoint.name, report)
        return tokenizer, model

    def rank_observer(*values, **kwargs):
        assert (kwargs.get("local_files_only", values[1] if len(values) > 1 else True)) is True
        started = time.perf_counter()
        resource = load_rank(*values, **kwargs)
        assert resource.model.config._commit_hash == RERANK_REVISION and not resource.model.training
        assert next(resource.model.parameters()).dtype == torch.float32
        report["model_loads"].append({"role": "reranker", "revision": resource.model.config._commit_hash,
                                     "dtype": str(next(resource.model.parameters()).dtype), "seconds": time.perf_counter() - started})
        save(checkpoint.name, report)
        return resource

    def scoring_observer(resource, carriers, ids, passages):
        assert state["query"]() is not None
        count = {"pairs_completed": 0, "last_log": time.perf_counter()}
        print(f"Real inference starting N={len(ids)}, pairs={4 * len(ids)}", flush=True)

        def forward_progress(model, inputs, output):
            count["pairs_completed"] += int(output.logits.shape[0])
            now = time.perf_counter()
            if now - count["last_log"] >= 30:
                value = {"status": "running", "n": len(ids), "pairs_completed": count["pairs_completed"],
                         "expected_pairs": 4 * len(ids), "rss_bytes": psutil.Process().memory_info().rss,
                         "total_elapsed_seconds": now - started_total}
                save(progress.name, value)
                print(json.dumps(value), flush=True)
                count["last_log"] = now

        hook = resource.model.register_forward_hook(forward_progress)
        try:
            evaluated = score_real(resource, carriers, ids, passages)
        finally:
            hook.remove()
        assert evaluated.logits.shape == (len(ids), 4) and np.isfinite(evaluated.logits).all()
        assert count["pairs_completed"] == 4 * len(ids)
        state["last_output"] = evaluated
        report["last_completed_score"] = {**evaluated.metrics, "ids": list(evaluated.candidate_ids),
                                           "four_logits": evaluated.logits.tolist(), "embedding_model_resident": True}
        save(checkpoint.name, report)
        print(f"Completed N={len(ids)} in {evaluated.metrics['elapsed_seconds']:.3f}s", flush=True)
        return evaluated

    def ranking_observer(candidates, evaluated, k):
        rows = apply_real(candidates, evaluated, k)
        all_rows = candidates.to_dict(orient="records")
        for row, logits in zip(all_rows, evaluated.logits.tolist()):
            row["four_logits"] = logits
            row["mean_raw_logit"] = float(np.mean(logits))
        report["completed_runs"].append({**evaluated.metrics, "candidates": all_rows,
                                          "reranker_top10": rows.to_dict(orient="records")})
        save(checkpoint.name, report)
        return rows

    argv = [str(APP_ROOT / "benchmark_reranker.py"), "--budgets", args.budgets, "--threads", "4",
            "--batch-size", "4", "--max-length", "512", "--output", str(output)]
    if args.carriers_file:
        argv.extend(["--carriers-file", str(owned(args.carriers_file))])
    report["benchmark_argv"] = argv
    save(checkpoint.name, report)
    try:
        with offline_observers(report), patch.object(sys, "argv", argv), \
                patch.object(canonical, "load_model", query_observer), \
                patch.object(rerank, "load_reranker", rank_observer), \
                patch.object(rerank, "score_candidates", scoring_observer), \
                patch.object(search, "apply_rerank", ranking_observer):
            helper.main()
        report.update(status="complete", total_elapsed_seconds=time.perf_counter() - started_total)
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["status"] == "complete" and len(result["runs"]) == len(args.budgets.split(","))
        assert len(report["completed_runs"]) == len(result["runs"]) and report["torch_bin_loads"]
        result["verification_observers"] = {k: report[k] for k in ("offline_assertions", "torch_bin_loads", "model_loads", "total_elapsed_seconds")}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc(), total_elapsed_seconds=time.perf_counter() - started_total)
        raise
    finally:
        save(checkpoint.name, report)
    print(json.dumps({"status": report["status"], "total_seconds": report["total_elapsed_seconds"], "output": str(output)}), flush=True)


def ui():
    import numpy as np
    import pandas as pd
    import torch
    from streamlit.testing.v1 import AppTest
    import app_resources
    import search_logic as search
    import rerank_logic as rerank

    torch.set_num_threads(4)
    previous_ui = EVIDENCE / "real-ui-workflow.json"
    if previous_ui.exists() and json.loads(previous_ui.read_text(encoding="utf-8")).get("status") == "failed":
        preserved = EVIDENCE / "real-ui-workflow-guard-failure.json"
        if preserved.exists():
            raise RuntimeError("Prior failed UI attempt already preserved; no automatic repeat loop")
        shutil.copy2(previous_ui, preserved)
    report = {"status": "started", "steps": [], "embeddings_calls": 0, "scoring_calls": [],
              "scope": "actual app/corpus/models; only passive delegating computation observers", "cpu_threads": 4}
    embed_real, score_real = search.embed_carriers, rerank.score_candidates

    def embedding_observer(*values, **kwargs):
        report["embeddings_calls"] += 1
        return embed_real(*values, **kwargs)

    def scoring_observer(*values, **kwargs):
        output = score_real(*values, **kwargs)
        assert output.logits.shape == (25, 4) and np.isfinite(output.logits).all()
        report["scoring_calls"].append(output.metrics)
        save("real-ui-workflow.json", report)
        return output

    def snapshot(at, label, expected_state, expected_scores):
        assert not at.exception, [str(e) for e in at.exception]
        assert not at.error, [e.value for e in at.error]
        results = at.session_state["final_results"]
        metadata = at.session_state["export_metadata"]
        assert metadata["reranker"]["state"] == expected_state
        assert ("rerank_score" in results) == (expected_state == "applied")
        assert len(report["scoring_calls"]) == expected_scores and report["embeddings_calls"] == 1
        if expected_state == "applied":
            values = results[[f"rerank_score_carrier_{i}" for i in range(1, 5)]].to_numpy()
            assert np.isfinite(values).all()
            np.testing.assert_allclose(results.rerank_score, values.mean(axis=1))
            assert results.rerank_score.is_monotonic_decreasing
        else:
            assert results.similitud_carrier.is_monotonic_decreasing
        table = next(n.value for n in at.dataframe if "embedding_row" in n.value)
        page = int(at.session_state["results_page"])
        size = int(at.session_state["page_size"])
        expected = results.iloc[(page - 1) * size:page * size]
        assert table.embedding_row.tolist() == expected.embedding_row.tolist()
        assert any(node.value == expected.iloc[0].texto_chunk for node in at.markdown)
        assert int(at.session_state["selected_embedding_row"]) == int(expected.iloc[0].embedding_row)
        exports = at.session_state["exports"]
        csv = pd.read_csv(io.BytesIO(exports["csv"]))
        txt = exports["txt"].decode("utf-8")
        cfg = json.loads(exports["json"])
        assert csv.embedding_row.tolist() == results.embedding_row.tolist()
        assert cfg["reranker"]["state"] == expected_state and cfg["returned_count"] == len(results)
        import re
        assert [int(v) for v in re.findall(r"embedding_row: (\d+)", txt)] == results.embedding_row.tolist()
        assert ("rerank_score" in csv) == (expected_state == "applied")
        for kind, data in exports.items():
            (EVIDENCE / f"ui-{label}.{kind}").write_bytes(data)
        report["steps"].append({"label": label, "state": expected_state, "page": page,
                                "ids": results.embedding_row.tolist(), "rendered_ids": table.embedding_row.tolist(),
                                "metadata": metadata, "embeddings_calls": report["embeddings_calls"], "scoring_calls": len(report["scoring_calls"]),
                                "rendered_table_and_default_detail_verified": True, "csv_json_txt_payloads_verified": True})
        save("real-ui-workflow.json", report)
        print(f"UI {label}: {expected_state}, embedding calls=1, scoring calls={len(report['scoring_calls'])}", flush=True)

    try:
        with offline_observers(report, allow_asyncio_loopback=True), patch.object(search, "embed_carriers", embedding_observer), \
                patch.object(rerank, "score_candidates", scoring_observer):
            at = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=900).run()
            assert not at.exception and not at.error
            at.slider(key="top_k").set_value(10).run()
            at.number_input(key="candidate_n").set_value(25).run()
            at.selectbox(key="page_size").set_value(10).run()
            assert report["embeddings_calls"] == 0 and not report["scoring_calls"]
            next(b for b in at.button if b.label == "Analizar cuatro carriers").click().run()
            snapshot(at, "analysis", "disabled", 0)
            at.toggle(key="reranker_enabled").set_value(True).run()
            snapshot(at, "pending", "pending", 0)
            at.button(key="apply_reranker").click().run()
            snapshot(at, "applied", "applied", 1)
            # K=10 has one page at minimum page size 10. K=20 reuses the same N=25 scores.
            at.slider(key="top_k").set_value(20).run()
            snapshot(at, "k20-reuse", "applied", 1)
            at.number_input(key="results_page").set_value(2).run()
            snapshot(at, "page2", "applied", 1)
            at.multiselect(key="countries").set_value(["arg"]).run()
            snapshot(at, "filter-pending", "pending", 1)
            assert at.session_state["results_page"] == 1
            at.slider(key="top_k").set_value(10).run()
            at.button(key="apply_reranker").click().run()
            snapshot(at, "reapplied", "applied", 2)
            at.toggle(key="reranker_enabled").set_value(False).run()
            snapshot(at, "disabled", "disabled", 2)
            at.run()
            snapshot(at, "cheap-rerun", "disabled", 2)
        report.update(status="complete", browser_row_click="manual pending", browser_download="manual pending",
                      actual_download_payloads=True, models_loaded_from_cache=True)
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        save("real-ui-workflow.json", report)
        app_resources.load_query_model_cached.clear()
        app_resources.load_reranker_cached.clear()
        gc.collect()
    print(json.dumps({"status": report["status"], "steps": len(report["steps"]), "embedding_calls": report["embeddings_calls"], "scoring_calls": len(report["scoring_calls"])}), flush=True)


def plain_local_probe():
    """Check the real app loader's local-only contract without process offline flags."""
    import requests
    import torch
    from huggingface_hub import constants
    from transformers.utils import is_offline_mode
    from search_logic import MODEL_ID, MODEL_REVISION, MAX_LENGTH
    import app_resources
    report = {"status": "started", "scope": "actual app query-resource loader, local_files_only=True, default process flags",
              "HF_HUB_OFFLINE_cached": constants.HF_HUB_OFFLINE, "transformers_offline_cached": is_offline_mode(),
              "network_attempts": []}

    def reject(*values, **kwargs):
        report["network_attempts"].append(str(values[1:]))
        raise AssertionError("Default-local diagnostic blocks metadata HTTP before traffic")

    try:
        with patch.object(requests.sessions.Session, "request", reject):
            resource = app_resources.load_query_model_cached(MODEL_ID, MODEL_REVISION, "cpu", "fp32", MAX_LENGTH,
                                                            _local_files_only=True)
            assert resource.model.config._commit_hash == BGE_REVISION
        report["status"] = "local_only_without_api_attempt_passed"
    except AssertionError as error:
        report.update(status="local_only_metadata_attempt_blocked", error=str(error), traceback=traceback.format_exc())
    finally:
        app_resources.load_query_model_cached.clear()
        save("default-local-query-probe.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}), flush=True)


def summarize():
    """Validate actual evidence and emit aligned top-ten records, not relevance metrics."""
    import numpy as np
    reports = {}
    for name in ("benchmark", "education-benchmark", "offline-benchmark"):
        reports[name] = json.loads((EVIDENCE / f"{name}.json").read_text(encoding="utf-8"))
        assert reports[name]["status"] == "complete"
        assert reports[name]["verification_observers"]["offline_assertions"]["attempts"] == 0
    comparison = {"status": "complete", "relevance": {"precision_at_10": "not measured", "ndcg_at_10": "not measured"}, "concepts": []}
    for name, label in (("benchmark", "corrupción electoral"), ("education-benchmark", "acceso a la educación")):
        report = reports[name]
        run = report["runs"][-1]
        checkpoint = json.loads((EVIDENCE / f"{name}-checkpoint.json").read_text(encoding="utf-8"))
        actual = checkpoint["completed_runs"][-1]
        assert actual["evaluated_candidates"] == run["requested_n"]
        assert len(actual["candidates"]) == run["requested_n"]
        assert all(len(r["four_logits"]) == 4 and np.isfinite(r["four_logits"]).all() for r in actual["candidates"])
        cosine_ids = [r["embedding_row"] for r in run["cosine_top10"]]
        rerank_ids = [r["embedding_row"] for r in run["reranker_top10"]]
        assert len(set(cosine_ids)) == len(set(rerank_ids)) == 10
        candidates = {r["embedding_row"]: r for r in actual["candidates"]}
        ranked_ids = [r["embedding_row"] for r in sorted(actual["candidates"], key=lambda r: (-r["mean_raw_logit"], r["retrieval_rank"]))]
        rows = []
        for row_id in dict.fromkeys(cosine_ids + rerank_ids):
            row = dict(candidates[row_id])
            row["cosine_top10_rank"] = cosine_ids.index(row_id) + 1 if row_id in cosine_ids else None
            row["reranker_top10_rank"] = rerank_ids.index(row_id) + 1 if row_id in rerank_ids else None
            row["reranker_candidate_rank"] = ranked_ids.index(row_id) + 1
            rows.append(row)
        comparison["concepts"].append({"concept": label, "carriers": report["carriers"], "candidate_n": run["requested_n"],
                                       "inference_seconds": run["elapsed_seconds"], "cosine_top10_ids": cosine_ids,
                                       "reranker_top10_ids": rerank_ids, "overlap_count_not_quality": len(set(cosine_ids) & set(rerank_ids)),
                                       "aligned_union_records": rows})
    save("ranking-comparisons.json", comparison)
    main25 = reports["benchmark"]["runs"][0]
    offline25 = reports["offline-benchmark"]["runs"][0]
    assert [r["embedding_row"] for r in main25["reranker_top10"]] == [r["embedding_row"] for r in offline25["reranker_top10"]]
    np.testing.assert_allclose([r["rerank_score"] for r in main25["reranker_top10"]],
                               [r["rerank_score"] for r in offline25["reranker_top10"]], rtol=1e-6, atol=1e-6)
    ui_report = json.loads((EVIDENCE / "real-ui-workflow.json").read_text(encoding="utf-8"))
    assert ui_report["status"] == "complete" and ui_report["embeddings_calls"] == 1 and len(ui_report["scoring_calls"]) == 2
    assert len(ui_report["steps"]) == 9 and not ui_report["network_attempts"]
    digest = {"status": "complete", "benchmarks": {}, "ui_score_metrics": ui_report["scoring_calls"],
              "ui_internal_loopback_count": len(ui_report["asyncio_loopback_connects"]),
              "comparison_summary": [{k: c[k] for k in ("concept", "candidate_n", "cosine_top10_ids", "reranker_top10_ids", "overlap_count_not_quality")}
                                     for c in comparison["concepts"]], "offline_n25_scores_and_ids_reproduced": True}
    for name, report in reports.items():
        digest["benchmarks"][name] = {"query_load_seconds": report["query_load_seconds"], "reranker_load_seconds": report["reranker_load_seconds"],
                                     "offline_reload_seconds": report["offline_reload_seconds"], "memory": report["memory"],
                                     "total_seconds": report["verification_observers"]["total_elapsed_seconds"],
                                     "runs": [{k: r[k] for k in ("requested_n", "pair_evaluations", "truncated_pairs", "elapsed_seconds")} for r in report["runs"]]}
        progress_path = EVIDENCE / f"{name}-progress.json"
        if progress_path.exists():
            previous = json.loads(progress_path.read_text(encoding="utf-8"))
            if previous.get("status") != "complete":
                save(progress_path.name, {"status": "complete", "source": f"{name}.json completed actual report",
                                          "last_forward_sample": previous, "completed_budgets": [r["requested_n"] for r in report["runs"]],
                                          "completed_budget_pairs": sum(r["pair_evaluations"] for r in report["runs"])})
    save("resume-results-summary.json", digest)
    print(json.dumps(digest, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "embedding", "benchmark", "ui", "plain-local-probe", "summarize"])
    parser.add_argument("--budgets", default="25,50,100,200")
    parser.add_argument("--carriers-file", type=Path)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "benchmark.json")
    args = parser.parse_args()
    if args.phase not in {"prepare", "plain-local-probe"}:
        # Hub/Transformers cache offline flags at import time; configure before any ML import.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from launch import validate_interpreter
    validate_interpreter()
    {"prepare": prepare, "embedding": embedding, "benchmark": lambda: benchmark(args), "ui": ui,
     "plain-local-probe": plain_local_probe, "summarize": summarize}[args.phase]()


if __name__ == "__main__":
    main()
