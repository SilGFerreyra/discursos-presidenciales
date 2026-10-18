"""Independent verification evidence only; never patches production logic or installs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch

from runtime import APP_ROOT, configure_runtime

EVIDENCE = APP_ROOT / ".cache/evaluation"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
BGE_MODEL_ID = "BAAI/bge-m3"
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"
BGE_WEIGHT_NAME = "pytorch_model.bin"
BGE_WEIGHT_BYTES = 2271145830
BGE_WEIGHT_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
MODEL_SUPPORT_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json",
                       "special_tokens_map.json", "sentencepiece.bpe.model")


def weight_policy(model_id, revision):
    """The bin exception is sealed to one official file/model/commit, never the reranker."""
    if (model_id, revision) == (BGE_MODEL_ID, BGE_REVISION):
        return {"format": "pytorch_bin", "name": BGE_WEIGHT_NAME, "size": BGE_WEIGHT_BYTES,
                "sha256": BGE_WEIGHT_SHA256, "weights_only": True, "trust_remote_code": False}
    if (model_id, revision) == (RERANK_MODEL_ID, RERANK_REVISION):
        return {"format": "safetensors", "name": "model.safetensors", "trust_remote_code": False}
    raise ValueError("Only the exact approved model identities and revisions may be provisioned")


def owned_cache():
    cache = APP_ROOT / ".cache/huggingface/hub"
    if not cache.resolve().is_relative_to(APP_ROOT.resolve()):
        raise ValueError("Model cache escapes NEWAPP")
    for path in (cache, *cache.parents):
        if path == APP_ROOT:
            break
        if path.is_symlink() or path.is_junction():
            raise ValueError("Model cache directory is redirected")
    return cache


def validate_snapshot_path(path, cache, model_id, revision, name):
    expected = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision / name
    if path.absolute() != expected.absolute() or not path.resolve().is_relative_to(cache.resolve()):
        raise ValueError("File does not belong to the exact NEWAPP model snapshot")


def file_integrity(path, expected_size, expected_sha=None):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    return {"size": expected_size, "sha256": actual, "size_verified": True,
            "sha256_verified": expected_sha is not None}


def validate_builtin_configuration(snapshot, model_id):
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    architecture = "XLMRobertaModel" if model_id == BGE_MODEL_ID else "XLMRobertaForSequenceClassification"
    if ("auto_map" in config or "auto_map" in tokenizer or config.get("model_type") != "xlm-roberta"
            or config.get("architectures") != [architecture]
            or tokenizer.get("tokenizer_class") not in {"XLMRobertaTokenizer", "XLMRobertaTokenizerFast"}):
        raise ValueError("Only the pinned built-in XLM-RoBERTa architecture/tokenizer is permitted")


def _canonical_embedding_load(device, precision, max_length):
    from search_logic import MODEL_ID, MODEL_REVISION
    from generar_embeddings_bge_m3_colab import load_model
    if (MODEL_ID, MODEL_REVISION) != (BGE_MODEL_ID, BGE_REVISION):
        raise ValueError("Canonical helper identity differs from the audited embedding pin")
    return load_model(device, precision, max_length, local_files_only=True)


def load_verified_embedding_model(device, precision, max_length):
    """Check pinned bin bytes before delegating to the unchanged, strictly local helper."""
    configure_runtime()
    from huggingface_hub import hf_hub_download
    cache = owned_cache()
    policy = weight_policy(BGE_MODEL_ID, BGE_REVISION)
    path = Path(hf_hub_download(BGE_MODEL_ID, BGE_WEIGHT_NAME, revision=BGE_REVISION,
                              cache_dir=str(cache), local_files_only=True))
    validate_snapshot_path(path, cache, BGE_MODEL_ID, BGE_REVISION, BGE_WEIGHT_NAME)
    integrity = file_integrity(path, policy["size"], policy["sha256"])
    for name in MODEL_SUPPORT_FILES:
        support = path.parent / name
        validate_snapshot_path(support, cache, BGE_MODEL_ID, BGE_REVISION, name)
        if not support.is_file():
            raise ValueError(f"Required local model support file is missing: {name}")
    if list(path.parent.glob("*.safetensors")) or list(path.parent.glob("*.safetensors.index.json")):
        raise ValueError("Converted/substituted embedding weights are not permitted")
    validate_builtin_configuration(path.parent, BGE_MODEL_ID)
    tokenizer, model = _canonical_embedding_load(device, precision, max_length)
    return tokenizer, model, {"model_id": BGE_MODEL_ID, "pinned_revision": BGE_REVISION,
                              **policy, **integrity, "path": str(path), "local_files_only": True,
                              "safetensors_conversion_disabled": True, "canonical_helper_unchanged": True}


def embedding_load():
    """Opt-in future verification; no inference or CPU-default changes."""
    import torch
    from search_logic import MAX_LENGTH
    started = time.perf_counter()
    tokenizer, model, provenance = load_verified_embedding_model(torch.device("cpu"), "fp32", MAX_LENGTH)
    save("embedding-local-load.json", {"status": "complete", "provenance": provenance,
                                      "load_seconds": time.perf_counter() - started,
                                      "model_training": model.training,
                                      "note": "Local embedding load only, not a dual-model benchmark"})


def save(name, value):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def provision():
    """Provision files separately from model loading under the exact format policies."""
    configure_runtime(offline=False)
    from huggingface_hub import HfApi, hf_hub_download
    import shutil

    cache = owned_cache()
    report = {"status": "started", "cache": str(cache), "disk_free_bytes": shutil.disk_usage(APP_ROOT).free,
              "models": [], "shared_cache_written": False, "weights_converted": False,
              "safetensors_conversion_disabled": True, "trust_remote_code": False}
    save("provisioning.json", report)
    started = time.perf_counter()
    api = HfApi()
    for model_id, revision in ((BGE_MODEL_ID, BGE_REVISION), (RERANK_MODEL_ID, RERANK_REVISION)):
        info = api.model_info(model_id, revision=revision, files_metadata=True, timeout=30)
        if info.sha != revision:
            raise ValueError("Published commit does not match the exact model pin")
        files = {s.rfilename: s for s in info.siblings}
        entry = {"model_id": model_id, "revision": revision, "api_sha": info.sha,
                 "published_files": sorted(files), "files": []}
        report["models"].append(entry)
        policy = weight_policy(model_id, revision)
        entry.update(weight_policy=policy, pinned_revision=revision, weight_format=policy["format"])
        selected = [*MODEL_SUPPORT_FILES, policy["name"]]
        if any(name not in files for name in selected):
            raise ValueError("Pinned revision lacks the required official weights/tokenizer files")
        weight_metadata = files[policy["name"]]
        remote_weight_sha = weight_metadata.lfs.sha256 if weight_metadata.lfs else None
        if (policy["format"] == "pytorch_bin" and
                (weight_metadata.size != policy["size"] or remote_weight_sha != policy["sha256"])):
            raise ValueError("Published embedding bin size/SHA256 differs from the audited official file")
        if remote_weight_sha is None:
            raise ValueError("Official weight SHA256 metadata is required")
        for name in selected:
            metadata = files[name]
            print(f"Downloading {model_id}/{name}: {metadata.size} bytes", flush=True)
            save("provisioning.json", report)
            path = Path(hf_hub_download(model_id, name, revision=revision, cache_dir=str(cache), etag_timeout=30))
            validate_snapshot_path(path, cache, model_id, revision, name)
            remote_lfs_sha = metadata.lfs.sha256 if metadata.lfs else None
            expected_sha = policy.get("sha256", remote_lfs_sha) if name == policy["name"] else remote_lfs_sha
            integrity = file_integrity(path, metadata.size, expected_sha)
            entry["files"].append({"name": name, "path": str(path), "resolved_path": str(path.resolve()),
                                    **integrity, "remote_lfs_sha256": remote_lfs_sha})
            save("provisioning.json", report)
            if name == "tokenizer_config.json":
                validate_builtin_configuration(path.parent, model_id)
        entry.update(status="provisioned", snapshot=str(path.parent), weight_sha256_verified=True,
                     load_policy={"local_files_only": True, "trust_remote_code": False,
                                  "safetensors_conversion_disabled": True})
    report.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                   both_models_provisioned=True, reason="Pinned files verified; models not loaded by provisioning")
    save("provisioning.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["elapsed_seconds"]}), flush=True)


def offline_smoke():
    """Real corpus passages, real reranker, no synthetic ranking and no query model."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import psutil
    import requests
    import torch
    from benchmark_reranker import DEFAULT_CARRIERS
    from rerank_logic import RerankConfig, load_reranker, score_candidates
    from search_logic import load_artifacts

    torch.set_num_threads(4)
    process = psutil.Process()
    peak = {"sampled_peak_rss_bytes": process.memory_info().rss}
    stop = threading.Event()
    network_attempts = []

    def forbid(*args, **kwargs):
        network_attempts.append(str(args[1:] if len(args) > 1 else args))
        raise AssertionError("External networking forbidden during offline verification")

    def sample():
        while not stop.wait(0.05):
            peak["sampled_peak_rss_bytes"] = max(peak["sampled_peak_rss_bytes"], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    report = {"status": "started", "embedding_model_resident": False,
              "scope": "reranker-only smoke, not cosine retrieval or dual-model benchmark",
              "cpu_threads": 4, "network_attempts": network_attempts}
    save("reranker-offline-smoke.json", report)
    try:
        with patch.object(requests.sessions.Session, "request", forbid), patch.object(socket.socket, "connect", forbid), patch.object(socket.socket, "connect_ex", forbid):
            bundle = load_artifacts(APP_ROOT.parent / "salida-embeddings-bge-m3")
            assert not bundle.embeddings.flags.writeable
            # Lexical selection is explicitly NOT embedding retrieval; only exercise real Spanish passages.
            mask = bundle.metadata.texto_chunk.str.contains("electoral|fraude|voto", case=False, regex=True)
            passages = bundle.metadata.loc[mask].head(25)
            assert len(passages) == 25
            config = RerankConfig()
            started = time.perf_counter()
            resource = load_reranker(config, True, str(APP_ROOT / ".cache/huggingface/hub"))
            report["local_load_seconds"] = time.perf_counter() - started
            assert resource.model.config._commit_hash == RERANK_REVISION
            assert not resource.model.training
            assert next(resource.model.parameters()).dtype == torch.float32
            output = score_candidates(resource, DEFAULT_CARRIERS, passages.embedding_row.tolist(), passages.texto_chunk.tolist())
            assert output.logits.shape == (25, 4) and np.isfinite(output.logits).all()
            rows = passages.to_dict(orient="records")
            for row, scores in zip(rows, output.logits.tolist()):
                row["four_logits"] = scores
                row["mean_raw_logit"] = float(np.mean(scores))
            report.update(status="complete", model_commit=resource.model.config._commit_hash,
                          model_dtype=str(next(resource.model.parameters()).dtype), model_training=resource.model.training,
                          corpus=bundle.manifest["compatibility_fingerprint"], metrics=output.metrics,
                          rows=rows, carriers=list(DEFAULT_CARRIERS), local_files_only=True,
                          offline_assertions={"HF_HUB_OFFLINE": True, "TRANSFORMERS_OFFLINE": True,
                                              "http_requests_and_socket_connect_blocked": True, "attempts": len(network_attempts)},
                          relevance={"precision_at_10": "not measured", "ndcg_at_10": "not measured"})
            assert not network_attempts
    finally:
        stop.set()
        sampler.join()
        report["memory"] = {**peak, "sample_interval_seconds": 0.05,
                            "os_peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None)}
        save("reranker-offline-smoke.json", report)
    print(json.dumps({"status": report["status"], "metrics": report.get("metrics"), "memory": report["memory"]}), flush=True)


def shell():
    """Actual AppTest startup without models, then stop before unsafe query loading."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import requests
    from streamlit.testing.v1 import AppTest
    import app_resources

    with patch.object(requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden")), \
            patch.object(app_resources, "load_query_model_cached", side_effect=AssertionError("No query model load allowed")) as query, \
            patch.object(app_resources, "load_reranker_cached", side_effect=AssertionError("No implicit reranker load allowed")) as reranker:
        at = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=120).run()
        assert not at.exception and not at.error
        at.toggle(key="reranker_enabled").set_value(True).run()
        assert not at.exception and not at.error
        at.multiselect(key="countries").set_value(["arg"]).run()
        assert not at.exception and not at.error
        at.selectbox(key="page_size").set_value(10).run()
        assert not at.exception and not at.error
        assert query.call_count == 0 and reranker.call_count == 0
        report = {"status": "complete", "scope": "Real-corpus UI shell only; no analyzed query, scores or browser interaction",
                  "metrics": [{"label": n.label, "value": n.value} for n in at.metric],
                  "query_model_loads": query.call_count, "reranker_model_loads": reranker.call_count,
                  "checks": ["actual real-corpus render", "enable toggle", "country filter", "page size", "no implicit model loads"],
                  "browser_packages": {n: importlib.util.find_spec(n) is not None for n in ("playwright", "selenium")},
                   "pending": ["real analysis/apply/paginate/reapply/disable/export flow requires verified BGE-M3 local provisioning/load", "browser row click and real download"]}
        save("real-ui-shell.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


def server():
    """Start and stop ONLY the launcher/process descendants created by this check."""
    import psutil
    import re
    import urllib.request

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    log_path = EVIDENCE / "server.log"
    report = {"status": "started", "launcher": str(APP_ROOT / "launch.py"), "log": str(log_path)}
    env = {**os.environ, "STREAMLIT_SERVER_HEADLESS": "true", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(APP_ROOT / "launch.py")], cwd=APP_ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        report["launcher_pid"] = proc.pid
        save("server-check.json", report)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Own launcher exited {proc.returncode}")
                text = log_path.read_text(encoding="utf-8")
                match = re.search(r"http://localhost:(\d+)", text)
                if match:
                    port = int(match.group(1))
                    assert port >= 8502
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as response:
                            health = {"status": response.status, "body": response.read().decode()}
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
                            root = {"status": response.status, "bytes": len(response.read())}
                        assert health == {"status": 200, "body": "ok"} and root["status"] == 200
                        break
                    except OSError:
                        pass
                time.sleep(1)
            else:
                raise TimeoutError("Own server was not healthy within 90s")
            descendants = psutil.Process(proc.pid).children(recursive=True)
            report.update(status="complete", port=port, health=health, root_http=root,
                          processes=[{"pid": p.pid, "cmdline": p.cmdline()} for p in descendants],
                          browser_session=False, model_loads_requested=0)
        finally:
            owned = psutil.Process(proc.pid).children(recursive=True) if proc.poll() is None else []
            # Establish ancestry and command ownership before stopping any process.
            for child in reversed(owned):
                command = child.cmdline()
                assert str(APP_ROOT / "launch.py") in command or str(APP_ROOT / "streamlit_app.py") in command, command
                child.terminate()
            if owned:
                _, alive = psutil.wait_procs(owned, timeout=15)
                assert not alive, "Own server did not stop; no escalation attempted"
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=15)
            report.update(server_at_return="stopped explicitly after health verification", stopped_created_pids=[p.pid for p in owned] + [proc.pid],
                           reason_stopped="Health verification finished; avoid leaving an idle verification server")
            save("server-check.json", report)
    print(json.dumps(report), flush=True)


def check_evidence():
    """Validate report consistency; does not repeat model inference or inventory."""
    status = json.loads((APP_ROOT / "verification-status.json").read_text(encoding="utf-8"))
    provision_report = json.loads((EVIDENCE / "provisioning.json").read_text(encoding="utf-8"))
    smoke = json.loads((EVIDENCE / "reranker-offline-smoke.json").read_text(encoding="utf-8"))
    ui = json.loads((EVIDENCE / "real-ui-shell.json").read_text(encoding="utf-8"))
    server_report = json.loads((EVIDENCE / "server-check.json").read_text(encoding="utf-8"))
    inventory = json.loads((APP_ROOT / "invariance-report.json").read_text(encoding="utf-8"))
    assert status["environment"]["reranker_provisioned"]
    bge = provision_report["models"][0]
    if bge["status"] == "blocked_weight_format":
        # Preserve historical provisioning evidence, without reinstating its incorrect policy.
        assert not status["environment"]["embedding_model_provisioned"]
        assert status["policy_correction"]["format_blocker_resolved"]
    else:
        assert bge["status"] == "provisioned" and bge["weight_format"] == "pytorch_bin"
        bin_file = next(f for f in bge["files"] if f["name"] == BGE_WEIGHT_NAME)
        assert bin_file["size"] == BGE_WEIGHT_BYTES and bin_file["sha256"] == BGE_WEIGHT_SHA256
        assert bin_file["sha256_verified"]
    assert status["policy_correction"]["embedding_weight_sha256"] == BGE_WEIGHT_SHA256
    assert status["policy_correction"]["safetensors_conversion_disabled"]
    weight = next(f for f in provision_report["models"][1]["files"] if f["name"] == "model.safetensors")
    assert weight["sha256"] == weight["remote_lfs_sha256"] == status["independent_verification"]["reranker_weight_sha256"]
    assert weight["size"] == status["independent_verification"]["reranker_weight_bytes"]
    assert smoke["status"] == ui["status"] == server_report["status"] == "complete"
    assert smoke["model_commit"] == RERANK_REVISION and len(smoke["rows"]) == 25
    assert smoke["metrics"]["pair_evaluations"] == 100
    assert smoke["offline_assertions"]["attempts"] == 0 and not smoke["embedding_model_resident"]
    assert all(len(row["four_logits"]) == 4 for row in smoke["rows"])
    assert ui["query_model_loads"] == ui["reranker_model_loads"] == 0
    assert server_report["health"] == {"status": 200, "body": "ok"}
    assert server_report["server_at_return"].startswith("stopped")
    assert inventory["unchanged"] and not inventory["changes"]
    assert inventory["protected_entries"] == 42643 and inventory["hashed_entries"] == 27
    baseline_hash = hashlib.sha256((APP_ROOT / "inventory-before.json").read_bytes()).hexdigest()
    assert baseline_hash == inventory["baseline_sha256"] == status["invariance"]["original_baseline_sha256"]
    assert not status["independent_verification"]["relevance"]["improvement_claimed"]
    for name in ("README.md", "real-verification-report.md"):
        text = (APP_ROOT / name).read_text(encoding="utf-8")
        assert "127.90" in text and "not measured" in text
    print(json.dumps({"status": "evidence_consistency_passed", "baseline_sha256": baseline_hash,
                      "note": "No repeat inference or protected-tree traversal; validates completed evidence/report consistency"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["provision", "embedding-load", "offline-smoke", "ui-shell", "server", "check-evidence"])
    phase = parser.parse_args().phase
    configure_runtime(offline=phase != "provision")
    from launch import validate_interpreter
    validate_interpreter()
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    try:
        {"provision": provision, "embedding-load": embedding_load, "offline-smoke": offline_smoke, "ui-shell": shell, "server": server,
         "check-evidence": check_evidence}[phase]()
    except BaseException as error:
        save(f"{phase}-failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
