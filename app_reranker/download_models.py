"""Download and verify the two pinned models in this app's own cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from runtime import APP_ROOT, configure_runtime

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
    if (model_id, revision) == (BGE_MODEL_ID, BGE_REVISION):
        return {"format": "pytorch_bin", "name": BGE_WEIGHT_NAME, "size": BGE_WEIGHT_BYTES,
                "sha256": BGE_WEIGHT_SHA256, "weights_only": True, "trust_remote_code": False}
    if (model_id, revision) == (RERANK_MODEL_ID, RERANK_REVISION):
        return {"format": "safetensors", "name": "model.safetensors", "trust_remote_code": False}
    raise ValueError("Only the exact pinned model identities and revisions may be downloaded")


def owned_cache():
    cache = APP_ROOT / ".cache/huggingface/hub"
    if not cache.resolve().is_relative_to(APP_ROOT.resolve()):
        raise ValueError("Model cache escapes app_reranker")
    for path in (cache, *cache.parents):
        if path == APP_ROOT:
            break
        if path.is_symlink() or path.is_junction():
            raise ValueError("Model cache directory is redirected")
    return cache


def validate_snapshot_path(path, cache, model_id, revision, name):
    expected = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision / name
    if path.absolute() != expected.absolute() or not path.resolve().is_relative_to(cache.resolve()):
        raise ValueError("File does not belong to the expected model snapshot")


def file_integrity(path, expected_size, expected_sha=None):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {path.name}")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    return {"size": expected_size, "sha256": actual, "sha256_verified": expected_sha is not None}


def validate_builtin_configuration(snapshot, model_id):
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    architecture = "XLMRobertaModel" if model_id == BGE_MODEL_ID else "XLMRobertaForSequenceClassification"
    if ("auto_map" in config or "auto_map" in tokenizer or config.get("model_type") != "xlm-roberta"
            or config.get("architectures") != [architecture]
            or tokenizer.get("tokenizer_class") not in {"XLMRobertaTokenizer", "XLMRobertaTokenizerFast"}):
        raise ValueError("Only the pinned built-in XLM-RoBERTa model and tokenizer are permitted")


def provision():
    configure_runtime(offline=False)
    from huggingface_hub import HfApi, hf_hub_download

    cache = owned_cache()
    started = time.perf_counter()
    models = []
    api = HfApi()
    for model_id, revision in ((BGE_MODEL_ID, BGE_REVISION), (RERANK_MODEL_ID, RERANK_REVISION)):
        info = api.model_info(model_id, revision=revision, files_metadata=True, timeout=30)
        if info.sha != revision:
            raise ValueError("Published commit does not match the pinned model revision")
        files = {item.rfilename: item for item in info.siblings}
        policy = weight_policy(model_id, revision)
        selected = [*MODEL_SUPPORT_FILES, policy["name"]]
        if any(name not in files for name in selected):
            raise ValueError("Pinned revision lacks required model or tokenizer files")
        weight = files[policy["name"]]
        remote_weight_sha = weight.lfs.sha256 if weight.lfs else None
        if policy["format"] == "pytorch_bin" and (
                weight.size != policy["size"] or remote_weight_sha != policy["sha256"]):
            raise ValueError("Published embedding weights differ from the pinned file")
        if remote_weight_sha is None:
            raise ValueError("Official weight SHA256 metadata is required")
        verified = []
        for name in selected:
            metadata = files[name]
            print(f"Downloading {model_id}/{name}: {metadata.size} bytes", flush=True)
            path = Path(hf_hub_download(model_id, name, revision=revision, cache_dir=str(cache), etag_timeout=30))
            validate_snapshot_path(path, cache, model_id, revision, name)
            remote_sha = metadata.lfs.sha256 if metadata.lfs else None
            expected_sha = policy.get("sha256", remote_sha) if name == policy["name"] else remote_sha
            verified.append({"name": name, **file_integrity(path, metadata.size, expected_sha)})
            if name == "tokenizer_config.json":
                validate_builtin_configuration(path.parent, model_id)
        models.append({"model_id": model_id, "revision": revision, "files": verified})
    report = {"status": "complete", "seconds": time.perf_counter() - started, "models": models}
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from launch import validate_interpreter
    validate_interpreter()
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    provision()


if __name__ == "__main__":
    main()
