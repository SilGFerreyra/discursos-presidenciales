"""Canonical BGE-M3 query encoding compatible with the published corpus."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import Version
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIM = 1024
MAX_LENGTH = 8192
EXPECTED_INPUT_ROWS = 87546
EXPECTED_INPUT_SHA256 = "6bdd12428df44ffcff9397edeb53833953f79570c8186e77da6cdeed42612365"


def config_fingerprint(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_model(device, precision, max_length, local_files_only=True):
    if local_files_only is not True:
        raise ValueError("Descargue los modelos con download_models.py antes de ejecutar la app.")
    if Version(torch.__version__.split("+")[0]) < Version("2.6"):
        raise RuntimeError("PyTorch >=2.6 es obligatorio para cargar BGE-M3.")
    model_dtype = torch.float16 if precision == "fp16" else torch.float32
    common = {"revision": MODEL_REVISION, "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
    model = AutoModel.from_pretrained(
        MODEL_ID, **common, dtype=model_dtype, use_safetensors=False, weights_only=True,
    )
    if model.config.hidden_size != EMBEDDING_DIM:
        raise RuntimeError(f"Dimensión inesperada: {model.config.hidden_size}")
    if tokenizer.cls_token_id is None:
        raise RuntimeError("El tokenizador no expone un token CLS/BOS.")
    if max_length > tokenizer.model_max_length:
        raise ValueError("El límite de tokens supera el máximo del tokenizador.")
    model.eval()
    model.to(device)
    return tokenizer, model


@torch.inference_mode()
def encode_batch(texts, tokenizer, model, device, max_length):
    tokenized = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    token_counts = np.asarray([len(ids) for ids in tokenized["input_ids"]], dtype=np.int32)
    if np.any(token_counts > max_length):
        positions = np.flatnonzero(token_counts > max_length).tolist()
        raise ValueError(f"Hay textos que superarían max_length en el batch: {positions}")
    encoded = tokenizer(
        texts, add_special_tokens=True, padding=True, truncation=True, max_length=max_length,
        pad_to_multiple_of=8 if device.type == "cuda" else None, return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    cls_raw = model(**encoded).last_hidden_state[:, 0, :].float()
    raw_norms = torch.linalg.vector_norm(cls_raw, ord=2, dim=1)
    valid = torch.isfinite(cls_raw).all(dim=1) & torch.isfinite(raw_norms) & raw_norms.gt(0)
    normalized = F.normalize(cls_raw, p=2, dim=1, eps=1e-12)
    normalized_norms = torch.linalg.vector_norm(normalized, ord=2, dim=1)
    valid &= torch.isfinite(normalized_norms) & normalized_norms.sub(1.0).abs().le(1e-4)
    normalized[~valid] = torch.nan
    normalized_norms = torch.linalg.vector_norm(normalized, ord=2, dim=1)
    flags = np.where(valid.cpu().numpy(), "ok", "error")
    return (
        normalized.cpu().numpy().astype(np.float32, copy=False),
        raw_norms.cpu().numpy().astype(np.float32, copy=False),
        normalized_norms.cpu().numpy().astype(np.float32, copy=False),
        token_counts, flags.astype("U5"),
    )
