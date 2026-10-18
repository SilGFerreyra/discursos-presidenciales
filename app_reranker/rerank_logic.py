"""Pinned local cross-encoder inference; atomic four-carrier raw-logit results."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from numbers import Integral
from threading import RLock
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@dataclass(frozen=True)
class RerankConfig:
    model_id: str = RERANK_MODEL_ID
    revision: str = RERANK_REVISION
    device: str = "cpu"
    batch_size: int = 4
    max_length: int = 512

    def __post_init__(self):
        if self.model_id != RERANK_MODEL_ID or self.revision != RERANK_REVISION:
            raise ValueError("La identidad del reranker no coincide con la revisión fijada.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Dispositivo no admitido.")
        for name, maximum, minimum in (("batch_size", 64, 1), ("max_length", 512, 8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
                raise ValueError(f"{name} debe estar entre {minimum} y {maximum}.")

    @property
    def precision(self):
        return "fp16" if self.device == "cuda" else "fp32"

    def metadata(self):
        return {**asdict(self), "precision": self.precision, "aggregation": "mean_raw_logits",
                "carrier_count": 4, "truncation": "passage_only", "tokens_include_special": True}


@dataclass
class RerankerResource:
    tokenizer: Any
    model: Any
    config: RerankConfig
    lock: Any = field(default_factory=RLock)


@dataclass(frozen=True)
class RerankOutput:
    candidate_ids: tuple[int, ...]
    logits: np.ndarray
    metrics: dict[str, Any]


def load_reranker(config=RerankConfig(), local_files_only=True, cache_dir=None):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible; seleccione CPU.")
    common = {"revision": config.revision, "trust_remote_code": False,
              "local_files_only": local_files_only, "cache_dir": cache_dir}
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id, **common, use_safetensors=True,
        dtype=torch.float16 if config.device == "cuda" else torch.float32,
    )
    if model.config.num_labels != 1:
        raise RuntimeError("Se requiere un único logit por par.")
    if config.max_length > tokenizer.model_max_length:
        raise ValueError("El límite de tokens supera el máximo del tokenizador.")
    model.to(torch.device(config.device))
    model.eval()
    return RerankerResource(tokenizer, model, config)


def score_candidates(resource, carriers, candidate_ids, passages):
    """Return only after all 4*N pairs succeed. No sigmoid or partial publication."""
    if len(carriers) != 4 or any(not isinstance(t, str) or not t.strip() for t in carriers):
        raise ValueError("Se requieren exactamente cuatro carriers no vacías.")
    ids = tuple(candidate_ids)
    if len(ids) != len(passages) or len(set(ids)) != len(ids) or len(ids) > 1000:
        raise ValueError("Identificadores, fragmentos o presupuesto inválidos.")
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids):
        raise ValueError("embedding_row debe ser un entero no negativo.")
    if any(not isinstance(t, str) or not t.strip() for t in passages):
        raise ValueError("Los fragmentos deben contener texto.")
    started = time.perf_counter()
    config = resource.config
    flat_logits = []
    truncated_pairs = 0
    with resource.lock, torch.inference_mode():
        tokenizer = resource.tokenizer
        carrier_ids = tokenizer(list(carriers), add_special_tokens=False, truncation=False)["input_ids"]
        special_count = tokenizer.num_special_tokens_to_add(pair=True)
        if len(carrier_ids) != 4 or any(not ids_ or len(ids_) + special_count >= config.max_length for ids_ in carrier_ids):
            raise ValueError("Una carrier no deja espacio para el fragmento dentro del límite de tokens.")
        pending = []

        def flush():
            if not pending:
                return
            batch = tokenizer.pad(pending, padding=True, return_tensors="pt")
            if batch["input_ids"].shape[1] > config.max_length:
                raise ValueError("El relleno excedió el límite de tokens.")
            batch = {key: value.to(config.device) for key, value in batch.items()}
            logits = resource.model(**batch).logits
            if tuple(logits.shape) != (len(pending), 1) or not torch.isfinite(logits).all().item():
                raise ValueError("El reranker no produjo un logit finito por par.")
            flat_logits.extend(logits.detach().float().cpu().numpy().reshape(-1).tolist())
            pending.clear()

        for passage in passages:
            passage_ids = tokenizer(passage, add_special_tokens=False, truncation=False)["input_ids"]
            if not passage_ids:
                raise ValueError("Un fragmento no produjo tokens.")
            for query_ids in carrier_ids:
                room = config.max_length - special_count - len(query_ids)
                trimmed = passage_ids[:room]
                truncated_pairs += int(len(passage_ids) > room)
                encoded = tokenizer.prepare_for_model(
                    query_ids, pair_ids=trimmed, add_special_tokens=True,
                    truncation=False, padding=False, return_attention_mask=True,
                )
                expected = tokenizer.build_inputs_with_special_tokens(query_ids, trimmed)
                if encoded["input_ids"] != expected or len(expected) > config.max_length:
                    raise ValueError("No se pudo conservar la carrier dentro del límite de tokens.")
                pending.append(encoded)
                if len(pending) == config.batch_size:
                    flush()
        flush()
    values = np.asarray(flat_logits, dtype=np.float64)
    if len(values) != 4 * len(ids) or not np.isfinite(values).all():
        raise ValueError("Evaluación incompleta; no se publicaron puntajes.")
    values = values.reshape(len(ids), 4)
    values.setflags(write=False)
    return RerankOutput(tuple(int(i) for i in ids), values,
                        {**config.metadata(), "evaluated_candidates": len(ids),
                         "pair_evaluations": 4 * len(ids), "truncated_pairs": truncated_pairs,
                         "elapsed_seconds": time.perf_counter() - started})
