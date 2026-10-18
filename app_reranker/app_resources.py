"""Bounded shared resources. All inference/tokenization uses resource-owned locks."""
from runtime import APP_ROOT, configure_runtime
configure_runtime()  # Must precede canonical helper, Hub, Transformers, and Torch imports.

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import streamlit as st
import torch

from search_logic import ArtifactBundle, MODEL_ID, MODEL_REVISION, load_artifacts
from generar_embeddings_bge_m3_colab import load_model
from rerank_logic import RerankConfig, load_reranker

PLOT_LOCK = RLock()


@dataclass
class QueryResource:
    tokenizer: Any
    model: Any
    lock: Any = field(default_factory=RLock)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_artifacts_cached(output_dir: str, signature: str) -> ArtifactBundle:
    return load_artifacts(Path(output_dir), require_full=True)


@st.cache_resource(max_entries=1, show_spinner=False)
def _query_model_cached(model_id, model_revision, device_type, precision, max_length):
    if model_id != MODEL_ID or model_revision != MODEL_REVISION:
        raise RuntimeError("El modelo de consulta no coincide con la configuración canónica.")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")
    tokenizer, model = load_model(torch.device(device_type), precision, max_length, local_files_only=True)
    return QueryResource(tokenizer, model)


@st.cache_resource(max_entries=1, show_spinner=False)
def _reranker_cached(config: RerankConfig):
    return load_reranker(config, True, str(APP_ROOT / ".cache/huggingface/hub"))


def _require_offline(local_files_only):
    # Validate outside cache wrappers: underscore arguments are excluded from cache keys.
    if local_files_only is not True:
        raise ValueError("La aplicación solo carga modelos locales. Ejecute provision en un proceso separado.")
    configure_runtime(offline=True)


def load_query_model_cached(model_id, model_revision, device_type, precision, max_length,
                            _local_files_only=True):
    _require_offline(_local_files_only)
    return _query_model_cached(model_id, model_revision, device_type, precision, max_length)


def load_reranker_cached(config: RerankConfig, _local_files_only=True):
    _require_offline(_local_files_only)
    return _reranker_cached(config)


load_query_model_cached.clear = _query_model_cached.clear
load_reranker_cached.clear = _reranker_cached.clear
