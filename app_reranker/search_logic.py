"""Independent source-derived copy of ../app/search_logic.py, with candidate ranking.

Only the canonical embedding helper and corpus are shared read-only inputs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from generar_embeddings_bge_m3_colab import (
    EMBEDDING_DIM, EXPECTED_INPUT_ROWS, EXPECTED_INPUT_SHA256,
    MAX_LENGTH, MODEL_ID, MODEL_REVISION, config_fingerprint, encode_batch,
)

MATRIX_NAME = "embeddings-chunks-bge-m3.npy"
METADATA_NAME = "df-chunks-emb-metadata-2026-09-10.csv"
MANIFEST_NAME = "manifest-embeddings-bge-m3.json"
MAX_TOP_K = 500
MAX_CANDIDATES = 1000
DEFAULT_TOP_K = 10
CARRIER_COUNT = 4
SCORE_COLUMNS = ["similitud_carrier"] + [f"similitud_carrier_{i}" for i in range(1, 5)]
RERANK_COLUMNS = ["rerank_score"] + [f"rerank_score_carrier_{i}" for i in range(1, 5)]


class ArtifactError(ValueError):
    """Persisted embeddings violate the corpus contract."""


@dataclass(frozen=True)
class ArtifactBundle:
    metadata: pd.DataFrame
    embeddings: np.ndarray
    manifest: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True)
class CarrierEmbedding:
    vectors: np.ndarray
    centroid: np.ndarray
    flags: np.ndarray
    token_counts: np.ndarray
    raw_norms: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RankedResults:
    rows: pd.DataFrame
    metadata_match_count: int
    similarity_match_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_signature(output_dir: Path) -> str:
    try:
        stat = (output_dir / MANIFEST_NAME).stat()
    except FileNotFoundError as exc:
        raise ArtifactError(f"No existe el manifiesto: {output_dir / MANIFEST_NAME}") from exc
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def read_manifest(output_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"No se puede leer el manifiesto: {output_dir / MANIFEST_NAME}") from exc


def validate_manifest(manifest: dict[str, Any], require_full=True) -> dict[str, Any]:
    if manifest.get("status") != "complete":
        raise ArtifactError("El manifiesto no declara una ejecución completa.")
    config = manifest.get("compatibility_config")
    if not isinstance(config, dict):
        raise ArtifactError("El manifiesto no contiene compatibility_config.")
    expected = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIM,
        "pooling": "CLS (last_hidden_state[:, 0, :])", "prefixes": "ninguno",
        "max_length": MAX_LENGTH, "precision": "fp16",
        "input_sha256": EXPECTED_INPUT_SHA256, "source_rows": EXPECTED_INPUT_ROWS,
    }
    mismatches = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if mismatches:
        raise ArtifactError(f"Configuración incompatible en el manifiesto: {mismatches}")
    if not isinstance(config.get("rows"), int) or config["rows"] <= 0:
        raise ArtifactError("La cantidad de filas del manifiesto es inválida.")
    if require_full and config["rows"] != EXPECTED_INPUT_ROWS:
        raise ArtifactError("La salida no contiene el corpus completo.")
    pooling = config.get("official_pooling_config", {})
    if (pooling.get("word_embedding_dimension") != EMBEDDING_DIM
            or pooling.get("pooling_mode_cls_token") is not True
            or pooling.get("pooling_mode_mean_tokens") is not False):
        raise ArtifactError("El pooling oficial no coincide con CLS exclusivo.")
    if manifest.get("compatibility_fingerprint") != config_fingerprint(config):
        raise ArtifactError("La huella del manifiesto no coincide con su configuración.")
    checksums = manifest.get("artifact_checksums_sha256")
    if not isinstance(checksums, dict) or any(not checksums.get(n) for n in (MATRIX_NAME, METADATA_NAME)):
        raise ArtifactError("Faltan checksums de artefactos.")
    return config


def _validate_metadata(metadata, row_count, max_length):
    required = {"doc_id", "chunk_id", "texto_chunk", "anio", "presidente", "pais",
                "embedding_row", "embedding_norm", "embedding_norm_normalized",
                "n_tokens_chunk", "embedding_flag"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ArtifactError(f"Faltan columnas de metadata: {sorted(missing)}")
    if len(metadata) != row_count or metadata[list(required)].isna().any().any():
        raise ArtifactError("La metadata contiene filas faltantes o valores nulos.")
    if metadata.duplicated(["doc_id", "chunk_id"]).any():
        raise ArtifactError("La metadata contiene claves duplicadas.")
    if not np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(row_count)):
        raise ArtifactError("embedding_row no coincide con el orden de la matriz NPY.")
    if not metadata["embedding_flag"].eq("ok").all():
        raise ArtifactError("La metadata contiene embeddings con flag distinto de ok.")
    raw = metadata["embedding_norm"].to_numpy(dtype=np.float32)
    normalized = metadata["embedding_norm_normalized"].to_numpy(dtype=np.float32)
    tokens = metadata["n_tokens_chunk"].to_numpy()
    if not np.isfinite(raw).all() or not np.all(raw > 0):
        raise ArtifactError("La metadata contiene normas crudas inválidas.")
    if not np.isfinite(normalized).all() or not np.allclose(normalized, 1, atol=1e-4):
        raise ArtifactError("La metadata contiene normas normalizadas inválidas.")
    if not np.all((tokens > 0) & (tokens <= max_length)):
        raise ArtifactError("La metadata contiene conteos de tokens inválidos.")


def _validate_matrix(matrix, row_count):
    if matrix.shape != (row_count, EMBEDDING_DIM) or matrix.dtype != np.float32:
        raise ArtifactError("La forma o el dtype de la matriz son incompatibles.")
    for start in range(0, row_count, 4096):
        block = np.asarray(matrix[start:start + 4096])
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-4):
            raise ArtifactError("La matriz contiene vectores no finitos o sin normalización L2.")


def load_artifacts(output_dir: Path, require_full=True) -> ArtifactBundle:
    output_dir = output_dir.resolve()
    manifest = read_manifest(output_dir)
    config = validate_manifest(manifest, require_full)
    for name in (MATRIX_NAME, METADATA_NAME):
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != manifest["artifact_checksums_sha256"][name]:
            raise ArtifactError(f"Artefacto ausente o checksum inválido: {name}")
    metadata = pd.read_csv(output_dir / METADATA_NAME)
    try:
        embeddings = np.load(output_dir / MATRIX_NAME, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArtifactError("No se pudo abrir la matriz NPY.") from exc
    _validate_metadata(metadata, config["rows"], config["max_length"])
    _validate_matrix(embeddings, config["rows"])
    return ArtifactBundle(metadata, embeddings, manifest, output_dir)


def build_carrier_embedding(vectors, flags: Sequence[str] | None = None) -> CarrierEmbedding:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (CARRIER_COUNT, EMBEDDING_DIM) or not np.isfinite(vectors).all():
        raise ValueError("Se requieren cuatro vectores finitos de 1024 dimensiones.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Las carriers deben estar normalizadas antes de promediarlas.")
    flags_array = np.full(4, "ok", dtype="U5") if flags is None else np.asarray(flags)
    if flags_array.shape != (4,) or not np.all(flags_array == "ok"):
        raise ValueError("Las cuatro carriers deben producir flag ok.")
    mean = vectors.mean(axis=0, dtype=np.float32)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("El promedio de las carriers tiene norma cero o inválida.")
    centroid = (mean / norm).astype(np.float32)
    pairwise = vectors @ vectors.T
    upper = pairwise[np.triu_indices(4, k=1)]
    diagnostics = {
        "centroid_norm_before_l2": float(norm), "centroid_norm": float(np.linalg.norm(centroid)),
        "mean_pairwise_cosine": float(upper.mean()), "min_pairwise_cosine": float(upper.min()),
        "max_pairwise_cosine": float(upper.max()), "carrier_to_centroid": (vectors @ centroid).tolist(),
        "pairwise_cosine": pairwise.tolist(),
    }
    return CarrierEmbedding(vectors, centroid, flags_array.astype("U5"),
                            np.zeros(4, dtype=np.int32), np.zeros(4, dtype=np.float32), diagnostics)


def embed_carriers(texts, tokenizer, model, device, max_length=MAX_LENGTH) -> CarrierEmbedding:
    if len(texts) != 4 or any(not str(text).strip() for text in texts):
        raise ValueError("Las cuatro carrier sentences son obligatorias.")
    cleaned = [str(text).strip() for text in texts]
    vectors, raw, normalized, tokens, flags = encode_batch(cleaned, tokenizer, model, device, max_length)
    result = build_carrier_embedding(vectors, flags)
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise ValueError("La inferencia de las carriers produjo normas inválidas.")
    return CarrierEmbedding(result.vectors, result.centroid, result.flags,
                            np.asarray(tokens, dtype=np.int32), np.asarray(raw, dtype=np.float32),
                            {**result.diagnostics, "token_counts": [int(v) for v in tokens],
                             "raw_norms": [float(v) for v in raw],
                             "normalized_norms": [float(v) for v in normalized]})


def compute_score_matrix(embeddings, carrier_embedding) -> np.ndarray:
    query = np.column_stack((carrier_embedding.centroid, carrier_embedding.vectors.T)).astype(np.float32)
    scores = np.asarray(embeddings @ query, dtype=np.float32)
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.0001):
        raise ValueError("La matriz de similitudes contiene valores inválidos.")
    return np.clip(scores, -1, 1).astype(np.float32, copy=False)


def _integer_budget(value, maximum, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"{name} debe ser un entero entre 1 y {maximum}.")
    return int(value)


def validate_search_controls(cosine_range, top_k, year_range):
    if len(cosine_range) != 2:
        raise ValueError("El rango de coseno debe tener mínimo y máximo.")
    lower, upper = map(float, cosine_range)
    if not np.isfinite([lower, upper]).all() or not -1 <= lower <= upper <= 1:
        raise ValueError("El rango de similitud debe cumplir -1 <= mínimo <= máximo <= 1.")
    top_k = _integer_budget(top_k, MAX_TOP_K, "K")
    if len(year_range) != 2:
        raise ValueError("El rango de años es inválido.")
    years = []
    for value in year_range:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Los años deben ser números enteros finitos, no booleanos.")
        if not isinstance(value, Integral) and (not np.isfinite(value) or value != np.floor(value)):
            raise ValueError("Los años deben ser números enteros finitos.")
        years.append(int(value))
    if years[0] > years[1]:
        raise ValueError("El rango de años es inválido.")
    return lower, upper, top_k, years[0], years[1]


def validate_candidate_budget(candidate_n, top_k):
    candidate_n = _integer_budget(candidate_n, MAX_CANDIDATES, "N")
    top_k = _integer_budget(top_k, MAX_TOP_K, "K")
    if candidate_n < top_k:
        raise ValueError("N debe ser mayor o igual que K.")
    return candidate_n


def initial_candidate_budget(top_k):
    """Measured conservative CPU preset; manual N is never clamped on K changes."""
    top_k = _integer_budget(top_k, MAX_TOP_K, "K")
    return max(25, 2 * top_k)


def _retrieve(metadata, score_matrix, countries, year_range, presidents, cosine_range, limit):
    lower, upper, _, year_min, year_max = validate_search_controls(cosine_range, 1, year_range)
    if score_matrix.shape != (len(metadata), 5) or not np.isfinite(score_matrix).all():
        raise ValueError("La matriz de scores no coincide con la metadata o no es finita.")
    if metadata["embedding_row"].duplicated().any():
        raise ValueError("embedding_row debe identificar cada fragmento de forma única.")
    mask = metadata["anio"].between(year_min, year_max).to_numpy(dtype=bool, copy=True)
    if countries:
        mask &= metadata["pais"].astype(str).isin(list(map(str, countries))).to_numpy()
    if presidents:
        mask &= metadata["presidente"].astype(str).isin(list(map(str, presidents))).to_numpy()
    scores = score_matrix[:, 0]
    matched = np.flatnonzero(mask & (scores >= lower) & (scores <= upper))
    shortlisted = matched
    if len(matched) > limit:
        partition = np.argpartition(scores[matched], -limit)[-limit:]
        cutoff = scores[matched[partition]].min()
        shortlisted = matched[scores[matched] >= cutoff]
    rows = metadata.iloc[shortlisted]
    order = np.lexsort((rows["embedding_row"].to_numpy(), rows["chunk_id"].to_numpy(),
                        rows["presidente"].astype(str).to_numpy(), rows["anio"].to_numpy(), -scores[shortlisted]))
    selected = shortlisted[order][:limit]
    result = metadata.iloc[selected].copy().reset_index(drop=True)
    for index, column in enumerate(SCORE_COLUMNS):
        result[column] = score_matrix[selected, index]
    return RankedResults(result, int(mask.sum()), len(matched))


def rank_results(metadata, score_matrix, countries, year_range, presidents, cosine_range, top_k):
    validate_search_controls(cosine_range, top_k, year_range)
    return _retrieve(metadata, score_matrix, countries, year_range, presidents, cosine_range, int(top_k))


def select_candidates(metadata, score_matrix, countries, year_range, presidents, cosine_range, candidate_n, top_k):
    validate_search_controls(cosine_range, top_k, year_range)
    n = validate_candidate_budget(candidate_n, top_k)
    ranked = _retrieve(metadata, score_matrix, countries, year_range, presidents, cosine_range, n)
    ranked.rows["retrieval_rank"] = np.arange(1, len(ranked.rows) + 1)
    return ranked


def apply_rerank(candidates, output, top_k):
    _integer_budget(top_k, MAX_TOP_K, "K")
    if tuple(candidates["embedding_row"].tolist()) != tuple(output.candidate_ids):
        raise ValueError("Los identificadores del reranker no coinciden con los candidatos.")
    values = np.asarray(output.logits, dtype=np.float64)
    if values.shape != (len(candidates), 4) or not np.isfinite(values).all():
        raise ValueError("Faltan puntajes finitos para las cuatro carriers y todos los candidatos.")
    result = candidates.copy()
    if result["embedding_row"].duplicated().any() or result["retrieval_rank"].duplicated().any():
        raise ValueError("Los candidatos o sus puestos están duplicados.")
    means = (values / 4.0).sum(axis=1)
    if not np.isfinite(means).all():
        raise ValueError("El promedio de logits no es finito.")
    result["rerank_score"] = means
    for index in range(4):
        result[f"rerank_score_carrier_{index + 1}"] = values[:, index]
    result = result.sort_values(["rerank_score", "retrieval_rank"], ascending=[False, True], kind="stable")
    result["final_rank"] = np.arange(1, len(result) + 1)
    return result.head(top_k).copy().reset_index(drop=True)


def candidate_signature(corpus_identity, carriers, candidates, filters, candidate_n, config):
    digest = hashlib.sha256()
    relevant = {key: filters[key] for key in ("countries", "presidents", "year_range", "cosine_range")}
    relevant["countries"] = sorted(map(str, relevant["countries"]))
    relevant["presidents"] = sorted(map(str, relevant["presidents"]))
    digest.update(json.dumps({"corpus": corpus_identity, "carriers": list(carriers), "filters": relevant,
                              "candidate_n": int(candidate_n), "config": config}, sort_keys=True).encode())
    digest.update(candidates[["embedding_row", "texto_chunk", "retrieval_rank"]].to_json(orient="split", index=False).encode())
    digest.update(candidates[SCORE_COLUMNS].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


def build_export_metadata(carriers, carrier_embedding, manifest, filters, device, precision,
                          matching_count, returned_count, reranker=None):
    config = manifest["compatibility_config"]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "carriers": list(carriers), "carrier_count": 4,
        "aggregation": "mean of four individually normalized vectors, followed by L2 renormalization",
        "carrier_diagnostics": carrier_embedding.diagnostics,
        "model": {"id": config["model_id"], "revision": config["model_revision"],
                  "pooling": config["pooling"], "prefixes": config["prefixes"], "max_length": config["max_length"]},
        "corpus": {"rows": config["rows"], "input_sha256": config["input_sha256"],
                   "compatibility_fingerprint": manifest["compatibility_fingerprint"]},
        "execution": {"device": device, "precision": precision}, "filters": filters,
        "matching_count": matching_count, "returned_count": returned_count,
        "methodological_note": "Semantic proximity and raw reranker logits are not human relevance labels, probabilities, or proof of corruption or intent.",
    }
    if reranker is not None:
        payload["reranker"] = dict(reranker)
    return payload


def build_results_text(results, export_metadata):
    lines = ["Búsqueda semántica sobre discursos presidenciales", "", "Carrier sentences:"]
    lines.extend(f"{i}. {text}" for i, text in enumerate(export_metadata["carriers"], 1))
    reranker = export_metadata.get("reranker", {})
    applied = reranker.get("state") == "applied" and reranker.get("effective_method") == "reranker"
    lines.extend(["", f"Orden efectivo: {reranker.get('effective_method', 'cosine')}",
                  f"Estado del reranker: {reranker.get('state', 'disabled')}",
                  f"Coincidencias dentro del rango: {export_metadata['matching_count']}",
                  f"Resultados exportados: {export_metadata['returned_count']}",
                  "Nota: los puntajes no prueban corrupción ni intención.", "=" * 88])
    for row in results.itertuples(index=False):
        lines.extend(["", f"País: {row.pais} | Año: {row.anio} | Presidente: {row.presidente} | Chunk: {row.chunk_id} | embedding_row: {row.embedding_row}",
                      f"Similitud carrier: {row.similitud_carrier:.6f}",
                      "Cosenos individuales: " + ", ".join(f"{getattr(row, f'similitud_carrier_{i}'):.6f}" for i in range(1, 5))])
        if applied:
            lines.extend([f"Logit promedio: {row.rerank_score:.6f} | Puesto original: {row.retrieval_rank} | Puesto final: {row.final_rank}",
                          "Logits individuales: " + ", ".join(f"{getattr(row, f'rerank_score_carrier_{i}'):.6f}" for i in range(1, 5))])
        lines.extend([row.texto_chunk, "-" * 88])
    return "\n".join(lines)


def build_exports(results, metadata, *, cosine_results=None):
    """Fallback requires fresh cosine top-K rows, never a stripped reranked subset.

    The caller must supply the independently retrieved cosine fallback explicitly;
    excluded cosine candidates cannot be reconstructed from a reranked top-K.
    """
    info = metadata.get("reranker", {})
    applied = info.get("state") == "applied" and info.get("effective_method") == "reranker"
    if applied:
        rows = results.copy()
    else:
        if cosine_results is None:
            raise ValueError("La exportación por coseno requiere sus resultados top-K de recuperación originales.")
        if any(column in cosine_results for column in RERANK_COLUMNS + ["retrieval_rank", "final_rank"]):
            raise ValueError("El fallback por coseno no puede reutilizar resultados del reranker.")
        rows = cosine_results.copy()
    if len(rows) != metadata["returned_count"]:
        raise ValueError("La cantidad de filas exportadas no coincide con la configuración.")
    return {"csv": rows.to_csv(index=False).encode("utf-8"),
            "json": json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
            "txt": build_results_text(rows, metadata).encode("utf-8")}
