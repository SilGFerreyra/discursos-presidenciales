"""Generate resumable BGE-M3 CLS embeddings for corpus chunks.

Designed for Google Colab GPU. The full run is intentionally blocked on CPU
unless --allow-cpu is provided because BGE-M3 has 570M parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from packaging.version import Version
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIM = 1024
MAX_LENGTH = 8192
PIPELINE_SCHEMA_VERSION = 2
EXPECTED_INPUT_ROWS = 87546
EXPECTED_INPUT_SHA256 = "6bdd12428df44ffcff9397edeb53833953f79570c8186e77da6cdeed42612365"

DEFAULT_INPUT = Path("df-chunks-2026-09-10.csv")
DEFAULT_OUTPUT_DIR = Path("salida-embeddings-bge-m3")
PARQUET_NAME = "df-chunks-emb-2026-09-10.parquet"
MATRIX_NAME = "embeddings-chunks-bge-m3.npy"
METADATA_NAME = "df-chunks-emb-metadata-2026-09-10.csv"
REPORT_NAME = "reporte-embeddings-bge-m3.json"
MANIFEST_NAME = "manifest-embeddings-bge-m3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for testing.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow a slow CPU execution.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a previous output directory.")
    parser.add_argument(
        "--accept-other-input",
        action="store_true",
        help="Allow an input whose row count or SHA-256 differs from the audited corpus.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def locate_input(path: Path) -> Path:
    if path.exists():
        return path
    try:
        from google.colab import files
    except ImportError as exc:
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}") from exc

    print(f"Subí {path.name}")
    uploaded = files.upload()
    if path.name in uploaded:
        return Path(path.name)
    csv_files = [Path(name) for name in uploaded if name.lower().endswith(".csv")]
    if len(csv_files) != 1:
        raise FileNotFoundError(f"No se encontró {path.name} entre los archivos subidos.")
    return csv_files[0]


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 solo está habilitado para CUDA en este pipeline.")
    return requested


def hardware_identity(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return {
        "device": "cpu",
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def verify_official_pooling(local_files_only: bool) -> dict[str, Any]:
    pooling_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="1_Pooling/config.json",
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    )
    modules_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="modules.json",
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    )
    pooling = json.loads(Path(pooling_path).read_text(encoding="utf-8"))
    modules = json.loads(Path(modules_path).read_text(encoding="utf-8"))
    module_types = {module["type"] for module in modules}

    if not pooling.get("pooling_mode_cls_token") or pooling.get("pooling_mode_mean_tokens"):
        raise RuntimeError("La configuración oficial descargada no usa CLS pooling exclusivo.")
    if pooling.get("word_embedding_dimension") != EMBEDDING_DIM:
        raise RuntimeError("La dimensión declarada por el pooling no es 1024.")
    if "sentence_transformers.models.Normalize" not in module_types:
        raise RuntimeError("La configuración oficial no contiene el módulo Normalize esperado.")
    return {"pooling": pooling, "modules": modules}


def validate_input(df: pd.DataFrame) -> None:
    required = {"doc_id", "chunk_id", "texto_chunk"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if df.empty:
        raise ValueError("El DataFrame de entrada está vacío.")
    for column in ("doc_id", "chunk_id"):
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contiene valores nulos o vacíos.")
    if df["texto_chunk"].isna().any() or df["texto_chunk"].astype(str).str.strip().eq("").any():
        raise ValueError("texto_chunk contiene valores nulos o vacíos.")
    if df.duplicated(["doc_id", "chunk_id"]).any():
        raise ValueError("La clave (doc_id, chunk_id) no es única.")


def validate_canonical_input(row_count: int, input_sha256: str, accept_other_input: bool) -> None:
    if not accept_other_input and (
        row_count != EXPECTED_INPUT_ROWS or input_sha256 != EXPECTED_INPUT_SHA256
    ):
        raise RuntimeError(
            "El archivo no coincide con el corpus auditado de 87.546 filas. "
            "Revisá la entrada o usá --accept-other-input de forma explícita."
        )


def load_model(
    device: torch.device,
    precision: str,
    max_length: int,
    local_files_only: bool,
) -> tuple[Any, Any]:
    torch_version = Version(torch.__version__.split("+")[0])
    if torch_version < Version("2.6"):
        raise RuntimeError(
            "PyTorch >=2.6 es obligatorio para cargar de forma segura el "
            "pytorch_model.bin oficial de BGE-M3."
        )
    model_dtype = torch.float16 if precision == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        dtype=model_dtype,
    )
    if model.config.hidden_size != EMBEDDING_DIM:
        raise RuntimeError(f"Dimensión inesperada: {model.config.hidden_size}")
    if tokenizer.cls_token_id is None:
        raise RuntimeError("El tokenizador no expone un token CLS/BOS.")
    if max_length > tokenizer.model_max_length:
        raise ValueError(
            f"max_length={max_length} supera el máximo del tokenizador: {tokenizer.model_max_length}."
        )

    model.eval()
    model.to(device)
    return tokenizer, model


@torch.inference_mode()
def encode_batch(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tokenized_without_truncation = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    token_counts = np.asarray(
        [len(token_ids) for token_ids in tokenized_without_truncation["input_ids"]],
        dtype=np.int32,
    )
    if np.any(token_counts > max_length):
        positions = np.flatnonzero(token_counts > max_length).tolist()
        raise ValueError(f"Hay textos que superarían max_length en el batch: {positions}")

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)

    # BGE-M3's SentenceTransformers configuration selects the first token.
    cls_raw = outputs.last_hidden_state[:, 0, :].float()
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
        token_counts,
        flags.astype("U5"),
    )


def compatibility_config(
    input_sha256: str,
    source_rows: int,
    row_count: int,
    max_length: int,
    precision: str,
    shard_size: int,
    official_pooling_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "source_rows": source_rows,
        "rows": row_count,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIM,
        "pooling": "CLS (last_hidden_state[:, 0, :])",
        "normalization": "L2 explícita después de medir la norma cruda",
        "prefixes": "ninguno",
        "max_length": max_length,
        "precision": precision,
        "shard_size": shard_size,
        "official_pooling_config": official_pooling_config,
    }


def execution_provenance(
    input_path: Path,
    device: torch.device,
    requested_batch_size: int,
) -> dict[str, Any]:
    return {
        "input_file": input_path.name,
        "requested_batch_size": requested_batch_size,
        "runtime": {
            "device_type": device.type,
            "hardware": hardware_identity(device),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def prepare_output(
    output_dir: Path,
    config: dict[str, Any],
    provenance: dict[str, Any],
    overwrite: bool,
) -> Path:
    manifest_path = output_dir / MANIFEST_NAME
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)

    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("compatibility_config") != config:
            raise RuntimeError(
                "La salida existente pertenece a otra configuración. "
                "Usá otro --output-dir o --overwrite."
            )
        manifest = previous
    else:
        manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "running",
            "compatibility_config": config,
            "compatibility_fingerprint": config_fingerprint(config),
            "execution_history": [],
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        }
    execution = dict(provenance)
    execution["started_at_utc"] = pd.Timestamp.utcnow().isoformat()
    manifest["status"] = "running"
    manifest.setdefault("execution_history", []).append(execution)
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def validate_shard(
    shard_path: Path,
    shard_start: int,
    shard_end: int,
    fingerprint: str,
    max_length: int,
) -> dict[str, Any]:
    required = {
        "embeddings",
        "raw_norms",
        "normalized_norms",
        "token_counts",
        "flags",
        "shard_start",
        "shard_end",
        "config_fingerprint",
        "generation_provenance",
    }
    with np.load(shard_path, allow_pickle=False) as saved:
        if not required.issubset(saved.files):
            raise RuntimeError(f"Shard incompleto: {shard_path}")
        try:
            provenance = json.loads(str(saved["generation_provenance"]))
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"Procedencia inválida en el shard: {shard_path}") from exc
        if not isinstance(provenance, dict):
            raise RuntimeError(f"Procedencia inválida en el shard: {shard_path}")
        embeddings = saved["embeddings"]
        raw_norms = saved["raw_norms"]
        normalized_norms = saved["normalized_norms"]
        token_counts = saved["token_counts"]
        flags = saved["flags"]
        expected_rows = shard_end - shard_start
        requested_batch_size = provenance.get("requested_batch_size")
        effective_batch_size = provenance.get("effective_batch_size")
        valid = (
            int(saved["shard_start"]) == shard_start
            and int(saved["shard_end"]) == shard_end
            and str(saved["config_fingerprint"]) == fingerprint
            and embeddings.shape == (expected_rows, EMBEDDING_DIM)
            and embeddings.dtype == np.float32
            and raw_norms.shape == (expected_rows,)
            and normalized_norms.shape == (expected_rows,)
            and token_counts.shape == (expected_rows,)
            and flags.shape == (expected_rows,)
            and np.isfinite(embeddings).all()
            and np.isfinite(raw_norms).all()
            and np.all(raw_norms > 0)
            and np.isfinite(normalized_norms).all()
            and np.all(np.abs(normalized_norms - 1.0) <= 1e-4)
            and np.all((token_counts > 0) & (token_counts <= max_length))
            and np.all(flags == "ok")
            and isinstance(provenance.get("runtime"), dict)
            and isinstance(requested_batch_size, int)
            and requested_batch_size > 0
            and isinstance(effective_batch_size, int)
            and effective_batch_size > 0
        )
    if not valid:
        raise RuntimeError(f"Shard incompatible o inválido: {shard_path}")
    return provenance


def summarize_shard_provenance(provenances: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_groups: dict[str, dict[str, Any]] = {}
    generation_groups: dict[str, dict[str, Any]] = {}
    for provenance in provenances:
        runtime = provenance["runtime"]
        runtime_fingerprint = config_fingerprint(runtime)
        runtime_group = runtime_groups.setdefault(
            runtime_fingerprint,
            {"runtime_fingerprint": runtime_fingerprint, "runtime": runtime, "shard_count": 0},
        )
        runtime_group["shard_count"] += 1

        generation_fingerprint = config_fingerprint(provenance)
        generation_group = generation_groups.setdefault(
            generation_fingerprint,
            {
                "generation_fingerprint": generation_fingerprint,
                "provenance": provenance,
                "shard_count": 0,
            },
        )
        generation_group["shard_count"] += 1

    runtimes = sorted(runtime_groups.values(), key=lambda item: item["runtime_fingerprint"])
    profiles = sorted(
        generation_groups.values(), key=lambda item: item["generation_fingerprint"]
    )
    return {
        "mixed_runtimes": len(runtimes) > 1,
        "unique_runtime_count": len(runtimes),
        "runtimes": runtimes,
        "generation_profiles": profiles,
    }


def process_shards(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
    shard_size: int,
    output_dir: Path,
    fingerprint: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    shard_dir = output_dir / "shards"
    stats = {"oom_retries": 0, "minimum_batch_size": batch_size, "reused_shards": 0}
    shard_provenances = []
    current_batch_size = batch_size
    for shard_start in tqdm(range(0, len(texts), shard_size), desc="Shards"):
        shard_end = min(shard_start + shard_size, len(texts))
        shard_path = shard_dir / f"shard-{shard_start:06d}-{shard_end:06d}.npz"
        if shard_path.exists():
            shard_provenances.append(
                validate_shard(shard_path, shard_start, shard_end, fingerprint, max_length)
            )
            stats["reused_shards"] += 1
            continue

        embedding_parts = []
        raw_norm_parts = []
        normalized_norm_parts = []
        token_count_parts = []
        flag_parts = []
        batch_start = shard_start
        progress = tqdm(total=shard_end - shard_start, desc=f"Filas {shard_start}-{shard_end}", leave=False)
        while batch_start < shard_end:
            batch_end = min(batch_start + current_batch_size, shard_end)
            try:
                result = encode_batch(
                    texts[batch_start:batch_end], tokenizer, model, device, max_length
                )
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda" or current_batch_size == 1:
                    progress.close()
                    raise
                stats["oom_retries"] += 1
                current_batch_size = max(1, current_batch_size // 2)
                stats["minimum_batch_size"] = min(stats["minimum_batch_size"], current_batch_size)
                torch.cuda.empty_cache()
                continue
            if np.any(result[4] != "ok"):
                progress.close()
                raise RuntimeError(f"Embedding inválido detectado en filas {batch_start}:{batch_end}.")
            embedding_parts.append(result[0])
            raw_norm_parts.append(result[1])
            normalized_norm_parts.append(result[2])
            token_count_parts.append(result[3])
            flag_parts.append(result[4])
            progress.update(batch_end - batch_start)
            batch_start = batch_end
        progress.close()

        shard_provenance = dict(provenance)
        shard_provenance["effective_batch_size"] = current_batch_size
        temporary = shard_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                embeddings=np.concatenate(embedding_parts),
                raw_norms=np.concatenate(raw_norm_parts),
                normalized_norms=np.concatenate(normalized_norm_parts),
                token_counts=np.concatenate(token_count_parts),
                flags=np.concatenate(flag_parts),
                shard_start=np.int64(shard_start),
                shard_end=np.int64(shard_end),
                config_fingerprint=np.asarray(fingerprint),
                generation_provenance=np.asarray(
                    json.dumps(shard_provenance, ensure_ascii=False, sort_keys=True)
                ),
            )
        temporary.replace(shard_path)
        shard_provenances.append(
            validate_shard(shard_path, shard_start, shard_end, fingerprint, max_length)
        )
    stats["shard_provenance"] = summarize_shard_provenance(shard_provenances)
    return stats


def consolidate_shards(
    df: pd.DataFrame,
    output_dir: Path,
    shard_size: int,
) -> tuple[pd.DataFrame, np.memmap]:
    matrix_path = output_dir / MATRIX_NAME
    temporary_matrix = matrix_path.with_suffix(".tmp.npy")
    embeddings = np.lib.format.open_memmap(
        temporary_matrix,
        mode="w+",
        dtype=np.float32,
        shape=(len(df), EMBEDDING_DIM),
    )
    raw_norms = np.empty(len(df), dtype=np.float32)
    normalized_norms = np.empty(len(df), dtype=np.float32)
    token_counts = np.empty(len(df), dtype=np.int32)
    flags = np.empty(len(df), dtype="U5")

    for shard_start in range(0, len(df), shard_size):
        shard_end = min(shard_start + shard_size, len(df))
        shard_path = output_dir / "shards" / f"shard-{shard_start:06d}-{shard_end:06d}.npz"
        with np.load(shard_path, allow_pickle=False) as shard:
            embeddings[shard_start:shard_end] = shard["embeddings"]
            raw_norms[shard_start:shard_end] = shard["raw_norms"]
            normalized_norms[shard_start:shard_end] = shard["normalized_norms"]
            token_counts[shard_start:shard_end] = shard["token_counts"]
            flags[shard_start:shard_end] = shard["flags"]
    embeddings.flush()
    del embeddings
    os.replace(temporary_matrix, matrix_path)
    embeddings = np.load(matrix_path, mmap_mode="r")

    metadata = df.copy()
    metadata["embedding_row"] = np.arange(len(df), dtype=np.int64)
    metadata["embedding_norm"] = raw_norms
    metadata["embedding_norm_normalized"] = normalized_norms
    metadata["n_tokens_chunk"] = token_counts
    metadata["embedding_flag"] = flags
    metadata_path = output_dir / METADATA_NAME
    temporary_metadata = metadata_path.with_suffix(".tmp.csv")
    metadata.to_csv(temporary_metadata, index=False, encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)

    table = pa.Table.from_pandas(metadata, preserve_index=False)
    flat_embeddings = pa.array(np.asarray(embeddings).reshape(-1), type=pa.float32())
    embedding_column = pa.FixedSizeListArray.from_arrays(flat_embeddings, EMBEDDING_DIM)
    table = table.append_column("embedding_chunk", embedding_column)
    parquet_path = output_dir / PARQUET_NAME
    temporary_parquet = parquet_path.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary_parquet, compression="zstd")
    os.replace(temporary_parquet, parquet_path)

    df_chunks_emb = metadata.copy()
    df_chunks_emb["embedding_chunk"] = list(embeddings)
    return df_chunks_emb, embeddings


def validate_final_artifacts(
    source: pd.DataFrame,
    output_dir: Path,
    embeddings: np.ndarray,
    max_length: int,
) -> dict[str, Any]:
    matrix_path = output_dir / MATRIX_NAME
    metadata_path = output_dir / METADATA_NAME
    parquet_path = output_dir / PARQUET_NAME
    row_count = len(source)

    if embeddings.shape != (row_count, EMBEDDING_DIM) or embeddings.dtype != np.float32:
        raise RuntimeError("La matriz NPY consolidada tiene forma o dtype incorrectos.")
    for start in range(0, row_count, 4096):
        block = np.asarray(embeddings[start : start + 4096])
        if not np.isfinite(block).all():
            raise RuntimeError("La matriz NPY contiene NaN o infinitos.")
        if not np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=1e-4):
            raise RuntimeError("La matriz NPY contiene vectores sin normalización L2.")

    metadata = pd.read_csv(
        metadata_path,
        usecols=[
            "doc_id",
            "chunk_id",
            "embedding_row",
            "embedding_norm",
            "embedding_norm_normalized",
            "n_tokens_chunk",
            "embedding_flag",
        ],
    )
    if len(metadata) != row_count:
        raise RuntimeError("El CSV de metadatos no tiene la cantidad esperada de filas.")
    if not np.array_equal(metadata["embedding_row"].to_numpy(), np.arange(row_count)):
        raise RuntimeError("embedding_row no coincide con el orden de la matriz NPY.")
    if not metadata["doc_id"].astype(str).equals(source["doc_id"].astype(str)):
        raise RuntimeError("doc_id no coincide entre la entrada y los metadatos.")
    if not np.array_equal(metadata["chunk_id"].to_numpy(), source["chunk_id"].to_numpy()):
        raise RuntimeError("chunk_id no coincide entre la entrada y los metadatos.")
    if not metadata["embedding_flag"].eq("ok").all():
        raise RuntimeError("El CSV de metadatos contiene embeddings marcados con error.")
    if not metadata["n_tokens_chunk"].between(1, max_length).all():
        raise RuntimeError("El CSV contiene conteos de tokens inválidos.")

    parquet_file = pq.ParquetFile(parquet_path)
    if parquet_file.metadata.num_rows != row_count:
        raise RuntimeError("El Parquet no tiene la cantidad esperada de filas.")
    embedding_type = parquet_file.schema_arrow.field("embedding_chunk").type
    if not pa.types.is_fixed_size_list(embedding_type) or embedding_type.list_size != EMBEDDING_DIM:
        raise RuntimeError("embedding_chunk no es una lista fija de 1024 float32 en el Parquet.")
    parquet_keys = pq.read_table(parquet_path, columns=["doc_id", "chunk_id"]).to_pandas()
    if not parquet_keys["doc_id"].astype(str).equals(source["doc_id"].astype(str)):
        raise RuntimeError("doc_id no coincide entre la entrada y el Parquet.")
    if not np.array_equal(parquet_keys["chunk_id"].to_numpy(), source["chunk_id"].to_numpy()):
        raise RuntimeError("chunk_id no coincide entre la entrada y el Parquet.")

    offset = 0
    for batch in parquet_file.iter_batches(batch_size=4096, columns=["embedding_chunk"]):
        column = batch.column(0)
        parquet_vectors = column.values.to_numpy(zero_copy_only=False).reshape(-1, EMBEDDING_DIM)
        end = offset + len(parquet_vectors)
        if not np.array_equal(parquet_vectors, np.asarray(embeddings[offset:end])):
            raise RuntimeError("Los vectores del Parquet no coinciden con la matriz NPY.")
        offset = end
    if offset != row_count:
        raise RuntimeError("No se validaron todas las filas del Parquet.")

    return {
        "status": "ok",
        "rows_checked": row_count,
        "npy_parquet_vectors_identical": True,
        "input_metadata_order_preserved": True,
        "checksums_sha256": {
            MATRIX_NAME: sha256_file(matrix_path),
            METADATA_NAME: sha256_file(metadata_path),
            PARQUET_NAME: sha256_file(parquet_path),
        },
        "file_sizes_bytes": {
            MATRIX_NAME: matrix_path.stat().st_size,
            METADATA_NAME: metadata_path.stat().st_size,
            PARQUET_NAME: parquet_path.stat().st_size,
        },
    }


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.nanmin(values)),
        "p25": float(np.nanpercentile(values, 25)),
        "median": float(np.nanmedian(values)),
        "mean": float(np.nanmean(values)),
        "p75": float(np.nanpercentile(values, 75)),
        "max": float(np.nanmax(values)),
    }


def build_report(
    df_chunks_emb: pd.DataFrame,
    embeddings: np.ndarray,
    config: dict[str, Any],
    fingerprint: str,
    provenance: dict[str, Any],
    device: torch.device,
    batch_size: int,
    shard_size: int,
    elapsed_seconds: float,
    output_dir: Path,
    shard_stats: dict[str, Any],
    artifact_validation: dict[str, Any],
) -> dict[str, Any]:
    errors = int(df_chunks_emb["embedding_flag"].ne("ok").sum())
    token_counts = df_chunks_emb["n_tokens_chunk"].to_numpy()
    raw_norms = df_chunks_emb["embedding_norm"].to_numpy()
    normalized_norms = df_chunks_emb["embedding_norm_normalized"].to_numpy()
    return {
        "status": "complete" if errors == 0 else "complete_with_errors",
        "configuration": config,
        "compatibility_fingerprint": fingerprint,
        "execution": {
            "current_provenance": provenance,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "batch_size": batch_size,
            "shard_size": shard_size,
            "oom_retries": shard_stats["oom_retries"],
            "minimum_batch_size": shard_stats["minimum_batch_size"],
            "reused_shards": shard_stats["reused_shards"],
            "elapsed_seconds": round(elapsed_seconds, 2),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "shard_provenance": shard_stats["shard_provenance"],
        "quality": {
            "total_chunks": int(len(df_chunks_emb)),
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "embedding_errors": errors,
            "zero_or_nan_raw_norms": int((~np.isfinite(raw_norms) | (raw_norms <= 0)).sum()),
            "normalized_norms_outside_tolerance": int(
                (~np.isfinite(normalized_norms) | (np.abs(normalized_norms - 1.0) > 1e-4)).sum()
            ),
            "chunks_truncated": int((token_counts > config["max_length"]).sum()),
            "n_tokens_chunk": describe(token_counts),
            "embedding_norm_before_l2": describe(raw_norms),
            "embedding_norm_after_l2": describe(normalized_norms),
        },
        "artifacts": {
            "parquet_with_embedding_chunk": str((output_dir / PARQUET_NAME).resolve()),
            "npy_matrix": str((output_dir / MATRIX_NAME).resolve()),
            "csv_metadata_without_vectors": str((output_dir / METADATA_NAME).resolve()),
        },
        "artifact_validation": artifact_validation,
    }


def main(args: argparse.Namespace | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    args = args or parse_args()
    started = time.perf_counter()
    input_path = locate_input(args.input)
    df = pd.read_csv(input_path)
    validate_input(df)
    source_rows = len(df)
    input_sha256 = sha256_file(input_path)
    validate_canonical_input(source_rows, input_sha256, args.accept_other_input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit debe ser mayor que cero.")
        df = df.head(args.limit).copy()

    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    if device.type == "cpu" and args.limit is None and not args.allow_cpu:
        raise RuntimeError(
            "La ejecución completa de BGE-M3 está bloqueada en CPU. "
            "Usá Colab con GPU o agregá --allow-cpu bajo tu responsabilidad."
        )
    batch_size = args.batch_size or (32 if device.type == "cuda" else 8)
    if batch_size <= 0 or args.shard_size <= 0 or args.max_length <= 0:
        raise ValueError("batch-size, shard-size y max-length deben ser mayores que cero.")
    if args.max_length > MAX_LENGTH:
        raise ValueError(f"max-length no puede superar {MAX_LENGTH}.")

    official_config = verify_official_pooling(args.local_files_only)
    config = compatibility_config(
        input_sha256=input_sha256,
        source_rows=source_rows,
        row_count=len(df),
        max_length=args.max_length,
        precision=precision,
        shard_size=args.shard_size,
        official_pooling_config=official_config["pooling"],
    )
    fingerprint = config_fingerprint(config)
    provenance = execution_provenance(input_path, device, batch_size)
    manifest_path = prepare_output(args.output_dir, config, provenance, args.overwrite)

    tokenizer, model = load_model(device, precision, args.max_length, args.local_files_only)
    texts = df["texto_chunk"].astype(str).tolist()
    shard_stats = process_shards(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=batch_size,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        fingerprint=fingerprint,
        provenance=provenance,
    )
    df_chunks_emb, embeddings = consolidate_shards(df, args.output_dir, args.shard_size)
    artifact_validation = validate_final_artifacts(
        source=df,
        output_dir=args.output_dir,
        embeddings=embeddings,
        max_length=args.max_length,
    )
    report = build_report(
        df_chunks_emb=df_chunks_emb,
        embeddings=embeddings,
        config=config,
        fingerprint=fingerprint,
        provenance=provenance,
        device=device,
        batch_size=batch_size,
        shard_size=args.shard_size,
        elapsed_seconds=time.perf_counter() - started,
        output_dir=args.output_dir,
        shard_stats=shard_stats,
        artifact_validation=artifact_validation,
    )
    if report["status"] != "complete":
        raise RuntimeError("La ejecución produjo embeddings inválidos; revisá los shards.")
    write_json_atomic(args.output_dir / REPORT_NAME, report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": report["status"],
            "artifact_checksums_sha256": artifact_validation["checksums_sha256"],
            "shard_provenance": shard_stats["shard_provenance"],
            "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
            "report": REPORT_NAME,
        }
    )
    write_json_atomic(manifest_path, manifest)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return df_chunks_emb, report


if __name__ == "__main__":
    df_chunks_emb, reporte_embeddings = main()
