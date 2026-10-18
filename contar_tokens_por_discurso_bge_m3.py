"""Count BGE-M3 content tokens once per presidential speech.

The input is the canonical chunk metadata. Chunks are reconstructed by
discarding only the recorded word overlap before tokenization. The resulting
count is therefore not the sum of per-chunk token counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from generar_embeddings_bge_m3_colab import (
    EXPECTED_INPUT_ROWS,
    EXPECTED_INPUT_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    config_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR_NAME = "salida-embeddings-bge-m3"
METADATA_NAME = "df-chunks-emb-metadata-2026-09-10.csv"
MANIFEST_NAME = "manifest-embeddings-bge-m3.json"
SOURCE_NAME = "df-chunks-2026-09-10.csv"
OUTPUT_NAME = "tokens-por-discurso-bge-m3-2026-09-10.csv"
EXPECTED_DOCUMENTS = 297

REQUIRED_METADATA_COLUMNS = {
    "doc_id",
    "chunk_id",
    "texto_chunk",
    "n_palabras",
    "inicio_palabra",
    "fin_palabra",
    "overlap_previo",
    "n_palabras_nuevas",
    "chunk_flag",
    "nombre_pdf",
    "anio",
    "presidente",
    "pais",
    "source_row",
}
REQUIRED_SOURCE_COLUMNS = {"nombre_pdf", "texto_embedding", "anio", "presidente", "pais"}
HISTORICAL_COLUMNS = ("pais", "presidente", "anio")
OUTPUT_COLUMNS = ("pais", "presidente", "anio", "cantidad_tokens", "doc_id")
SORT_COLUMNS = ("pais", "presidente", "anio", "cantidad_tokens", "doc_id")
MIN_OVERLAP = 5
MAX_OVERLAP = 10
VALID_CHUNK_FLAGS = {"ok", "forzado"}


class TokenCountError(ValueError):
    """Raised when the chunk contract is not safe to aggregate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TokenCountError(f"No existe el manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TokenCountError(f"El manifest no es JSON valido: {path}") from exc
    if not isinstance(payload, dict):
        raise TokenCountError("El manifest debe contener un objeto JSON.")
    return payload


def validate_manifest(
    manifest_path: Path,
    metadata_path: Path,
    source_path: Path | None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise TokenCountError("El manifest no declara una corrida completa.")

    config = manifest.get("compatibility_config")
    if not isinstance(config, dict):
        raise TokenCountError("El manifest no contiene compatibility_config.")

    expected_config = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input_sha256": EXPECTED_INPUT_SHA256,
        "source_rows": EXPECTED_INPUT_ROWS,
        "rows": EXPECTED_INPUT_ROWS,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise TokenCountError(f"Configuracion incompatible en el manifest: {mismatches}")
    if manifest.get("compatibility_fingerprint") != config_fingerprint(config):
        raise TokenCountError("La huella de compatibilidad del manifest no coincide.")

    checksums = manifest.get("artifact_checksums_sha256")
    if not isinstance(checksums, dict) or not checksums.get(METADATA_NAME):
        raise TokenCountError("Falta el checksum de la metadata en el manifest.")
    if sha256_file(metadata_path) != checksums[METADATA_NAME]:
        raise TokenCountError("El checksum de la metadata no coincide con el manifest.")
    if source_path is not None and sha256_file(source_path) != EXPECTED_INPUT_SHA256:
        raise TokenCountError("El checksum de la fuente no coincide con el corpus canonico.")
    return manifest


def read_metadata(path: Path) -> pd.DataFrame:
    try:
        metadata = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise TokenCountError(f"No existe la metadata: {path}") from exc

    missing = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
    if missing:
        raise TokenCountError(f"Faltan columnas en la metadata: {sorted(missing)}")
    if metadata.empty:
        raise TokenCountError("La metadata esta vacia.")
    if metadata[list(REQUIRED_METADATA_COLUMNS)].isna().any().any():
        raise TokenCountError("La metadata contiene nulos en columnas obligatorias.")
    if len(metadata) != EXPECTED_INPUT_ROWS:
        raise TokenCountError(
            f"La metadata tiene {len(metadata)} filas; se esperaban {EXPECTED_INPUT_ROWS}."
        )
    if metadata.duplicated(["doc_id", "chunk_id"]).any():
        raise TokenCountError("La metadata contiene claves (doc_id, chunk_id) duplicadas.")
    unknown_flags = set(metadata["chunk_flag"].astype(str)) - VALID_CHUNK_FLAGS
    if unknown_flags:
        raise TokenCountError(f"La metadata contiene chunk_flag desconocido: {sorted(unknown_flags)}")
    if metadata["doc_id"].nunique() != EXPECTED_DOCUMENTS:
        raise TokenCountError(
            f"La metadata contiene {metadata['doc_id'].nunique()} discursos; "
            f"se esperaban {EXPECTED_DOCUMENTS}."
        )
    return metadata


def read_source(path: Path) -> pd.DataFrame:
    try:
        source = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise TokenCountError(f"No existe la fuente canonica: {path}") from exc

    missing = REQUIRED_SOURCE_COLUMNS.difference(source.columns)
    if missing:
        raise TokenCountError(f"Faltan columnas en la fuente: {sorted(missing)}")
    if source["nombre_pdf"].duplicated().any():
        raise TokenCountError("La fuente contiene nombres de PDF duplicados.")
    if len(source) != 297:
        raise TokenCountError(f"La fuente tiene {len(source)} discursos; se esperaban 297.")
    return source


def _constant_value(group: pd.DataFrame, column: str) -> Any:
    values = group[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise TokenCountError(f"{column} cambia dentro del discurso {group.name!r}.")
    return values[0]


def reconstruct_documents(
    metadata: pd.DataFrame,
    source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source_by_doc = source.set_index("nombre_pdf", drop=False) if source is not None else None
    documents: list[dict[str, Any]] = []

    for doc_id, unsorted_group in metadata.groupby("doc_id", sort=True):
        if source_by_doc is not None and doc_id not in source_by_doc.index:
            raise TokenCountError(f"El discurso {doc_id!r} no existe en la fuente.")

        group = unsorted_group.sort_values("chunk_id", kind="mergesort")
        expected_ids = list(range(1, len(group) + 1))
        actual_ids = [int(value) for value in group["chunk_id"].tolist()]
        if actual_ids != expected_ids:
            raise TokenCountError(f"chunk_id discontinuo para el discurso {doc_id!r}.")

        rebuilt_words: list[str] = []
        previous_end = 0
        for index, row in enumerate(group.itertuples(index=False)):
            chunk_words = str(row.texto_chunk).split()
            start = int(row.inicio_palabra) - 1
            end = int(row.fin_palabra)
            word_count = int(row.n_palabras)
            overlap = int(row.overlap_previo)
            new_words = int(row.n_palabras_nuevas)

            if start < 0 or end <= start or len(chunk_words) != word_count:
                raise TokenCountError(f"Posiciones invalidas en {doc_id!r}, chunk {row.chunk_id}.")
            if end - start != word_count:
                raise TokenCountError(f"n_palabras inconsistente en {doc_id!r}, chunk {row.chunk_id}.")

            if index == 0:
                if start != 0 or overlap != 0 or new_words != word_count:
                    raise TokenCountError(f"El primer chunk es invalido para {doc_id!r}.")
                rebuilt_words.extend(chunk_words)
            else:
                expected_overlap = previous_end - start
                if expected_overlap < MIN_OVERLAP or expected_overlap > MAX_OVERLAP:
                    raise TokenCountError(
                        f"Overlap fuera de rango en {doc_id!r}, chunk {row.chunk_id}."
                    )
                if overlap != expected_overlap or start > previous_end:
                    raise TokenCountError(
                        f"Overlap o hueco invalido en {doc_id!r}, chunk {row.chunk_id}."
                    )
                if chunk_words[:overlap] != rebuilt_words[-overlap:]:
                    raise TokenCountError(
                        f"Las palabras solapadas no coinciden en {doc_id!r}, chunk {row.chunk_id}."
                    )
                expected_new_words = end - previous_end
                if new_words != expected_new_words or expected_new_words <= 0:
                    raise TokenCountError(
                        f"n_palabras_nuevas invalido en {doc_id!r}, chunk {row.chunk_id}."
                    )
                rebuilt_words.extend(chunk_words[overlap:])

            previous_end = end

        source_row = source_by_doc.loc[doc_id] if source_by_doc is not None else None
        if source_row is not None:
            source_words = str(source_row["texto_embedding"]).split()
            if rebuilt_words != source_words:
                raise TokenCountError(f"La reconstruccion no coincide con la fuente para {doc_id!r}.")
        if len(rebuilt_words) != sum(int(value) for value in group["n_palabras_nuevas"]):
            raise TokenCountError(f"La suma de palabras nuevas no coincide para {doc_id!r}.")
        for column in HISTORICAL_COLUMNS:
            metadata_value = _constant_value(group, column)
            if source_row is not None and str(metadata_value) != str(source_row[column]):
                raise TokenCountError(f"{column} no coincide con la fuente para {doc_id!r}.")

        documents.append(
            {
                "doc_id": str(doc_id),
                "pais": _constant_value(group, "pais"),
                "presidente": _constant_value(group, "presidente"),
                "anio": int(_constant_value(group, "anio")),
                "texto_reconstruido": " ".join(rebuilt_words),
            }
        )

    result = pd.DataFrame(documents)
    expected_documents = len(source) if source is not None else EXPECTED_DOCUMENTS
    if len(result) != expected_documents or result["doc_id"].nunique() != expected_documents:
        raise TokenCountError("La reconstruccion no produjo exactamente un discurso por doc_id.")
    return result


def count_content_tokens(text: str, tokenizer: Any) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        verbose=False,
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, list) or (input_ids and isinstance(input_ids[0], list)):
        raise TokenCountError("El tokenizer no devolvio input_ids para un texto individual.")
    return len(input_ids)


def build_output(documents: pd.DataFrame, tokenizer: Any) -> pd.DataFrame:
    output = documents[["pais", "presidente", "anio", "doc_id", "texto_reconstruido"]].copy()
    output["cantidad_tokens"] = [
        count_content_tokens(text, tokenizer)
        for text in output["texto_reconstruido"]
    ]
    if (output["cantidad_tokens"] <= 0).any():
        raise TokenCountError("Se obtuvo un conteo de tokens no positivo.")
    output = output[list(OUTPUT_COLUMNS) + ["texto_reconstruido"]]
    output = output.sort_values(list(SORT_COLUMNS), kind="mergesort").reset_index(drop=True)
    return output[list(OUTPUT_COLUMNS)]


def load_tokenizer(tokenizer_path: Path | None) -> Any:
    if tokenizer_path is not None:
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            use_fast=True,
        )
    return AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        use_fast=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / OUTPUT_DIR_NAME / METADATA_NAME,
        help="Metadata completa de chunks.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / OUTPUT_DIR_NAME / MANIFEST_NAME,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="CSV fuente opcional para validar el contenido reconstruido.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / OUTPUT_NAME,
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Snapshot local opcional del tokenizer BGE-M3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida y cuenta sin escribir el CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.input.resolve()
    manifest_path = args.manifest.resolve()
    source_path = args.source.resolve() if args.source is not None else None

    validate_manifest(manifest_path, metadata_path, source_path)
    metadata = read_metadata(metadata_path)
    source = read_source(source_path) if source_path is not None else None
    documents = reconstruct_documents(metadata, source)
    tokenizer = load_tokenizer(args.tokenizer_path)
    output = build_output(documents, tokenizer)

    print(f"Discursos procesados: {len(output)}")
    print(f"Tokens minimos: {int(output['cantidad_tokens'].min()):,}")
    print(f"Tokens medianos: {int(output['cantidad_tokens'].median()):,}")
    print(f"Tokens maximos: {int(output['cantidad_tokens'].max()):,}")

    if args.dry_run:
        print("Dry-run: no se escribio ningun CSV.")
        return

    output_path = args.output.resolve()
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"CSV escrito: {output_path}")


if __name__ == "__main__":
    main()
