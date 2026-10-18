"""Segment sentence-aware text chunks with semantic overlap.

The script is self-contained apart from pandas, which is preinstalled in
Google Colab. It preserves every whitespace-delimited token from the source
and only normalizes whitespace in the generated chunks. Consecutive chunks
share 5-10 words to retain context for embeddings.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("df-completo-embedding-2026-09-10.csv")
OUTPUT_FILE = Path("df-chunks-2026-09-10.csv")
REPORT_FILE = Path("reporte-segmentacion-2026-09-10.json")

MIN_WORDS = 40
TARGET_WORDS = 55
MAX_WORDS = 70

MIN_OVERLAP = 5
TARGET_OVERLAP = 8
MAX_OVERLAP = 10

# Conservative new-content limits guarantee 40-70 physical words for every
# allowed overlap size: 35 + 5 = 40 and 60 + 10 = 70.
MIN_NEW_WORDS = MIN_WORDS - MIN_OVERLAP
TARGET_NEW_WORDS = TARGET_WORDS - TARGET_OVERLAP
MAX_NEW_WORDS = MAX_WORDS - MAX_OVERLAP

METHOD_PRIORITY = {
    "oracion": 0,
    "punto_y_coma": 1,
    "dos_puntos": 2,
    "conector": 3,
    "enumeracion": 4,
    "conjuncion": 5,
    "palabra": 6,
}

ABBREVIATIONS = {
    "sr.", "sra.", "srta.", "dr.", "dra.", "ud.", "uds.", "etc.",
    "art.", "arts.", "cap.", "caps.", "pag.", "pags.", "num.", "nro.",
    "inc.", "ej.", "gral.", "prof.", "ing.", "lic.", "av.", "depto.",
    "prov.", "rep.", "tel.", "vol.", "vols.", "fig.", "figs.", "aprox.",
    "cfr.", "cf.", "vs.", "p.", "pp.", "ee.", "uu.", "ss.",
}

CONNECTORS = (
    ("sin", "embargo"),
    ("por", "lo", "tanto"),
    ("en", "cambio"),
    ("no", "obstante"),
    ("asi", "mismo"),
    ("por", "consiguiente"),
    ("en", "consecuencia"),
    ("aun", "asi"),
    ("de", "este", "modo"),
    ("ademas",),
    ("asimismo",),
)

CONJUNCTIONS = (
    ("ya", "que"),
    ("puesto", "que"),
    ("dado", "que"),
    ("si", "bien"),
    ("a", "pesar", "de", "que"),
    ("pero",),
    ("aunque",),
    ("porque",),
    ("mientras",),
    ("sino",),
)


def index_phrases(phrases: tuple[tuple[str, ...], ...]) -> dict[str, tuple[tuple[str, ...], ...]]:
    indexed: dict[str, list[tuple[str, ...]]] = {}
    for phrase in phrases:
        indexed.setdefault(phrase[0], []).append(phrase)
    return {key: tuple(value) for key, value in indexed.items()}


CONNECTORS_BY_FIRST = index_phrases(CONNECTORS)
CONJUNCTIONS_BY_FIRST = index_phrases(CONJUNCTIONS)

CLOSING_MARKS = "\"'”’»)]}"
MATCH_STRIP = ".,;:!?¿¡…()[]{}\"“”'’"


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to one ASCII space."""
    return " ".join(str(text).split())


def clean_match_token(token: str) -> str:
    return token.lower().strip(MATCH_STRIP)


def terminal_core(token: str) -> str:
    return token.rstrip(CLOSING_MARKS)


def is_sentence_end(token: str, following_whitespace: str) -> bool:
    """Detect strong sentence boundaries conservatively for noisy OCR."""
    if "\n" in following_whitespace or "\r" in following_whitespace:
        return True

    core = terminal_core(token)
    lower = core.lower()
    if re.search(r"[!?…]+$", core) or re.search(r"\.{2,}$", core):
        return True
    if not core.endswith("."):
        return False
    if lower in ABBREVIATIONS:
        return False
    if re.fullmatch(r"(?:\d+|[a-záéíóúüñ])\.", lower):
        return False
    if re.fullmatch(r"(?:[a-záéíóúüñ]\.){2,}", lower):
        return False
    return True


def starts_phrase(tokens: list[str], index: int, phrase: tuple[str, ...]) -> bool:
    if index + len(phrase) > len(tokens):
        return False
    return tuple(tokens[index : index + len(phrase)]) == phrase


def mark_boundary(boundaries: dict[int, str], position: int, method: str) -> None:
    previous = boundaries.get(position)
    if previous is None or METHOD_PRIORITY[method] < METHOD_PRIORITY[previous]:
        boundaries[position] = method


def find_boundaries(text: str, words: list[str], spans: list[tuple[int, int]]) -> dict[int, str]:
    """Map word offsets to the best linguistic boundary available there."""
    boundaries: dict[int, str] = {}
    sentence_ends: list[int] = []
    match_tokens = [clean_match_token(word) for word in words]

    for index, word in enumerate(words):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else len(text)
        whitespace = text[spans[index][1] : next_start]
        position = index + 1
        core = terminal_core(word)

        if is_sentence_end(word, whitespace):
            mark_boundary(boundaries, position, "oracion")
            sentence_ends.append(position)
        elif core.endswith(";"):
            mark_boundary(boundaries, position, "punto_y_coma")
        elif core.endswith(":"):
            mark_boundary(boundaries, position, "dos_puntos")

    for index in range(1, len(words)):
        current = match_tokens[index]
        for phrase in CONNECTORS_BY_FIRST.get(current, ()):
            if starts_phrase(match_tokens, index, phrase):
                mark_boundary(boundaries, index, "conector")
                break

        raw = words[index].lower().lstrip(CLOSING_MARKS)
        if re.match(r"^(?:\d+|[a-záéíóúüñ])[.)](?:$|\s)", raw) or raw in {
            "-", "–", "—", "*", "•"
        }:
            mark_boundary(boundaries, index, "enumeracion")

        for phrase in CONJUNCTIONS_BY_FIRST.get(current, ()):
            if starts_phrase(match_tokens, index, phrase):
                mark_boundary(boundaries, index, "conjuncion")
                break

    # Repeated commas inside one sentence are treated as possible list limits.
    sentence_start = 0
    for sentence_end in sorted(set(sentence_ends + [len(words)])):
        comma_positions = [
            index + 1
            for index in range(sentence_start, sentence_end)
            if terminal_core(words[index]).endswith(",")
        ]
        if len(comma_positions) >= 3:
            for position in comma_positions:
                mark_boundary(boundaries, position, "enumeracion")
        sentence_start = sentence_end

    # End of document is a natural boundary, even when OCR omitted punctuation.
    boundaries[len(words)] = "oracion"
    return boundaries


def suffix_is_partitionable(word_count: int) -> bool:
    if word_count == 0:
        return True
    minimum_chunks = math.ceil(word_count / MAX_NEW_WORDS)
    maximum_chunks = word_count // MIN_NEW_WORDS
    return minimum_chunks <= maximum_chunks


def choose_overlap_start(cursor: int, boundaries: dict[int, str]) -> tuple[int, int, str]:
    """Choose a 5-10 word suffix whose left edge is linguistically meaningful."""
    candidates = []
    for overlap in range(MIN_OVERLAP, MAX_OVERLAP + 1):
        start = cursor - overlap
        if start < 0:
            continue
        method = boundaries.get(start, "palabra")
        score = (
            METHOD_PRIORITY[method],
            abs(overlap - TARGET_OVERLAP),
            -overlap,
        )
        candidates.append((score, start, overlap, method))

    if not candidates:
        raise RuntimeError("No se pudo construir el overlap solicitado.")

    _, start, overlap, method = min(candidates)
    return start, overlap, method


def choose_cut(
    chunk_start: int,
    coverage_cursor: int,
    total: int,
    boundaries: dict[int, str],
    first_chunk: bool,
) -> tuple[int, str]:
    remaining = total - coverage_cursor
    if first_chunk:
        minimum_new = MIN_WORDS
        maximum_new = MAX_WORDS
    else:
        minimum_new = MIN_NEW_WORDS
        maximum_new = MAX_NEW_WORDS

    if remaining <= maximum_new:
        return total, boundaries.get(total, "oracion")

    candidates = []
    for end in range(
        coverage_cursor + minimum_new,
        min(coverage_cursor + maximum_new, total) + 1,
    ):
        if not suffix_is_partitionable(total - end):
            continue
        method = boundaries.get(end, "palabra")
        score = (
            METHOD_PRIORITY[method],
            abs((end - chunk_start) - TARGET_WORDS),
            abs((end - coverage_cursor) - TARGET_NEW_WORDS),
            -(end - coverage_cursor),
        )
        candidates.append((score, end, method))

    if not candidates:
        raise RuntimeError(
            f"No se pudo particionar el remanente de {remaining} palabras "
            f"desde la posición {coverage_cursor}."
        )

    _, end, method = min(candidates)
    return end, method


def segment_text(text: str) -> list[dict[str, object]]:
    matches = list(re.finditer(r"\S+", str(text)))
    if not matches:
        return []

    words = [match.group(0) for match in matches]
    spans = [(match.start(), match.end()) for match in matches]
    boundaries = find_boundaries(str(text), words, spans)

    chunks: list[dict[str, object]] = []
    coverage_cursor = 0
    while coverage_cursor < len(words):
        first_chunk = not chunks
        if first_chunk:
            start = 0
            overlap = 0
            overlap_method = "inicio_documento"
        else:
            start, overlap, overlap_method = choose_overlap_start(coverage_cursor, boundaries)

        end, method = choose_cut(
            chunk_start=start,
            coverage_cursor=coverage_cursor,
            total=len(words),
            boundaries=boundaries,
            first_chunk=first_chunk,
        )
        count = end - start
        new_count = end - coverage_cursor
        if count < MIN_WORDS:
            flag = "corto"
        elif count > MAX_WORDS:
            flag = "largo"
        elif method == "palabra":
            flag = "forzado"
        else:
            flag = "ok"

        chunks.append(
            {
                "texto_chunk": " ".join(words[start:end]),
                "n_palabras": count,
                "inicio_palabra": start + 1,
                "fin_palabra": end,
                "overlap_previo": overlap,
                "n_palabras_nuevas": new_count,
                "metodo_inicio_overlap": overlap_method,
                "metodo_corte": method,
                "chunk_flag": flag,
            }
        )
        coverage_cursor = end

    return chunks


def segment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"nombre_pdf", "texto_embedding", "anio", "presidente", "pais"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if df["texto_embedding"].isna().any():
        raise ValueError("La columna texto_embedding contiene valores nulos.")
    if df["nombre_pdf"].duplicated().any():
        raise ValueError("nombre_pdf no es único; no puede usarse como doc_id.")

    rows: list[dict[str, object]] = []
    for source_row, record in df.iterrows():
        doc_chunks = segment_text(record["texto_embedding"])
        for chunk_id, chunk in enumerate(doc_chunks, start=1):
            rows.append(
                {
                    "doc_id": record["nombre_pdf"],
                    "chunk_id": chunk_id,
                    **chunk,
                    "nombre_pdf": record["nombre_pdf"],
                    "anio": record["anio"],
                    "presidente": record["presidente"],
                    "pais": record["pais"],
                    "source_row": source_row,
                }
            )

    return pd.DataFrame(rows)


def validate_output(source: pd.DataFrame, chunks: pd.DataFrame) -> dict[str, int]:
    content_failures = 0
    slice_failures = 0
    position_failures = 0
    overlap_failures = 0
    gap_failures = 0
    chunk_id_failures = 0

    grouped = {doc_id: group for doc_id, group in chunks.groupby("doc_id", sort=False)}
    for _, record in source.iterrows():
        doc_id = record["nombre_pdf"]
        group = grouped.get(doc_id)
        if group is None:
            content_failures += 1
            continue

        group = group.sort_values("chunk_id")
        source_words = str(record["texto_embedding"]).split()
        expected_ids = list(range(1, len(group) + 1))
        if group["chunk_id"].tolist() != expected_ids:
            chunk_id_failures += 1

        rebuilt_words: list[str] = []
        previous_end = 0
        for index, chunk in enumerate(group.itertuples()):
            start = int(chunk.inicio_palabra) - 1
            end = int(chunk.fin_palabra)
            chunk_words = str(chunk.texto_chunk).split()
            expected_overlap = 0 if index == 0 else previous_end - start
            expected_new = end - previous_end

            if chunk_words != source_words[start:end]:
                slice_failures += 1
            if end - start != chunk.n_palabras or len(chunk_words) != chunk.n_palabras:
                position_failures += 1

            if index == 0:
                if start != 0 or chunk.overlap_previo != 0:
                    gap_failures += 1
                rebuilt_words.extend(chunk_words)
            else:
                if start > previous_end or expected_new <= 0:
                    gap_failures += 1
                if (
                    chunk.overlap_previo != expected_overlap
                    or not MIN_OVERLAP <= expected_overlap <= MAX_OVERLAP
                    or chunk_words[:expected_overlap] != source_words[start:previous_end]
                ):
                    overlap_failures += 1
                rebuilt_words.extend(chunk_words[expected_overlap:])

            if chunk.n_palabras_nuevas != expected_new:
                position_failures += 1
            previous_end = end

        if previous_end != len(source_words) or rebuilt_words != source_words:
            content_failures += 1

    too_short = int((chunks["n_palabras"] < MIN_WORDS).sum())
    too_long = int((chunks["n_palabras"] > MAX_WORDS).sum())
    if any(
        (
            content_failures,
            slice_failures,
            position_failures,
            overlap_failures,
            gap_failures,
            chunk_id_failures,
            too_short,
            too_long,
        )
    ):
        raise AssertionError(
            "Validación fallida: "
            f"contenido={content_failures}, slices={slice_failures}, "
            f"posiciones={position_failures}, overlap={overlap_failures}, "
            f"huecos={gap_failures}, ids={chunk_id_failures}, "
            f"cortos={too_short}, largos={too_long}"
        )

    return {
        "documentos_con_contenido_alterado": content_failures,
        "chunks_que_no_coinciden_con_fuente": slice_failures,
        "chunks_con_posiciones_invalidas": position_failures,
        "transiciones_con_overlap_invalido": overlap_failures,
        "transiciones_con_huecos": gap_failures,
        "documentos_con_chunk_id_invalido": chunk_id_failures,
        "chunks_menores_a_40": too_short,
        "chunks_mayores_a_70": too_long,
    }


def build_report(source: pd.DataFrame, chunks: pd.DataFrame, validation: dict[str, int]) -> dict[str, object]:
    counts = chunks["n_palabras"]
    bins = [-math.inf, 39, 44, 49, 54, 59, 64, 70, math.inf]
    labels = ["<40", "40-44", "45-49", "50-54", "55-59", "60-64", "65-70", ">70"]
    distribution = pd.cut(counts, bins=bins, labels=labels).value_counts(sort=False)
    forced = chunks[chunks["metodo_corte"] == "palabra"].head(10)
    overlapped = chunks[chunks["overlap_previo"] > 0]
    semantic_overlap = overlapped[overlapped["metodo_inicio_overlap"] != "palabra"]
    overlap_examples = semantic_overlap.head(10)

    return {
        "total_documentos": int(len(source)),
        "total_chunks": int(len(chunks)),
        "resumen_n_palabras": {
            "minimo": int(counts.min()),
            "p25": float(counts.quantile(0.25)),
            "mediana": float(counts.median()),
            "media": round(float(counts.mean()), 2),
            "p75": float(counts.quantile(0.75)),
            "maximo": int(counts.max()),
        },
        "distribucion_n_palabras": {str(key): int(value) for key, value in distribution.items()},
        "chunks_fuera_de_40_70": int(((counts < MIN_WORDS) | (counts > MAX_WORDS)).sum()),
        "distribucion_overlap_previo": {
            str(int(key)): int(value)
            for key, value in overlapped["overlap_previo"].value_counts().sort_index().items()
        },
        "overlap_promedio": round(float(overlapped["overlap_previo"].mean()), 2),
        "palabras_duplicadas_por_overlap": int(overlapped["overlap_previo"].sum()),
        "transiciones_con_overlap_fuera_de_5_10": int(
            (~overlapped["overlap_previo"].between(MIN_OVERLAP, MAX_OVERLAP)).sum()
        ),
        "inicios_overlap_semanticos": int(len(semantic_overlap)),
        "inicios_overlap_forzados_por_palabra": int(
            (overlapped["metodo_inicio_overlap"] == "palabra").sum()
        ),
        "metodos_inicio_overlap": {
            str(key): int(value)
            for key, value in overlapped["metodo_inicio_overlap"].value_counts().items()
        },
        "metodos_de_corte": {
            str(key): int(value) for key, value in chunks["metodo_corte"].value_counts().items()
        },
        "chunk_flags": {
            str(key): int(value) for key, value in chunks["chunk_flag"].value_counts().items()
        },
        "ejemplos_cortes_forzados": [
            {
                "doc_id": row.doc_id,
                "chunk_id": int(row.chunk_id),
                "n_palabras": int(row.n_palabras),
                "texto_chunk": row.texto_chunk,
            }
            for row in forced.itertuples()
        ],
        "ejemplos_overlap_semantico": [
            {
                "doc_id": row.doc_id,
                "chunk_id": int(row.chunk_id),
                "overlap_previo": int(row.overlap_previo),
                "metodo_inicio_overlap": row.metodo_inicio_overlap,
                "texto_overlap": " ".join(row.texto_chunk.split()[: row.overlap_previo]),
            }
            for row in overlap_examples.itertuples()
        ],
        "validacion": validation,
        "posiciones_de_palabra": "1-based e inclusivas; los intervalos se solapan intencionalmente",
    }


def locate_input() -> Path:
    if INPUT_FILE.exists():
        return INPUT_FILE

    try:
        from google.colab import files
    except ImportError as exc:
        raise FileNotFoundError(f"No se encontró {INPUT_FILE}") from exc

    print(f"Subí {INPUT_FILE.name}")
    uploaded = files.upload()
    if INPUT_FILE.name not in uploaded:
        csv_files = [Path(name) for name in uploaded if name.lower().endswith(".csv")]
        if len(csv_files) != 1:
            raise FileNotFoundError(f"No se encontró {INPUT_FILE.name} entre los archivos subidos.")
        return csv_files[0]
    return INPUT_FILE


def main() -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = locate_input()
    df_completo = pd.read_csv(input_path)
    df_chunks = segment_dataframe(df_completo)
    validation = validate_output(df_completo, df_chunks)
    report = build_report(df_completo, df_chunks, validation)

    df_chunks.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nChunks guardados en: {OUTPUT_FILE.resolve()}")
    print(f"Reporte guardado en: {REPORT_FILE.resolve()}")

    try:
        from google.colab import files
    except ImportError:
        pass
    else:
        files.download(str(OUTPUT_FILE))
        files.download(str(REPORT_FILE))

    return df_chunks, report


if __name__ == "__main__":
    df_chunks, reporte = main()
