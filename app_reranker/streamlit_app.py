"""Independent source-derived UI from ../app/streamlit_app.py, with explicit reranking."""
from __future__ import annotations

from runtime import configure_runtime
configure_runtime()

import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from adjustText import adjust_text

from search_logic import (
    CARRIER_COUNT, MAX_TOP_K, MAX_CANDIDATES, DEFAULT_TOP_K, initial_candidate_budget, MODEL_ID, MODEL_REVISION, SCORE_COLUMNS,
    RERANK_COLUMNS, ArtifactError, build_export_metadata, build_exports, compute_score_matrix,
    embed_carriers, manifest_signature, rank_results, select_candidates, apply_rerank, candidate_signature,
)
from rerank_logic import RerankConfig, score_candidates
from rerank_state import invalidate, reconcile, publish, fail, reranker_metadata
from app_resources import load_artifacts_cached, load_query_model_cached, load_reranker_cached, PLOT_LOCK

st.set_page_config(page_title="Discursos · reranker local", page_icon=":material/search:", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_OUTPUT_DIR = PROJECT_ROOT / "salida-embeddings-bge-m3"
COUNTRY_LABELS = {"arg": "Argentina", "chile": "Chile"}
DEFAULT_CARRIERS = (
    "La corrupción electoral altera la voluntad popular mediante fraude, compra de votos o manipulación de los comicios.",
    "Las prácticas corruptas en las elecciones incluyen el uso ilegítimo de recursos, el fraude y la presión sobre los votantes.",
    "La manipulación del voto y de los resultados electorales constituye una forma de corrupción política.",
    "El uso de dinero, cargos o poder para falsear la competencia electoral expresa corrupción en el proceso democrático.",
)


def country_label(value):
    return COUNTRY_LABELS.get(value, value)


def sanitize_filename(text):
    normalized = text.lower()
    for source, target in zip("áéíóúñ", "aeioun"):
        normalized = normalized.replace(source, target)
    return (re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "busqueda_semantica")[:60]


def initialize_session_state(identity):
    if st.session_state.get("corpus_identity") != identity:
        st.session_state.pop("analysis", None)
        st.session_state.pop("analysis_error", None)
        invalidate(st.session_state)
        st.session_state["corpus_identity"] = identity
    st.session_state.setdefault("results_page", 1)
    st.session_state.setdefault("selected_embedding_row", None)


def render_corpus_summary(bundle):
    metadata = bundle.metadata
    config = bundle.manifest["compatibility_config"]
    with st.container(horizontal=True):
        st.metric("Chunks cargados", f"{len(metadata):,}", border=True)
        st.metric("Documentos", f"{metadata['doc_id'].nunique():,}", border=True)
        st.metric("Período", f"{int(metadata['anio'].min())}–{int(metadata['anio'].max())}", border=True)
        st.metric("Modelo del corpus", "BGE-M3 · FP16", border=True)
    st.caption(f"{MODEL_ID} · revisión {MODEL_REVISION[:12]}… · CLS · {config['max_length']:,} tokens · corpus validado por checksum")


def render_sidebar(bundle):
    metadata = bundle.metadata
    countries = sorted(metadata["pais"].astype(str).unique())
    presidents = sorted(metadata["presidente"].astype(str).unique())
    min_year, max_year = int(metadata["anio"].min()), int(metadata["anio"].max())
    with st.sidebar:
        st.header("Filtros históricos")
        selected_countries = st.multiselect("Países", countries, default=countries, format_func=country_label,
                                           key="countries", help="Una selección vacía significa todos los países.")
        years = (min_year, max_year) if min_year == max_year else st.slider(
            "Rango de años", min_year, max_year, (min_year, max_year), key="years")
        selected_presidents = st.multiselect("Presidentes", presidents, default=[], key="presidents",
                                            help="Vacío significa todos los presidentes.")
        cosine = st.slider("Rango de similitud de coseno", -1.0, 1.0, (-1.0, 1.0), step=0.01, key="cosine")
        top_k = st.slider("Máximo de resultados (K)", 10, MAX_TOP_K, DEFAULT_TOP_K, step=10, key="top_k")
        page_size = st.selectbox("Resultados por página", [10, 25, 50], index=1, key="page_size")
        st.header("Reranker local")
        enabled = st.toggle("Activar reranker local", value=False, key="reranker_enabled")
        st.session_state.setdefault("candidate_n", initial_candidate_budget(top_k))
        candidate_n = st.number_input("Candidatos por coseno (N)", min_value=1, max_value=MAX_CANDIDATES,
                                     step=5, key="candidate_n", help="N debe ser mayor o igual que K. Se evaluarán 4 × N pares. No se ajusta automáticamente un N manual.")
        offline = True
        st.caption("Aplicación sin conexión desde el inicio. Los modelos se cargan solo al analizar o aplicar. Para descargar modelos, ejecute verify_real_runtime.py provision en un proceso separado antes de abrir la aplicación.")
        with st.expander("Configuración del reranker"):
            batch_size = st.selectbox("Pares por lote", [4, 1, 2, 8], key="batch_size")
            max_length = st.selectbox("Tokens por par completo", [512, 256, 128], key="max_length")
            devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
            device = st.selectbox("Dispositivo del reranker", devices, key="rerank_device")
            st.caption("CPU: FP32. CUDA: FP16, únicamente cuando esté disponible.")
        st.caption("Presupuesto inicial conservador: K=10 / N=25, no instantáneo. En este equipo (CPU, 4 hilos), N=25 tardó 105–169 s en benchmarks; N=200, 786 s (unos 13 min). No es una ETA: otros textos y condiciones pueden tardar más o menos. N=1000 es un límite técnico, no una recomendación.")
    return {"countries": selected_countries, "year_range": years, "presidents": selected_presidents,
            "cosine_range": cosine, "top_k": top_k, "page_size": page_size, "candidate_n": int(candidate_n),
            "enabled": enabled, "offline": offline,
            "config": RerankConfig(device=device, batch_size=batch_size, max_length=max_length)}


def render_carrier_form():
    st.subheader("Definir el significado objetivo")
    st.caption("Introduzca cuatro formulaciones cercanas del mismo concepto. Los cambios sin enviar el formulario son borradores; la consulta vigente es la última analizada.")
    with st.form("carrier_form", border=True):
        first, second = st.columns(2)
        values = {}
        for index in (0, 2):
            with first:
                values[index] = st.text_area(f"Carrier {index + 1}", DEFAULT_CARRIERS[index], height=130, key=f"carrier_{index + 1}")
        for index in (1, 3):
            with second:
                values[index] = st.text_area(f"Carrier {index + 1}", DEFAULT_CARRIERS[index], height=130, key=f"carrier_{index + 1}")
        submitted = st.form_submit_button("Analizar cuatro carriers", type="primary", width="stretch")
    return submitted, [values[i] for i in range(4)]


def run_analysis(bundle, carrier_texts, offline):
    # Invalidate before validation/loading: even failed submissions discard reranker scores.
    invalidate(st.session_state)
    if len(carrier_texts) != CARRIER_COUNT or any(not text.strip() for text in carrier_texts):
        raise ValueError("Complete las cuatro carrier sentences antes de analizar.")
    config = bundle.manifest["compatibility_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "fp16" if device.type == "cuda" else "fp32"
    with st.spinner("Cargando el modelo de embeddings y calculando las cuatro carriers…"):
        resource = load_query_model_cached(MODEL_ID, MODEL_REVISION, device.type, precision,
                                          config["max_length"], _local_files_only=offline)
        with resource.lock:
            carrier_embedding = embed_carriers(carrier_texts, resource.tokenizer, resource.model, device, config["max_length"])
        scores = compute_score_matrix(bundle.embeddings, carrier_embedding)
    st.session_state["analysis"] = {"carriers": tuple(text.strip() for text in carrier_texts),
                                     "carrier_embedding": carrier_embedding, "score_matrix": scores,
                                     "device": str(device), "precision": precision}
    st.session_state.pop("analysis_error", None)


def render_carrier_diagnostics(analysis):
    diagnostics = analysis["carrier_embedding"].diagnostics
    with st.expander("Consulta vigente y cohesión de las carriers"):
        for i, text in enumerate(analysis["carriers"], 1):
            st.write(f"**Carrier {i}:** {text}")
        st.caption("La cohesión no es una etiqueta temática ni una validación humana.")
        st.metric("Similitud media entre frases", f"{diagnostics['mean_pairwise_cosine']:.4f}")
        st.metric("Similitud mínima entre frases", f"{diagnostics['min_pairwise_cosine']:.4f}")
        st.metric("Norma del centroide", f"{diagnostics['centroid_norm']:.4f}")
        st.dataframe(pd.DataFrame({"Frase": [f"Carrier {i}" for i in range(1, 5)],
                                   "Similitud con el centroide": diagnostics["carrier_to_centroid"],
                                   "Tokens": analysis["carrier_embedding"].token_counts}), hide_index=True, width="stretch")


def result_table(results):
    columns = ["embedding_row", "anio", "pais", "presidente", "chunk_id"] + SCORE_COLUMNS
    columns += [c for c in RERANK_COLUMNS + ["retrieval_rank", "final_rank"] if c in results]
    table = results[columns].copy()
    table["pais"] = table["pais"].map(country_label)
    labels = {"anio": "Año", "pais": "País", "presidente": "Presidente", "chunk_id": "Chunk",
              "similitud_carrier": "Coseno del centroide", "rerank_score": "Logit promedio",
              "retrieval_rank": "Puesto original", "final_rank": "Puesto final"}
    labels.update({f"similitud_carrier_{i}": f"Coseno carrier {i}" for i in range(1, 5)})
    labels.update({f"rerank_score_carrier_{i}": f"Logit carrier {i}" for i in range(1, 5)})
    return table.rename(columns=labels)


def render_selected_chunk(selected):
    with st.container(border=True):
        st.subheader("Fragmento seleccionado")
        st.caption(f"{country_label(str(selected['pais']))} · {int(selected['anio'])} · {selected['presidente']} · chunk {selected['chunk_id']} · embedding_row {selected['embedding_row']}")
        if "rerank_score" in selected:
            st.write(f"**Logit promedio:** {selected['rerank_score']:.6f} · **Puesto original:** {int(selected['retrieval_rank'])} · **Puesto final:** {int(selected['final_rank'])}")
            st.caption("Logits crudos, no probabilidades ni similitudes coseno.")
            st.dataframe(pd.DataFrame({"Carrier": [1, 2, 3, 4],
                                       "Logit": [selected[f"rerank_score_carrier_{i}"] for i in range(1, 5)]}), hide_index=True)
        st.write(selected["texto_chunk"])
        profile = pd.DataFrame({"Carrier": [f"Carrier {i}" for i in range(1, 5)],
                                "Similitud coseno": [selected[f"similitud_carrier_{i}"] for i in range(1, 5)]}).set_index("Carrier")
        st.bar_chart(profile, y="Similitud coseno", height=250)


def render_graphs(results, effective_method):
    """Preserve original chunk/president scatter semantics; serialize Matplotlib access."""
    if results.empty:
        return
    st.subheader("Cercanía semántica a lo largo del tiempo")
    st.caption(f"Similitud coseno de los resultados finales seleccionados por {'reranker' if effective_method == 'reranker' else 'coseno'}. Este subconjunto no representa frecuencia histórica.")
    plot_df = results[["anio", "pais", "presidente", "similitud_carrier"]].copy().reset_index(drop=True)
    plot_df["pais"] = plot_df["pais"].map(country_label)
    palette = {country_label("arg"): "blue", country_label("chile"): "red"}
    palette.update({p: "gray" for p in plot_df["pais"].unique() if p not in palette})
    tab_chunks, tab_pres = st.tabs(["Por fragmento", "Por presidente"])
    with PLOT_LOCK:
        for tab, by_president in ((tab_chunks, False), (tab_pres, True)):
            with tab:
                data = plot_df
                size = None
                if by_president:
                    average = plot_df.groupby("presidente")["similitud_carrier"].mean().rename("similitud_promedio").reset_index()
                    data = plot_df.loc[plot_df.groupby("presidente")["similitud_carrier"].idxmax()].merge(average, on="presidente")
                    size = "similitud_promedio"
                fig, ax = plt.subplots(figsize=(15, 8))
                try:
                    kwargs = {"size": size, "sizes": (60, 900)} if by_president else {"s": 80}
                    sns.scatterplot(data=data, x="anio", y="similitud_carrier", hue="pais", palette=palette,
                                    alpha=0.65, ax=ax, **kwargs)
                    ax.set_title("Similitud de coseno por presidente (máximo y promedio)" if by_president else "Similitud de coseno por fragmento y año")
                    ax.set_xlabel("Año del chunk más similar" if by_president else "Año")
                    ax.set_ylabel("Similitud de coseno")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    labels = data.loc[data.groupby("presidente")["similitud_carrier"].idxmax()]
                    texts = [ax.text(row["anio"], row["similitud_carrier"], row["presidente"], fontsize=8)
                             for _, row in labels.iterrows()]
                    if texts:
                        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5})
                    low, high = float(data["similitud_carrier"].min()), float(data["similitud_carrier"].max())
                    ax.set_ylim(max(-1.05, low - 0.05), min(1.05, high + 0.05))
                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                finally:
                    plt.close(fig)


def render_results(bundle, analysis, filters):
    args = (bundle.metadata, analysis["score_matrix"], filters["countries"], filters["year_range"],
            filters["presidents"], filters["cosine_range"])
    fallback = rank_results(*args, filters["top_k"])
    config = filters["config"]
    candidate_error = None
    try:
        ranked = select_candidates(*args, filters["candidate_n"], filters["top_k"])
        candidates = ranked.rows
    except ValueError as error:
        candidate_error = str(error)
        candidates = fallback.rows.iloc[0:0].copy()
        candidates["retrieval_rank"] = np.asarray([], dtype=np.int64)
        st.error(candidate_error)
    config_identity = {**config.metadata(), "local_files_only": filters["offline"]}
    signature = candidate_signature(st.session_state["corpus_identity"], analysis["carriers"], candidates,
                                    filters, filters["candidate_n"], config_identity)
    status, record = reconcile(st.session_state, signature, filters["enabled"])
    st.caption(f"N solicitado: {filters['candidate_n']} · N efectivo: {len(candidates)} · Evaluaciones previstas: {4 * len(candidates)} · K: {filters['top_k']}")
    apply_clicked = st.button("Aplicar reranker", key="apply_reranker", type="primary",
                              disabled=not filters["enabled"] or candidates.empty or candidate_error is not None)
    if apply_clicked:
        # Remove the previous successful output before any loader/inference can fail.
        fail(st.session_state, signature, "Evaluación no completada")
        try:
            with st.spinner("Cargando el reranker local y evaluando cuatro pares por candidato…"):
                resource = load_reranker_cached(config, _local_files_only=filters["offline"])
                output = score_candidates(resource, analysis["carriers"],
                                          candidates["embedding_row"].tolist(), candidates["texto_chunk"].tolist())
                apply_rerank(candidates, output, filters["top_k"])
            publish(st.session_state, signature, output)
        except Exception as error:
            fail(st.session_state, signature, error)
        status, record = reconcile(st.session_state, signature, filters["enabled"])
    if candidate_error and filters["enabled"]:
        status, record = "error", {"error": candidate_error}
    output = record.get("output") if status == "applied" and record else None
    results = apply_rerank(candidates, output, filters["top_k"]) if output else fallback.rows
    labels = {"disabled": "desactivado", "pending": "pendiente", "applied": "aplicado", "error": "error"}
    st.caption(f"Estado del reranker: {labels[status]}")
    if status == "pending":
        st.info("Orden por coseno. Pulse ‘Aplicar reranker’ para evaluar los candidatos vigentes.")
    elif status == "error":
        st.error(f"Reranker no aplicado: {record.get('error')}. Se conserva el orden por coseno; no hay reintentos automáticos.")
    elif status == "applied":
        st.success(f"Reranker aplicado: {output.metrics['evaluated_candidates']} candidatos · {output.metrics['pair_evaluations']} pares · {output.metrics['elapsed_seconds']:.2f} s · {output.metrics['truncated_pairs']} pares recortados.")
        st.caption("Promedio de cuatro logits crudos. No es una probabilidad ni una mejora de relevancia validada.")
    reranker_info = reranker_metadata(config.metadata(), status, filters["candidate_n"], len(candidates),
                                      filters["top_k"], output, record.get("error") if status == "error" else None,
                                      filters["offline"])
    display_signature = hashlib.sha256((signature + status + results.to_json(orient="split", index=False)).encode()).hexdigest()
    if st.session_state.get("display_signature") != display_signature:
        st.session_state["results_page"] = 1
        st.session_state["selected_embedding_row"] = None
        st.session_state["display_signature"] = display_signature
    with st.container(horizontal=True):
        st.metric("Coincidencias en rango", f"{fallback.similarity_match_count:,}", border=True)
        st.metric("Candidatos evaluados", str(reranker_info["evaluated_candidates"]), border=True)
        st.metric("Resultados mostrados", f"{len(results):,}", border=True)
    if results.empty:
        st.warning("No hay chunks dentro del rango y los filtros. Amplíe el rango de coseno o el período histórico.")
    else:
        render_graphs(results, reranker_info["effective_method"])
        page_size = int(filters["page_size"])
        page_count = max(1, (len(results) + page_size - 1) // page_size)
        st.session_state["results_page"] = min(max(1, int(st.session_state["results_page"])), page_count)
        page = st.number_input(f"Página de resultados (1–{page_count})", min_value=1, max_value=page_count,
                               step=1, key="results_page")
        start = (int(page) - 1) * page_size
        page_results = results.iloc[start:start + page_size]
        selection = st.dataframe(result_table(page_results), hide_index=True, width="stretch", height=430,
                                 on_select="rerun", selection_mode="single-row",
                                 key=f"results_{display_signature}_{page}_{page_size}")
        selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
        position = min(max(0, int(selected_rows[0])), len(page_results) - 1) if selected_rows else 0
        selected = page_results.iloc[position]
        st.session_state["selected_embedding_row"] = int(selected["embedding_row"])
        render_selected_chunk(selected)
    export_filters = {key: filters[key] for key in ("countries", "year_range", "presidents", "cosine_range", "top_k")}
    metadata = build_export_metadata(analysis["carriers"], analysis["carrier_embedding"], bundle.manifest,
                                     export_filters, analysis["device"], analysis["precision"],
                                     fallback.similarity_match_count, len(results), reranker_info)
    metadata["analysis"] = {"using_previous_after_failure": bool(st.session_state.get("analysis_error"))}
    exports = build_exports(results, metadata, cosine_results=fallback.rows)
    # Session-owned snapshots are useful for audit and tests, never global score caches.
    st.session_state["final_results"] = results.copy()
    st.session_state["export_metadata"] = metadata
    st.session_state["exports"] = exports
    slug = sanitize_filename(analysis["carriers"][0])
    with st.container(horizontal=True):
        for label, kind, prefix, mime in (("Descargar CSV top-k", "csv", "resultados", "text/csv"),
                                          ("Descargar configuración JSON", "json", "configuracion", "application/json"),
                                          ("Descargar textos TXT", "txt", "textos", "text/plain")):
            st.download_button(label, exports[kind], file_name=f"{prefix}_{slug}.{kind}", mime=mime,
                               width="stretch", on_click="ignore")


st.title("Corrupción como problema histórico · versión con reranker local")
st.caption("Aplicación independiente. Cuatro carriers → coseno → N candidatos → reranker local → K resultados.")
st.info("La proximidad semántica y los logits no constituyen etiquetas humanas, pruebas de corrupción, inferencias de intención ni mediciones ideológicas.")
try:
    signature = manifest_signature(EMBEDDINGS_OUTPUT_DIR)
    corpus = load_artifacts_cached(str(EMBEDDINGS_OUTPUT_DIR), signature)
except (ArtifactError, OSError) as error:
    st.error(f"No se puede cargar el corpus validado: {error}")
    st.stop()
identity = json.dumps({"path": str(corpus.output_dir), "signature": signature,
                       "fingerprint": corpus.manifest["compatibility_fingerprint"],
                       "checksums": corpus.manifest.get("artifact_checksums_sha256")}, sort_keys=True)
initialize_session_state(identity)
render_corpus_summary(corpus)
filters = render_sidebar(corpus)
submitted, carrier_texts = render_carrier_form()
if submitted:
    try:
        run_analysis(corpus, carrier_texts, filters["offline"])
        st.success("Consulta generada. Los filtros no vuelven a ejecutar los modelos.")
    except Exception as error:
        st.session_state["analysis_error"] = str(error)
        st.error(f"El nuevo análisis falló: {error}")
analysis = st.session_state.get("analysis")
if analysis is None:
    st.info("Complete las cuatro carriers y presione ‘Analizar cuatro carriers’ para comenzar.")
else:
    if st.session_state.get("analysis_error"):
        st.warning("Se muestra la consulta anterior porque el nuevo análisis falló. El reranking anterior fue invalidado.")
    render_carrier_diagnostics(analysis)
    render_results(corpus, analysis, filters)
