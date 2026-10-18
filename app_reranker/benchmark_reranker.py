"""Opt-in real-corpus local/offline benchmark. Never run by mocked test suites."""
from __future__ import annotations

import argparse
import gc
import json
import threading
import time
from pathlib import Path

from runtime import APP_ROOT, configure_runtime

DEFAULT_CARRIERS = (
    "La corrupción electoral altera la voluntad popular mediante fraude, compra de votos o manipulación de los comicios.",
    "Las prácticas corruptas en las elecciones incluyen el uso ilegítimo de recursos, el fraude y la presión sobre los votantes.",
    "La manipulación del voto y de los resultados electorales constituye una forma de corrupción política.",
    "El uso de dinero, cargos o poder para falsear la competencia electoral expresa corrupción en el proceso democrático.",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true", help="Deprecated: provision separately with verify_real_runtime.py provision")
    parser.add_argument("--budgets", default="25,50,100,200")
    parser.add_argument("--carriers-file", type=Path, help="Read a JSON array of four Spanish carrier strings")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=APP_ROOT / ".cache/evaluation/benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.allow_download:
        raise ValueError("Provision models in a separate process with verify_real_runtime.py provision; benchmarks are strictly offline.")
    output_path = args.output.resolve()
    if not output_path.is_relative_to(APP_ROOT):
        raise ValueError("Benchmark output must be under app_reranker")
    budgets = [int(v) for v in args.budgets.split(",")]
    if not budgets or any(not 10 <= n <= 1000 for n in budgets) or args.threads < 1:
        raise ValueError("Budgets must be 10..1000 and CPU threads positive")
    configure_runtime()
    from launch import validate_interpreter
    validate_interpreter()
    import psutil
    import torch
    from search_logic import (load_artifacts, embed_carriers, compute_score_matrix, select_candidates,
                              rank_results, apply_rerank)
    from generar_embeddings_bge_m3_colab import load_model, MAX_LENGTH
    from rerank_logic import RerankConfig, load_reranker, score_candidates

    carriers = json.loads(args.carriers_file.read_text(encoding="utf-8")) if args.carriers_file else list(DEFAULT_CARRIERS)
    if not isinstance(carriers, list) or len(carriers) != 4 or any(not isinstance(t, str) or not t.strip() for t in carriers):
        raise ValueError("Four nonempty carrier strings are required")
    torch.set_num_threads(args.threads)
    process = psutil.Process()
    stop = threading.Event()
    peak = {"rss_bytes": process.memory_info().rss}

    def sample():
        while not stop.wait(0.05):
            peak["rss_bytes"] = max(peak["rss_bytes"], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        bundle = load_artifacts(APP_ROOT.parent / "salida-embeddings-bge-m3")
        started = time.perf_counter()
        tokenizer, query_model = load_model(torch.device("cpu"), "fp32", MAX_LENGTH, not args.allow_download)
        query_load_seconds = time.perf_counter() - started
        embedding = embed_carriers(carriers, tokenizer, query_model, torch.device("cpu"))
        scores = compute_score_matrix(bundle.embeddings, embedding)
        years = (int(bundle.metadata["anio"].min()), int(bundle.metadata["anio"].max()))
        config = RerankConfig(batch_size=args.batch_size, max_length=args.max_length)
        started = time.perf_counter()
        resource = load_reranker(config, not args.allow_download, str(APP_ROOT / ".cache/huggingface/hub"))
        load_seconds = time.perf_counter() - started
        runs = []
        for n in budgets:
            candidates = select_candidates(bundle.metadata, scores, [], years, [], (-1.0, 1.0), n, 10).rows
            evaluated = score_candidates(resource, carriers, candidates["embedding_row"].tolist(), candidates["texto_chunk"].tolist())
            reranked = apply_rerank(candidates, evaluated, 10)
            cosine = rank_results(bundle.metadata, scores, [], years, [], (-1.0, 1.0), 10).rows
            runs.append({"requested_n": n, **evaluated.metrics,
                         "cosine_top10": cosine.to_dict(orient="records"),
                         "reranker_top10": reranked.to_dict(orient="records"),
                         "rss_bytes_after": process.memory_info().rss})
        # Keep the embeddings model resident, release the first reranker, reload strictly offline.
        del resource
        gc.collect()
        started = time.perf_counter()
        offline_resource = load_reranker(config, True, str(APP_ROOT / ".cache/huggingface/hub"))
        reload_seconds = time.perf_counter() - started
        smoke = score_candidates(offline_resource, carriers, candidates["embedding_row"].tolist()[:1],
                                 candidates["texto_chunk"].tolist()[:1])
        report = {"status": "complete", "carriers": carriers, "corpus": bundle.manifest["compatibility_fingerprint"],
                  "config": config.metadata(), "torch_version": torch.__version__,
                  "cpu_threads": torch.get_num_threads(), "embedding_model_resident": True,
                  "query_load_seconds": query_load_seconds, "reranker_load_seconds": load_seconds,
                  "offline_reload_seconds": reload_seconds, "offline_smoke": smoke.metrics,
                  "runs": runs, "memory": {**peak, "sample_interval_seconds": 0.05,
                    "os_peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None)},
                  "relevance": "Not measured: precision@10/nDCG@10 require human relevance judgments"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(str(output_path))
    finally:
        stop.set()
        sampler.join()


if __name__ == "__main__":
    main()
