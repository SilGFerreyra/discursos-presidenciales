"""Per-session state transitions, independent of Streamlit and model imports."""


def invalidate(state):
    state.pop("rerank_record", None)
    state.pop("rerank_signature", None)
    state["results_page"] = 1
    state["selected_embedding_row"] = None
    state.pop("display_signature", None)
    for key in ("final_results", "exports", "export_metadata"):
        state.pop(key, None)


def reconcile(state, signature, enabled):
    if state.get("rerank_signature") != signature:
        state.pop("rerank_record", None)
        state["rerank_signature"] = signature
    record = state.get("rerank_record")
    if not enabled:
        return "disabled", None
    if record is None:
        return "pending", None
    return record["state"], record


def publish(state, signature, output):
    state["rerank_signature"] = signature
    state["rerank_record"] = {"state": "applied", "signature": signature, "output": output}


def fail(state, signature, error):
    state["rerank_signature"] = signature
    state["rerank_record"] = {"state": "error", "signature": signature, "error": str(error)}


def reranker_metadata(config, status, requested_n, effective_n, top_k, output=None, error=None,
                      local_files_only=True):
    applied = status == "applied" and output is not None
    payload = {
        **config, "requested_method": "cosine" if status == "disabled" else "reranker",
        "effective_method": "reranker" if applied else "cosine", "state": status,
        "requested_n": requested_n, "effective_n": effective_n, "top_k": top_k,
        "local_files_only": local_files_only, "evaluated_candidates": 0,
        "pair_evaluations": 0, "truncated_pairs": None, "elapsed_seconds": None,
    }
    if applied:
        payload.update(output.metrics)
    if error is not None:
        payload["error"] = str(error)
    return payload
