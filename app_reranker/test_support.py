"""Synthetic app-owned test fixtures. No network, weights, or protected-tree writes."""
from runtime import configure_runtime
configure_runtime()

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

from search_logic import ArtifactBundle, build_carrier_embedding, EMBEDDING_DIM


def make_metadata(rows=4):
    return pd.DataFrame({"doc_id": [f"doc-{i}.pdf" for i in range(rows)],
                         "chunk_id": list(range(rows)), "texto_chunk": [f"Texto sintético {i}" for i in range(rows)],
                         "anio": [1900 + i for i in range(rows)], "presidente": [f"Presidente {i % 2}" for i in range(rows)],
                         "pais": ["arg" if i % 2 == 0 else "chile" for i in range(rows)],
                         "embedding_row": list(range(rows))})


def make_embedding():
    vectors = np.zeros((4, EMBEDDING_DIM), dtype=np.float32)
    vectors[np.arange(4), np.arange(4)] = 1
    return build_carrier_embedding(vectors)


def make_scores(rows=4):
    return np.column_stack([np.linspace(0.9, 0.5, rows)] * 5).astype(np.float32)


def make_bundle(rows=120):
    manifest = {"compatibility_fingerprint": "synthetic", "compatibility_config": {
        "model_id": "BAAI/bge-m3", "model_revision": "synthetic-revision", "pooling": "CLS", "prefixes": "ninguno",
        "max_length": 8192, "rows": rows, "input_sha256": "synthetic-input"}}
    return ArtifactBundle(make_metadata(rows), np.zeros((rows, EMBEDDING_DIM), np.float32), manifest,
                          Path(__file__).resolve().parent / ".cache/synthetic-corpus")


class FakeTokenizer:
    model_max_length = 512

    def __init__(self):
        self.prepared = []
        self.padded_widths = []

    def __call__(self, texts, **kwargs):
        encode = lambda t: [10 + len(word) for word in t.split()]
        return {"input_ids": encode(texts) if isinstance(texts, str) else [encode(t) for t in texts]}

    def num_special_tokens_to_add(self, pair=True):
        return 3

    def build_inputs_with_special_tokens(self, query_ids, passage_ids):
        return [0] + list(query_ids) + [1] + list(passage_ids) + [2]

    def prepare_for_model(self, ids, pair_ids, **kwargs):
        self.prepared.append((list(ids), list(pair_ids)))
        values = self.build_inputs_with_special_tokens(ids, pair_ids)
        return {"input_ids": values, "attention_mask": [1] * len(values)}

    def pad(self, items, padding=True, return_tensors="pt"):
        width = max(len(item["input_ids"]) for item in items)
        self.padded_widths.append(width)
        return {key: torch.tensor([item[key] + [0] * (width - len(item[key])) for item in items])
                for key in ("input_ids", "attention_mask")}


class FakeModel:
    def __init__(self, values=None, fail_batch=None):
        self.values = list(values) if values is not None else list(range(1, 4001))
        self.offset = 0
        self.calls = 0
        self.fail_batch = fail_batch
        self.grad_flags = []
        self.config = SimpleNamespace(num_labels=1)
        self.to = Mock(return_value=self)
        self.eval = Mock(return_value=self)

    def __call__(self, **batch):
        self.calls += 1
        self.grad_flags.append(torch.is_grad_enabled())
        if self.calls == self.fail_batch:
            raise RuntimeError("synthetic inference failure")
        count = len(batch["input_ids"])
        values = self.values[self.offset:self.offset + count]
        self.offset += count
        return SimpleNamespace(logits=torch.tensor(values, dtype=torch.float32).reshape(-1, 1))
