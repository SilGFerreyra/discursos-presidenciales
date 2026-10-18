import unittest

import pandas as pd

from contar_tokens_por_discurso_bge_m3 import (
    TokenCountError,
    build_output,
    count_content_tokens,
    reconstruct_documents,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        if kwargs["add_special_tokens"]:
            return {"input_ids": [0, *range(len(text.split())), 1]}
        return {"input_ids": list(range(len(text.split())))}


def make_metadata():
    return pd.DataFrame(
        [
            {
                "doc_id": "speech.pdf",
                "chunk_id": 1,
                "texto_chunk": "uno uno dos tres cuatro cinco seis siete",
                "n_palabras": 8,
                "inicio_palabra": 1,
                "fin_palabra": 8,
                "overlap_previo": 0,
                "n_palabras_nuevas": 8,
                "chunk_flag": "ok",
                "nombre_pdf": "speech.pdf",
                "anio": 1900,
                "presidente": "Presidente",
                "pais": "arg",
                "source_row": 0,
            },
            {
                "doc_id": "speech.pdf",
                "chunk_id": 2,
                "texto_chunk": "dos tres cuatro cinco seis siete ocho nueve nueve diez",
                "n_palabras": 10,
                "inicio_palabra": 3,
                "fin_palabra": 12,
                "overlap_previo": 6,
                "n_palabras_nuevas": 4,
                "chunk_flag": "ok",
                "nombre_pdf": "speech.pdf",
                "anio": 1900,
                "presidente": "Presidente",
                "pais": "arg",
                "source_row": 0,
            },
            {
                "doc_id": "speech.pdf",
                "chunk_id": 3,
                "texto_chunk": "seis siete ocho nueve nueve diez once",
                "n_palabras": 7,
                "inicio_palabra": 7,
                "fin_palabra": 13,
                "overlap_previo": 6,
                "n_palabras_nuevas": 1,
                "chunk_flag": "ok",
                "nombre_pdf": "speech.pdf",
                "anio": 1900,
                "presidente": "Presidente",
                "pais": "arg",
                "source_row": 0,
            },
        ]
    )


def make_source():
    return pd.DataFrame(
        [
            {
                "nombre_pdf": "speech.pdf",
                "texto_embedding": "uno uno dos tres cuatro cinco seis siete ocho nueve nueve diez once",
                "anio": 1900,
                "presidente": "Presidente",
                "pais": "arg",
            }
        ]
    )


class TokenCountTests(unittest.TestCase):
    def test_reconstructs_without_repeating_overlap_words(self):
        result = reconstruct_documents(make_metadata(), make_source())

        self.assertEqual(
            result.iloc[0].texto_reconstruido,
            "uno uno dos tres cuatro cinco seis siete ocho nueve nueve diez once",
        )

    def test_rejects_invalid_overlap(self):
        metadata = make_metadata()
        metadata.loc[1, "overlap_previo"] = 1

        with self.assertRaises(TokenCountError):
            reconstruct_documents(metadata, make_source())

    def test_preserves_legitimate_repeated_words(self):
        metadata = make_metadata()
        source = make_source()

        result = reconstruct_documents(metadata, source)

        self.assertIn("nueve nueve", result.iloc[0].texto_reconstruido)

    def test_excludes_special_tokens_and_does_not_truncate(self):
        tokenizer = FakeTokenizer()
        text = " ".join(["palabra"] * 9000)

        self.assertEqual(count_content_tokens(text, tokenizer), 9000)
        self.assertEqual(tokenizer.calls[0]["add_special_tokens"], False)
        self.assertEqual(tokenizer.calls[0]["truncation"], False)
        self.assertEqual(tokenizer.calls[0]["padding"], False)

    def test_output_is_sorted_by_requested_keys(self):
        documents = pd.DataFrame(
            [
                {
                    "pais": "chile",
                    "presidente": "B",
                    "anio": 1901,
                    "doc_id": "b.pdf",
                    "texto_reconstruido": "uno dos",
                },
                {
                    "pais": "arg",
                    "presidente": "A",
                    "anio": 1900,
                    "doc_id": "a.pdf",
                    "texto_reconstruido": "uno",
                },
            ]
        )
        output = build_output(documents, FakeTokenizer())

        self.assertEqual(output.columns.tolist(), ["pais", "presidente", "anio", "cantidad_tokens", "doc_id"])
        self.assertEqual(output["doc_id"].tolist(), ["a.pdf", "b.pdf"])
        self.assertEqual(output["cantidad_tokens"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
