"""Verifica que el corpus y la matriz de embeddings coincidan con sus SHA256 esperados.

Uso (desde la raíz del repositorio):  python verificar_datos.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "salida-embeddings-bge-m3"
CORPUS = ROOT / "df-chunks-2026-09-10.csv"
CORPUS_SHA256 = "6bdd12428df44ffcff9397edeb53833953f79570c8186e77da6cdeed42612365"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((OUT / "manifest-embeddings-bge-m3.json").read_text(encoding="utf-8"))
    expected = {CORPUS: CORPUS_SHA256}
    for name in ("embeddings-chunks-bge-m3.npy", "df-chunks-emb-metadata-2026-09-10.csv"):
        expected[OUT / name] = manifest["artifact_checksums_sha256"][name]
    ok = True
    for path, want in expected.items():
        rel = path.relative_to(ROOT)
        if not path.is_file():
            print(f"FALTA     {rel}")
            ok = False
            continue
        got = sha256(path)
        status = "OK" if got == want else "DISTINTO"
        ok &= got == want
        print(f"{status:<9} {rel}")
    if not ok:
        print("\nAlgún archivo falta o no coincide. Si es la matriz .npy, descárguela de la Release "
              "(ver README). Si es un CSV, revise que Git no haya convertido finales de línea.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
