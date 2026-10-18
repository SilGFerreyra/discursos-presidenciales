# Generar embeddings BGE-M3 reproducibles

Este flujo transforma los 87.546 registros de `df-chunks-2026-09-10.csv` en embeddings densos normalizados de 1024 dimensiones. Está preparado para Google Colab con GPU, guarda checkpoints reanudables y conserva una relación inequívoca entre cada fila y su vector.

## Ejecución en Colab

1. Seleccioná **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU**.
2. Subí `generar_embeddings_bge_m3_colab.py`, `requirements-embeddings.txt` y `df-chunks-2026-09-10.csv`, o montá una carpeta de Google Drive que los contenga.
3. Montá Google Drive para que los checkpoints sobrevivan al cierre de la sesión.

```python
from google.colab import drive
drive.mount("/content/drive")
```

4. Instalá las dependencias sin reemplazar el PyTorch CUDA incluido por Colab.

```python
!pip install -q -r requirements-embeddings.txt
```

5. Ejecutá primero una prueba real de 32 chunks.

```python
%run generar_embeddings_bge_m3_colab.py \
  --input df-chunks-2026-09-10.csv \
  --output-dir /content/drive/MyDrive/salida-embeddings-bge-m3-prueba \
  --device cuda \
  --precision fp16 \
  --limit 32 \
  --batch-size 32 \
  --shard-size 32 \
  --overwrite
```

6. Si la prueba termina con `status: complete`, ejecutá el corpus completo.

```python
%run generar_embeddings_bge_m3_colab.py \
  --input df-chunks-2026-09-10.csv \
  --output-dir /content/drive/MyDrive/salida-embeddings-bge-m3 \
  --device cuda \
  --precision fp16 \
  --batch-size 32 \
  --shard-size 1024
```

Si la sesión se interrumpe, repetí exactamente el último comando. Los shards completos se validan y se omiten. Ante un error de memoria CUDA, el pipeline reduce el batch automáticamente. Usá `--overwrite` solamente cuando quieras descartar toda una ejecución anterior.

La compatibilidad de los checkpoints depende del corpus y de las decisiones que determinan el embedding, no de la GPU asignada. Por eso una nueva sesión puede continuar shards generados en una T4, L4 o A100. El manifiesto y cada shard conservan la GPU, CUDA, librerías, batch y hash del script como procedencia auditable. Distintas GPUs en FP16 pueden producir diferencias numéricas mínimas; todos los vectores se validan y normalizan antes de aceptarlos.

Las salidas creadas con una versión anterior del manifiesto no se mezclan automáticamente. Para descartarlas, usá `--overwrite`; para conservarlas, elegí otro `--output-dir`. Reservá aproximadamente 1,2 GiB libres en Drive para shards y artefactos consolidados.

## Decisiones técnicas

| Tema | Implementación |
|---|---|
| Modelo | `BAAI/bge-m3`, revisión fija `5617a9f61b028005a4858fdac845db406aefb181` |
| Pooling | CLS explícito: `last_hidden_state[:, 0, :]` |
| Verificación | El script comprueba que la configuración oficial activa CLS, desactiva mean pooling y declara 1024 dimensiones |
| Normalización | L2 explícita en float32 después de medir la norma cruda |
| Prefijos | Ninguno; el texto se entrega al tokenizador sin modificaciones |
| Longitud máxima | 8192 tokens, con conteo previo sin truncamiento |
| Precisión | FP16 en CUDA y FP32 en CPU; los vectores finales siempre se guardan en float32 |
| Batch | 32 por defecto en GPU y 8 en CPU; se puede ajustar |
| Persistencia | Shards `.npz` con procedencia, matriz consolidada `.npy`, tabla Parquet y metadatos CSV |

No se usa `SentenceTransformer.encode()`. Se usa `transformers.AutoModel` para poder medir la norma del CLS **antes** de normalizar. El resultado matemático final equivale a aplicar `normalize_embeddings=True`, pero conserva además el control de calidad solicitado.

## Columnas agregadas

| Columna | Significado |
|---|---|
| `embedding_row` | Fila correspondiente en `embeddings-chunks-bge-m3.npy` |
| `embedding_chunk` | Vector float32 de 1024 dimensiones; está incluido en el Parquet |
| `embedding_norm` | Norma L2 cruda del CLS antes de normalizar |
| `embedding_norm_normalized` | Norma del vector final; debe ser aproximadamente 1 |
| `n_tokens_chunk` | Tokens reales, incluidos los tokens especiales y antes de truncar |
| `embedding_flag` | `ok` o `error` si la norma/vector no es válido |

La norma previa no debe esperarse cercana a 1. Esa condición corresponde a `embedding_norm_normalized`.

## Archivos de salida

| Archivo | Uso |
|---|---|
| `df-chunks-emb-2026-09-10.parquet` | DataFrame completo, incluida `embedding_chunk` |
| `embeddings-chunks-bge-m3.npy` | Matriz canónica `(87546, 1024)` para búsquedas rápidas |
| `df-chunks-emb-metadata-2026-09-10.csv` | Metadatos y controles sin repetir los vectores |
| `reporte-embeddings-bge-m3.json` | Configuración compatible, procedencias, tiempos y métricas de calidad |
| `manifest-embeddings-bge-m3.json` | Huella compatible, historial de ejecuciones y checksums finales |
| `shards/*.npz` | Checkpoints parciales con procedencia del runtime que los generó |

El `.npy` evita convertir 1024 números por fila a texto CSV. `embedding_row` mantiene la correspondencia exacta con los metadatos.

## Carrier sentences

Codificá exactamente cuatro formulaciones semánticamente cercanas del mismo concepto con el mismo modelo, CLS, normalización, precisión y ausencia de prefijos. No son cuatro categorías: funcionan como cuatro puntos de apoyo para cubrir diversidad expresiva. Después promediá sus vectores y normalizá nuevamente el promedio:

```python
import numpy as np
import pandas as pd
import torch
import hashlib
import json
from pathlib import Path

from generar_embeddings_bge_m3_colab import encode_batch, load_model

device = torch.device("cuda")
tokenizer, model = load_model(device, "fp16", 8192, local_files_only=False)

oraciones = [
    "La corrupción electoral altera la voluntad popular mediante fraude, compra de votos o manipulación de los comicios.",
    "Las prácticas corruptas en las elecciones incluyen el uso ilegítimo de recursos, el fraude y la presión sobre los votantes.",
    "La manipulación del voto y de los resultados electorales constituye una forma de corrupción política.",
    "El uso de dinero, cargos o poder para falsear la competencia electoral expresa corrupción en el proceso democrático.",
]

if len(oraciones) != 4:
    raise ValueError("La carrier debe contener exactamente cuatro oraciones.")

vectores, _, _, _, flags = encode_batch(oraciones, tokenizer, model, device, 8192)
if not np.all(flags == "ok"):
    raise RuntimeError("Falló el embedding de una oración de la carrier.")

carrier = vectores.mean(axis=0)
carrier /= np.linalg.norm(carrier)

base = Path("/content/drive/MyDrive/salida-embeddings-bge-m3")
manifest = json.loads((base / "manifest-embeddings-bge-m3.json").read_text())
if manifest["status"] != "complete":
    raise RuntimeError("La generación de embeddings no está completa.")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()

for nombre, esperado in manifest["artifact_checksums_sha256"].items():
    if sha256(base / nombre) != esperado:
        raise RuntimeError(f"Checksum inválido: {nombre}")

matriz = np.load(base / "embeddings-chunks-bge-m3.npy", mmap_mode="r")
metadata = pd.read_csv(base / "df-chunks-emb-metadata-2026-09-10.csv")
if len(metadata) != len(matriz) or not np.array_equal(metadata["embedding_row"], np.arange(len(matriz))):
    raise RuntimeError("Los metadatos no corresponden a la matriz de embeddings.")

scores = matriz @ carrier
top_k = 20
indices = np.argpartition(scores, -top_k)[-top_k:]
indices = indices[np.argsort(scores[indices])[::-1]]
resultados = metadata.iloc[indices].copy()
resultados["similitud_coseno"] = scores[indices]
resultados[["doc_id", "chunk_id", "texto_chunk", "similitud_coseno"]]
```

Promediar sin renormalizar produce un vector cuya magnitud depende de la dispersión entre las cuatro oraciones. La renormalización final es necesaria para que el producto punto siga siendo similitud de coseno. Si la similitud entre las frases es baja, revisá que sigan expresando el mismo concepto y no cuatro categorías diferentes.

## Controles ejecutados localmente

Se auditó el tokenizador oficial sobre todo el corpus:

- Chunks: 87.546.
- Tokens mínimos: 49.
- Mediana: 79.
- Percentil 95: 99.
- Máximo: 236.
- Chunks que superarían 8192 tokens: 0.

También se ejecutó BGE-M3 real en CPU sobre ocho chunks:

- Forma: `(8, 1024)`.
- Errores: 0.
- Truncamientos: 0.
- Norma cruda: 24,9243–26,0807.
- Norma final: 0,99999994–1,0.
- Parquet y `.npy`: vectores idénticos al recargarlos.

La máquina local no tiene GPU. Distintas corridas del batch de ocho tardaron entre 4,63 y 10,25 segundos solamente en inferencia, lo que proyecta aproximadamente 14–31 horas para el corpus completo. Por eso la corrida integral está bloqueada en CPU por defecto y debe ejecutarse en Colab GPU.

La reanudación también fue verificada: una segunda ejecución con la misma configuración compatible y distinta procedencia detectó el shard existente y no repitió la inferencia. Una interrupción solo obliga a repetir el shard incompleto, de hasta 1.024 filas con la configuración recomendada. La app debe aplicar el mismo procedimiento con cuatro oraciones de carrier; el promedio renormalizado debe tener norma 1 antes de producir similitudes por producto punto.

## Verificación final

Antes de usar los embeddings, comprobá en `reporte-embeddings-bge-m3.json`:

- `status == "complete"`.
- `embedding_shape == [87546, 1024]`.
- `embedding_errors == 0`.
- `normalized_norms_outside_tolerance == 0`.
- `chunks_truncated == 0`.
- La revisión del modelo y el SHA-256 del CSV coinciden con la ejecución esperada.

Las pruebas automatizadas del pooling CLS, normalización, errores, identidad de shards y equivalencia entre NPY y Parquet se ejecutan con:

```python
!python -m unittest -v test_generar_embeddings_bge_m3.py
```

## Referencias verificadas

- Modelo y uso oficial: <https://huggingface.co/BAAI/bge-m3>
- Configuración CLS: <https://huggingface.co/BAAI/bge-m3/blob/main/1_Pooling/config.json>
- Módulos del modelo: <https://huggingface.co/BAAI/bge-m3/blob/main/modules.json>
- Documentación de Transformers: <https://huggingface.co/docs/transformers/>
