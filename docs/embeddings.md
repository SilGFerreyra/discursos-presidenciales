# Corpus y metodología de búsqueda

La app utiliza **87.546 fragmentos de 297 discursos presidenciales de Argentina
y Chile**. Los metadatos conservan el documento, presidente, país, año, texto,
posiciones de palabras y la fila correspondiente en la matriz de embeddings.

## Segmentación del corpus publicado

La segmentación se realizó antes de distribuir la app. Los fragmentos tienen
entre **40 y 70 palabras**, con objetivo de 55. El solapamiento entre fragmentos
consecutivos de un documento es de **5 a 10 palabras**, con objetivo de 8,
priorizando límites de oración y otros cortes textuales.

La media observada es de 55,25 palabras por fragmento y 7,97 de solapamiento.
No se trata de ventanas fijas de 50 palabras con solapamiento de 5.
La app consume los fragmentos publicados; no vuelve a segmentar el corpus.

## Embeddings

| Parámetro | Valor |
|---|---|
| Modelo | `BAAI/bge-m3` |
| Revisión | `5617a9f61b028005a4858fdac845db406aefb181` |
| Dimensión | 1024 |
| Pooling | Primer token: `last_hidden_state[:, 0, :]` (CLS) |
| Normalización | L2 explícita en float32 |
| Prefijos | Ninguno |
| Máximo | 8192 tokens |
| Corpus publicado | Generado en GPU/FP16; matriz final en float32 |
| Consultas en CPU | FP32, con el mismo pooling y normalización |

Cada una de las cuatro carriers se codifica por separado. Se promedian sus
vectores y el promedio se normaliza otra vez. El producto punto con los embeddings
normalizados del corpus produce la similitud de coseno del centroide.

## Reranker

Se utiliza `BAAI/bge-reranker-v2-m3` en la revisión
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Primero se seleccionan N candidatos
por coseno y luego se evalúa cada candidato contra las cuatro carriers.
El resultado es el promedio de **cuatro logits crudos**, sin sigmoid.

El máximo de tokens se aplica al par completo; si es necesario, solo se recorta
el fragmento, preservando la carrier. Los valores iniciales son K=10 y N=25.

## Archivos e integridad

| Archivo en `salida-embeddings-bge-m3/` | Uso |
|---|---|
| `df-chunks-emb-metadata-2026-09-10.csv` | Textos y metadatos; `embedding_row` identifica el vector. |
| `embeddings-chunks-bge-m3.npy` | Matriz `(87546, 1024)` descargada de la release. |
| `manifest-embeddings-bge-m3.json` | Configuración, huella de compatibilidad, procedencia y SHA256. |
| `reporte-embeddings-bge-m3.json` | Reporte final de generación y calidad del corpus publicado. |

El manifiesto conserva la identidad del corpus de entrada y referencias a los
artefactos de su generación. La app solo necesita el CSV de metadatos, la matriz
y el manifiesto; no requiere el Parquet ni los checkpoints de esa ejecución.
El reporte se conserva como procedencia del conjunto de datos.

El verificador de la raíz comprueba los SHA256 de los dos archivos usados por
la app. Al cargarlos, la aplicación también valida orden de filas, dimensiones,
valores finitos y norma L2. Las revisiones de ambos modelos están fijadas y el
descargador verifica los tamaños y SHA256 de sus pesos antes de aceptar la descarga.

## Referencias

- [BGE-M3](https://huggingface.co/BAAI/bge-m3).
- [BGE reranker v2 M3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
- [Datos publicados](https://github.com/SilGFerreyra/discursos-presidenciales/releases/tag/datos-v1).
