# Estructura técnica y verificación

La [guía principal](../README.md) contiene la instalación y el uso de la app.
Todos los comandos de esta página se ejecutan desde `app_reranker/` con su entorno
`.venv`. No se requieren rutas personales ni archivos de una instalación anterior.

## Archivos de la aplicación

| Archivo | Responsabilidad |
|---|---|
| `streamlit_app.py` | Interfaz, filtros, navegación y descargas. |
| `app_resources.py` | Caché de corpus y modelos; bloqueos de inferencia compartida. |
| `embedding_logic.py` | Modelo BGE-M3 fijado, pooling CLS y normalización L2. |
| `search_logic.py` | Validación de artefactos, similitud, selección y exportaciones. |
| `rerank_logic.py` | Evaluación de los cuatro pares por candidato y promedio de logits. |
| `rerank_state.py` | Estado del reranker e invalidación de resultados anteriores. |
| `runtime.py` | Cachés locales y configuración del modo sin conexión antes de importar los modelos. |
| `download_models.py` | Descarga explícita de modelos y validación de sus pesos. |
| `launch.py`, `ejecutar_app.bat` | Validación del entorno y arranque en un puerto libre. |

Los datos se leen de `../salida-embeddings-bge-m3/`. La app comprueba los checksums,
la correspondencia entre filas y vectores, las dimensiones y la normalización.
La matriz se abre como un `numpy.memmap` de solo lectura.

## Dependencias

`requirements.txt` fija las bibliotecas principales y sus intervalos compatibles.
En Windows/Linux se instala PyTorch CPU primero, como indica la guía principal.
`requirements-lock.txt` registra el conjunto exacto validado en Windows/Python 3.12;
no se utiliza como lock multiplataforma.

## Tests automatizados

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B launch.py --check
```

En Linux/macOS, sustituya `.\.venv\Scripts\python.exe` por `.venv/bin/python`.

Los tests de lógica y de interfaz utilizan tensores y modelos sintéticos. No
descargan modelos ni realizan inferencia con pesos reales. El test de integridad
lee la matriz publicada, por lo que debe estar descargada antes de ejecutar toda
la suite. Se comprueban:

- pooling CLS, normalización y rechazo de consultas demasiado largas;
- revisiones, formatos, tamaños y hashes de las descargas de modelos;
- ranking, filtros, exportaciones y conservación de identificadores;
- aplicación explícita del reranker e invalidación de resultados obsoletos;
- arranque sin conexión en procesos nuevos y aislamiento del entorno local;
- interacción de Streamlit con `AppTest`.

## Comprobación opcional con BGE-M3 real

Después de descargar los modelos:

```powershell
.\.venv\Scripts\python.exe -B offline_startup_probe.py --entry launcher --real-model
```

El probe codifica cuatro frases con el modelo local y rechaza intentos de red.
Termina con una línea `OFFLINE_PROBE=` cuyo JSON debe indicar `"status": "passed"`.
Es una comprobación de inferencia, no una medición de calidad del ranking.

## Contratos de ejecución

- Solo la descarga explícita de modelos funciona en modo conectado.
- La app configura el modo sin conexión antes de importar Hugging Face y Transformers.
- Cambiar filtros o de página no vuelve a ejecutar los modelos.
- Los cambios de consulta invalidan los resultados anteriores.
- Los logits del reranker y el coseno son medidas distintas; no se mezclan ni se
  interpretan como probabilidades.
- La precisión temática del ranking necesita evaluación humana; los tests
  funcionales no prueban por sí solos relevancia histórica.
