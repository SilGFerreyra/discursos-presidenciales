# Discursos presidenciales: búsqueda semántica con reranker local

Corpus segmentado de discursos presidenciales, embeddings BGE-M3 y una aplicación
Streamlit (`app_reranker/`) que busca pasajes con cuatro *carrier sentences* y
los reordena con el reranker `BAAI/bge-reranker-v2-m3`, todo en CPU y sin conexión
una vez descargados los modelos.

## Contenido

| Ruta | Qué es |
|---|---|
| `df-chunks-2026-09-10.csv` | Corpus segmentado: 87.546 fragmentos (42 MB). |
| `salida-embeddings-bge-m3/` | Manifiesto, metadatos por fragmento y reporte de los embeddings. |
| `salida-embeddings-bge-m3/embeddings-chunks-bge-m3.npy` | Matriz 87.546 × 1024 (359 MB). **No está en el repositorio: se descarga de la Release** (ver abajo). |
| `segmentar_corpus_colab.py`, `generar_embeddings_bge_m3_colab.py`, `contar_tokens_por_discurso_bge_m3.py` | Pipeline de segmentación, embeddings y conteo de tokens (pensado para Colab). Ver `FLUJO_EMBEDDINGS_BGE_M3.md`. |
| `app_reranker/` | La aplicación, sus tests y sus reportes de verificación. Detalles técnicos en `app_reranker/README.md`. |
| `verificar_datos.py` | Comprueba los SHA256 del corpus y de la matriz. |

La app importa `generar_embeddings_bge_m3_colab.py` y lee `salida-embeddings-bge-m3/`
desde la carpeta padre, así que **hay que conservar esta estructura de carpetas**.

## Instalación (Windows)

Requisitos: Python **3.12** de 64 bits, ~6 GB libres (modelos) y ~4 GB de RAM.

### 1. Clonar y descargar la matriz de embeddings

```powershell
git clone https://github.com/SilGFerreyra/discursos-presidenciales.git
cd discursos-presidenciales
```

Descargue `embeddings-chunks-bge-m3.npy` de la Release
[`datos-v1`](https://github.com/SilGFerreyra/discursos-presidenciales/releases/tag/datos-v1)
y guárdelo en `salida-embeddings-bge-m3\`. Con GitHub CLI:

```powershell
gh release download datos-v1 -R SilGFerreyra/discursos-presidenciales -D salida-embeddings-bge-m3
```

Luego verifique los datos:

```powershell
py -3.12 verificar_datos.py
```

Las tres líneas deben decir `OK`.

### 2. Crear el entorno de la app

El lanzador solo acepta el entorno en `app_reranker\.venv`:

```powershell
cd app_reranker
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(`requirements-lock.txt` lista las versiones exactas con las que se verificó.)

### 3. Descargar los modelos (una sola vez, con internet)

```powershell
.\.venv\Scripts\python.exe -B verify_real_runtime.py provision
```

Descarga `BAAI/bge-m3` y `BAAI/bge-reranker-v2-m3` en revisiones fijas a
`app_reranker\.cache\huggingface\` y verifica sus SHA256 (~4,5 GB).

### 4. Ejecutar

```powershell
.\ejecutar_app.bat
```

Abre la app en `http://localhost:8502` (o el siguiente puerto libre). La app funciona
sin conexión y nunca descarga modelos por su cuenta.

## Otros sistemas (Linux/macOS)

`ejecutar_app.bat` y `launch.py` están escritos para Windows. En otros sistemas, con el
entorno creado en `app_reranker/.venv` y los modelos descargados, se puede iniciar
Streamlit directamente desde `app_reranker/`:

```bash
.venv/bin/python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B verify_real_runtime.py provision
.venv/bin/python -B -m streamlit run streamlit_app.py --server.address 127.0.0.1
```
