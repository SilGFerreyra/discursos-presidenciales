# Discursos presidenciales: búsqueda semántica con reranker local

Aplicación para buscar pasajes de discursos presidenciales de Argentina y Chile
mediante cuatro *carrier sentences*. Utiliza embeddings BGE-M3 y un reranker
local, funciona en CPU y no necesita internet después de descargar los datos y
los modelos.

## Instalación en Windows

Requisitos: [Git](https://git-scm.com/download/win), Python **3.12 de 64 bits**,
aproximadamente **10 GB libres** y **8 GB de RAM** (16 GB recomendados).
La inferencia en CPU puede tardar varios minutos según el equipo y los textos.

### 1. Clonar y descargar los embeddings

Desde PowerShell:

```powershell
git clone https://github.com/SilGFerreyra/discursos-presidenciales.git
cd discursos-presidenciales
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri https://github.com/SilGFerreyra/discursos-presidenciales/releases/download/datos-v1/embeddings-chunks-bge-m3.npy -OutFile salida-embeddings-bge-m3\embeddings-chunks-bge-m3.npy
py -3.12 verificar_datos.py
```

Las **dos líneas** del verificador deben decir `OK`. La matriz de 359 MB se
distribuye en la [release `datos-v1`](https://github.com/SilGFerreyra/discursos-presidenciales/releases/tag/datos-v1);
los metadatos y el manifiesto ya están incluidos en el repositorio.
La configuración de progreso evita la lentitud de la barra de descarga en Windows
PowerShell 5.1; no modifica el archivo descargado.

### 2. Instalar las dependencias

El entorno se crea dentro de `app_reranker/.venv`:

```powershell
cd app_reranker
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

### 3. Descargar los modelos

Ejecute una vez, con internet y antes de abrir la app:

```powershell
.\.venv\Scripts\python.exe -B download_models.py
```

Descarga `BAAI/bge-m3` y `BAAI/bge-reranker-v2-m3` en revisiones fijas, comprueba
los tamaños y SHA256 de los pesos y los guarda en `app_reranker/.cache/huggingface/`.
El comando termina con `"status": "complete"`. Si se interrumpe, puede volver a
ejecutarlo: Hugging Face reutiliza los archivos completos y reanuda cuando es posible.

El aviso de que falta `hf_xet` no impide la descarga: se utiliza HTTP normal.
Ese paquete es una optimización opcional.

### 4. Iniciar la app

```powershell
.\ejecutar_app.bat
```

Abra la dirección que imprime el lanzador: `http://localhost:8502`, o el siguiente
puerto disponible. Mantenga abierta la consola y use `Ctrl+C` para detenerla.
La app carga los modelos al analizar una consulta o aplicar el reranker;
no los descarga por su cuenta.

## Uso

1. Complete cuatro frases que expresen el mismo concepto y pulse **Analizar cuatro carriers**.
2. Ajuste los filtros históricos y el número de resultados.
3. Si desea reordenar los candidatos, active el reranker y pulse **Aplicar reranker**.
4. Examine los pasajes y descargue los resultados en CSV, JSON o TXT.

Los valores iniciales son **K=10 resultados** y **N=25 candidatos**. El reranker
evalúa cuatro pares por candidato; aumentar N incrementa el trabajo de la CPU.

- [Guía para redactar carrier sentences](docs/carriers.md).
- [Corpus, modelos y metodología](docs/embeddings.md).
- [Estructura técnica y tests](app_reranker/README.md).

## Linux y macOS

Después de clonar el repositorio, descargue y compruebe la matriz:

```bash
curl -fL -o salida-embeddings-bge-m3/embeddings-chunks-bge-m3.npy https://github.com/SilGFerreyra/discursos-presidenciales/releases/download/datos-v1/embeddings-chunks-bge-m3.npy
python3.12 verificar_datos.py
cd app_reranker
python3.12 -m venv .venv
```

En **Linux**, instale primero PyTorch CPU:

```bash
.venv/bin/python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

En **macOS**, PyTorch se instala desde PyPI con el resto de las dependencias.
Luego, en ambos sistemas:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -B download_models.py
.venv/bin/python -B launch.py
```

La verificación de despliegue de este proyecto se realiza en Windows; los pasos
para otros sistemas requieren Python 3.12 y disponibilidad de los paquetes fijados.

## Solución de problemas

- **Falta la matriz o no coincide el SHA256:** vuelva a descargarla de la release
  y ejecute `verificar_datos.py`. Conserve el nombre y la carpeta indicados.
- **Faltan modelos locales:** cierre la app, ejecute `download_models.py` y espere
  a que termine correctamente antes de reiniciarla.
- **El entorno está incompleto:** desde `app_reranker`, ejecute
  `.\.venv\Scripts\python.exe -B launch.py --check` para comprobar el intérprete y
  las dependencias. El lanzador nunca instala paquetes automáticamente.
- **Memoria insuficiente o inferencia lenta:** reduzca N y el lote del reranker;
  mantenga los valores iniciales para la primera prueba.

Los modelos, entornos, archivos temporales y la matriz descargada están excluidos
de Git. No copie `.venv` de otra instalación: créelo con los comandos anteriores.
