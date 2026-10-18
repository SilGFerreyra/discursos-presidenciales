@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "DISABLE_SAFETENSORS_CONVERSION=true"
if not defined APP_RERANKER_VENV (
  if exist "%~dp0venv-path.txt" (
    set /p "APP_RERANKER_VENV="<"%~dp0venv-path.txt"
  ) else (
    set "APP_RERANKER_VENV=%~dp0.venv"
  )
)
for %%I in ("%APP_RERANKER_VENV%") do set "APP_RERANKER_VENV=%%~fI"
set "OWN_ENV_ALLOWED="
if /I "%APP_RERANKER_VENV%"=="C:\Users\Usuario\AppData\Local\app-reranker\.venv" set "OWN_ENV_ALLOWED=1"
if /I "%APP_RERANKER_VENV%"=="%~dp0.venv" set "OWN_ENV_ALLOWED=1"
if not defined OWN_ENV_ALLOWED (
  echo La ruta del entorno no es una ubicacion propia autorizada de app_reranker.
  exit /b 1
)
if not exist "%APP_RERANKER_VENV%\Scripts\python.exe" (
  echo No existe el Python propio de app_reranker.
  exit /b 1
)
"%APP_RERANKER_VENV%\Scripts\python.exe" -B "%~dp0launch.py" %*
exit /b %errorlevel%
