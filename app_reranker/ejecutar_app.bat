@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "DISABLE_SAFETENSORS_CONVERSION=true"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo No existe app_reranker\.venv. Siga los pasos de instalacion del README.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0launch.py" %*
exit /b %errorlevel%
