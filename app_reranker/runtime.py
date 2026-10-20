"""Configure app-owned caches before importing ML or plotting libraries."""
import os
import sys
import tempfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent


def get_venv_root():
    root = APP_ROOT / ".venv"
    if root.is_symlink() or root.is_junction():
        raise RuntimeError("El entorno propio no puede ser un enlace a otro entorno.")
    return root


def configure_network_mode(offline=True):
    """Set mode before HF/TF imports; never repair their import-cached globals.

    A process that already imported the libraries in another mode must restart.
    Only the explicitly invoked provisioning process requests online mode.
    """
    if not isinstance(offline, bool):
        raise ValueError("offline must be a boolean")
    for module_name, attribute in (("huggingface_hub.constants", "HF_HUB_OFFLINE"),
                                   ("transformers.utils.hub", "_is_offline_mode")):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, attribute):
            if bool(getattr(module, attribute)) != offline:
                raise RuntimeError("El modo de red fue fijado al importar las bibliotecas. Reinicie en un proceso nuevo; ejecute download_models.py por separado.")
    value = "1" if offline else "0"
    os.environ["HF_HUB_OFFLINE"] = value
    os.environ["TRANSFORMERS_OFFLINE"] = value


def configure_runtime(offline=True):
    """Explicit, idempotent bootstrap; fail rather than use another directory."""
    sys.dont_write_bytecode = True
    configure_network_mode(offline)
    paths = {
        "HF_HOME": APP_ROOT / ".cache" / "huggingface",
        "HF_HUB_CACHE": APP_ROOT / ".cache" / "huggingface" / "hub",
        "HF_ASSETS_CACHE": APP_ROOT / ".cache" / "huggingface" / "assets",
        "HF_MODULES_CACHE": APP_ROOT / ".cache" / "huggingface" / "modules",
        "XDG_CACHE_HOME": APP_ROOT / ".cache",
        "TORCH_HOME": APP_ROOT / ".cache" / "torch",
        "MPLCONFIGDIR": APP_ROOT / ".cache" / "matplotlib",
        "PIP_CACHE_DIR": APP_ROOT / ".cache" / "pip",
        "TEMP": APP_ROOT / ".cache" / "tmp",
        "TMP": APP_ROOT / ".cache" / "tmp",
    }
    for path in paths.values():
        if not path.resolve().is_relative_to(APP_ROOT.resolve()):
            raise RuntimeError(f"Cache path escapes app_reranker: {path}")
        for ancestor in (path, *path.parents):
            if ancestor == APP_ROOT:
                break
            if ancestor.is_symlink() or ancestor.is_junction():
                raise RuntimeError(f"Linked cache path is not allowed: {ancestor}")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for variable, path in paths.items():
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # Suppress Transformers' online background conversion, including first query downloads.
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ.pop("TRANSFORMERS_CACHE", None)
    tempfile.tempdir = str(paths["TEMP"])
    return paths
