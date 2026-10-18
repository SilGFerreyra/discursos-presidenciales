"""Capture own runtime/dependency metadata without provisioning any model."""
import hashlib
import importlib.metadata as metadata
import json
import platform
import site
import sys

from runtime import APP_ROOT, LOCAL_ENV_ROOT, configure_runtime
from launch import validate_interpreter


def main():
    root = validate_interpreter()
    paths = configure_runtime()
    import numpy
    import pandas
    import torch
    import transformers
    import huggingface_hub
    import streamlit
    import matplotlib
    import seaborn
    import adjustText
    import pyarrow
    import psutil
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    distributions = sorted((d.metadata["Name"], d.version) for d in metadata.distributions())
    process = psutil.Process()
    memory = process.memory_info()
    report = {
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                   "user_site_enabled": site.ENABLE_USER_SITE, "sys_path": sys.path,
                   "bytecode_disabled": sys.dont_write_bytecode},
        "venv": str(root), "source_root": str(APP_ROOT),
        "runtime_paths": {key: str(value) for key, value in paths.items()},
        "installer_paths": {"temp": str(LOCAL_ENV_ROOT / ".cache/tmp"),
                            "pip_cache": str(LOCAL_ENV_ROOT / ".cache/pip"),
                            "wheels": str(LOCAL_ENV_ROOT / ".cache/wheels")},
        "versions": dict(distributions), "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(),
        "memory": {"process_rss_bytes_after_library_imports": memory.rss,
                   "process_peak_working_set_bytes": getattr(memory, "peak_wset", None),
                   "system_ram_bytes": psutil.virtual_memory().total,
                   "note": "Library import process only; NOT a loaded-model benchmark"},
        "canonical_helper": {"path": str(APP_ROOT.parent / "generar_embeddings_bge_m3_colab.py"),
                             "sha256": hashlib.sha256((APP_ROOT.parent / "generar_embeddings_bge_m3_colab.py").read_bytes()).hexdigest()},
        "preserved_partial_h_venv": {"path": str(APP_ROOT / ".venv"),
                                    "pyvenv_config_sha256": hashlib.sha256((APP_ROOT / ".venv/pyvenv.cfg").read_bytes()).hexdigest(),
                                    "note": "Not repaired, reused, copied or executed in this batch"},
        "models": {"provisioned_this_batch": False, "inference_benchmark_executed": False},
    }
    (APP_ROOT / "environment-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (APP_ROOT / "requirements-lock.txt").write_text(
        "# Resolved dedicated C: environment; install CPU Torch from the staged wheel first.\n"
        + "\n".join(f"{name}=={version}" for name, version in distributions) + "\n", encoding="utf-8")
    print(json.dumps({"venv": str(root), "packages": len(distributions), "torch": torch.__version__,
                      "transformers": transformers.__version__, "hub": huggingface_hub.__version__,
                      "streamlit": streamlit.__version__, "numpy": numpy.__version__,
                      "pandas": pandas.__version__, "cpu_threads": torch.get_num_threads(),
                      "rss_bytes": memory.rss}, indent=2))


if __name__ == "__main__":
    main()
