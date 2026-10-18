"""Launch only this app's interpreter, choosing a free port without stopping processes."""
import socket
import argparse
import json
import site
import subprocess
import sys
from pathlib import Path

from runtime import APP_ROOT, configure_runtime, get_venv_root


def validate_interpreter():
    root = get_venv_root()
    expected = root / "Scripts/python.exe"
    if (Path(sys.executable).resolve() != expected.resolve() or sys.version_info[:2] != (3, 12)
            or Path(sys.prefix).resolve() != root.resolve() or sys.prefix == sys.base_prefix
            or site.ENABLE_USER_SITE is not False):
        raise RuntimeError("El lanzador requiere el Python 3.12 del entorno propio configurado y aislado.")
    config = (root / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    if "include-system-site-packages = false" not in config:
        raise RuntimeError("El entorno propio no debe incluir paquetes del sistema.")
    for entry in sys.path:
        path = Path(entry).resolve()
        if "site-packages" in path.parts and not path.is_relative_to(root.resolve()):
            raise RuntimeError("sys.path contiene paquetes de otro entorno.")
    return root


def find_unused_port(start=8502, stop=8602):
    for port in range(start, stop):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No hay puertos disponibles en el rango 8502–8601.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check own interpreter/dependencies without starting a server")
    args = parser.parse_args()
    root = validate_interpreter()
    paths = configure_runtime()
    try:
        import streamlit
        import torch
        import transformers
        import matplotlib
        import seaborn
        import adjustText
    except (ImportError, OSError) as error:
        raise RuntimeError("El entorno nuevo está incompleto. No se modifica ni se instala automáticamente.") from error
    if not torch.__version__.endswith("+cpu"):
        raise RuntimeError("El entorno aprobado requiere Torch para CPU.")
    if args.check:
        print(json.dumps({"venv": str(root), "python": sys.executable, "prefix": sys.prefix,
                          "base_prefix": sys.base_prefix, "user_site_enabled": site.ENABLE_USER_SITE,
                          "versions": {"streamlit": streamlit.__version__, "torch": torch.__version__,
                                       "transformers": transformers.__version__},
                          "runtime_paths": {key: str(value) for key, value in paths.items()},
                          "network_policy": {"offline": True, "mode_set_before_hf_transformers_imports": True}}, indent=2))
        return 0
    port = find_unused_port()
    print(f"Versión con reranker local: http://localhost:{port}", flush=True)
    # There is a small bind/start race: Streamlit fails visibly if another process takes the port.
    return subprocess.call([sys.executable, "-B", "-m", "streamlit", "run", str(APP_ROOT / "streamlit_app.py"),
                            "--server.address", "127.0.0.1", "--server.port", str(port),
                            "--browser.gatherUsageStats", "false"], cwd=APP_ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
