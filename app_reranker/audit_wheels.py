"""Inspect cached wheel metadata without extracting or installing packages."""
import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    cache = root / ".cache/tmp/pip-unpack-ao9grfb8"
    report = []
    for path in sorted(cache.glob("*.whl")):
        with zipfile.ZipFile(path) as wheel:
            metadata_name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
            wheel_name = next(n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL"))
            metadata = wheel.read(metadata_name).decode("utf-8")
            item = {"file": path.name, "bytes": path.stat().st_size,
                    "entries": len(wheel.infolist()),
                    "uncompressed_bytes": sum(i.file_size for i in wheel.infolist()),
                    "metadata": [line for line in metadata.splitlines() if line.startswith(("Name:", "Version:", "Requires-Python:", "Requires-Dist:"))],
                    "wheel": wheel.read(wheel_name).decode("utf-8")}
            report.append(item)
    if not any(item["file"] == "torch-2.14.1+cpu-cp312-cp312-win_amd64.whl" for item in report):
        raise RuntimeError("Required CPU wheel missing")
    (root / "wheel-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
