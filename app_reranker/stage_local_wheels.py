"""Stage read-only cached wheels into the explicitly authorized local install cache."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_ROOT = Path(r"C:\Users\Usuario\AppData\Local\app-reranker")


def main():
    if not LOCAL_ROOT.is_dir() or LOCAL_ROOT.is_symlink() or LOCAL_ROOT.is_junction():
        raise RuntimeError("Dedicated local root missing or linked")
    source = ROOT / ".cache/tmp/pip-unpack-ao9grfb8"
    destination = LOCAL_ROOT / ".cache/wheels"
    destination.mkdir(parents=True, exist_ok=False)
    (LOCAL_ROOT / ".cache/tmp").mkdir(parents=True, exist_ok=False)
    records = []
    wheels = sorted(source.glob("*.whl"))
    if len(wheels) != 10:
        raise RuntimeError("Expected the ten audited CPU Torch closure wheels")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"Corrupt cached wheel: {wheel.name}")
        target = destination / wheel.name
        shutil.copyfile(wheel, target)
        source_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
        target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if source_hash != target_hash:
            raise RuntimeError(f"Staged wheel mismatch: {wheel.name}")
        records.append({"file": wheel.name, "source": str(wheel), "destination": str(target),
                        "size": wheel.stat().st_size, "sha256": source_hash})
    (ROOT / "local-wheel-staging.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Staged and ZIP/SHA256 validated {len(records)} wheels to {destination}")


if __name__ == "__main__":
    main()
