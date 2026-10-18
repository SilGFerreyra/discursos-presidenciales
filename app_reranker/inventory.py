"""Read-only protected-tree inventory; reports are confined to this app."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent


def _raise_walk_error(error):
    raise error


def snapshot(parent=PARENT, excluded_root=ROOT):
    """Match the original covered scope; reject links and traversal failures."""
    entries = {}
    for directory, dirs, files in os.walk(parent, onerror=_raise_walk_error, followlinks=False):
        dirs[:] = sorted(d for d in dirs if Path(directory, d) != excluded_root)
        for name in sorted(dirs + files):
            path = Path(directory, name)
            if path.is_symlink() or path.is_junction():
                raise RuntimeError(f"Protected-tree link is not traversed: {path}")
            stat = path.stat()
            key = str(path.relative_to(parent))
            item = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "kind": "dir" if path.is_dir() else "file"}
            # Dependency payloads and large corpus artifacts use metadata only.
            if path.is_file() and ".venv" not in path.parts and "__pycache__" not in path.parts:
                if path.suffix.lower() in {".py", ".bat", ".txt", ".toml", ".json", ".md", ".yaml", ".yml", ".cfg", ".ini"} or name == ".gitignore":
                    item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() and name == "pyvenv.cfg":
                item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                item["content"] = path.read_text(encoding="utf-8")
            entries[key] = item
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["before", "audit", "after", "hashes"])
    args = parser.parse_args()
    if args.phase == "hashes":
        baseline_path = ROOT / "inventory-before.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        checked, changes = 0, {}
        for relative, expected in baseline.items():
            if "sha256" not in expected:
                continue
            path = PARENT / relative
            if path.is_symlink() or path.is_junction():
                raise RuntimeError(f"Protected source link is not read: {path}")
            stat = path.stat()
            actual = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            checked += 1
            if any(actual[k] != expected[k] for k in actual):
                changes[relative] = actual
        report = {"checked_hashes": checked, "unchanged": not changes, "changes": changes,
                  "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  "scope": "Targeted source/config SHA256+size/mtime only; no full-tree traversal or all-byte proof"}
        (ROOT / "protected-hashes-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if changes:
            raise SystemExit(1)
        return
    destination = ROOT / f"inventory-{args.phase}.json"
    if args.phase == "before" and destination.exists():
        raise RuntimeError("Refusing to replace the before-work baseline")
    entries = snapshot()
    destination.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{args.phase}: {len(entries)} protected entries -> {destination}")
    if args.phase in {"after", "audit"}:
        baseline = json.loads((ROOT / "inventory-before.json").read_text(encoding="utf-8"))
        changes = {k: {"before": baseline.get(k), "after": entries.get(k)}
                   for k in sorted(set(baseline) | set(entries)) if baseline.get(k) != entries.get(k)}
        report = {"unchanged": not changes, "protected_entries": len(entries), "changes": changes,
                  "baseline_sha256": hashlib.sha256((ROOT / "inventory-before.json").read_bytes()).hexdigest(),
                  "hashed_entries": sum("sha256" in item for item in entries.values()),
                  "statement": "No changes detected by inventory" if not changes else "Changes detected",
                  "scope": "parent tree excluding app_reranker; file/dir size+mtime; selected source/config hashes; old venv metadata; not an all-byte proof"}
        report_name = "audit-invariance-report.json" if args.phase == "audit" else "invariance-report.json"
        (ROOT / report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"unchanged": not changes, "changed_entries": len(changes)}))
        if changes:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
