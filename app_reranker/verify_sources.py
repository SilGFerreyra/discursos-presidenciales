"""Dependency-free syntax and source-boundary checks, not a runtime substitute."""
import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    files = []
    tests = {}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        compile(tree, str(path), "exec")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        if any(module == "app" or module.startswith("app.") for module in imports):
            raise RuntimeError(f"Original app import is forbidden: {path}")
        count = sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_") for node in ast.walk(tree))
        if count:
            tests[path.name] = count
        files.append(path.name)
    report = {"status": "syntax_and_import_boundary_checks_passed", "source_files": files,
              "source_file_count": len(files), "test_definitions": tests,
              "total_test_definitions": sum(tests.values()),
              "note": "AST/compile checks do not execute ML/UI assertions or prove runtime readiness"}
    (root / "source-check-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
