"""Portable source bundle with a complete SHA256 inventory; no models or secrets."""
from __future__ import annotations
from pathlib import Path
import zipfile
from .io import file_sha256, atomic_json


def create_package(source_root, output_path):
    source_root, output_path = Path(source_root).resolve(), Path(output_path).resolve()
    allowed = ("src", "configs", "deploy", "tests", "docs", "third_party", "ops")
    entries = []
    for folder in allowed:
        for path in sorted((source_root / folder).rglob("*")):
            if path.is_file() and not any(part in {"__pycache__", ".git", ".pytest_cache"} for part in path.relative_to(source_root).parts):
                entries.append(path)
    names = ["pyproject.toml", "README.md", "requirements-modelmate.txt", ".gitignore", ".gitattributes", "ModelMate_操作指南.md"]
    names += [p.name for p in sorted(source_root.glob("run_modelmate*.py"))]
    for name in names:
        path = source_root / name
        if path.exists():
            entries.append(path)
    hashes = {p.relative_to(source_root).as_posix(): file_sha256(p) for p in entries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in entries:
            archive.write(path, path.relative_to(source_root).as_posix())
        import json
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps({"schema_version": 1, "files": hashes}, ensure_ascii=False, indent=2))
    result = {"path": str(output_path), "sha256": file_sha256(output_path), "files": len(entries), "bytes": output_path.stat().st_size}
    atomic_json(output_path.with_suffix(".manifest.json"), {**result, "contents": hashes})
    return result
