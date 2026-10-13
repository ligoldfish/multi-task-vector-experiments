"""Create byte-preserving code snapshots; never copy training data or weights.

This is a publication utility, not an experiment runner. An existing snapshot
directory is rejected rather than overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def source_manifest(root):
    # Matches snapshots' src/task_vector/config.py exactly; tests and most docs
    # are intentionally outside the experiment's original source signature.
    root = Path(root)
    entries = {}
    for folder in ("src", "configs", "deploy", "third_party", "ops"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
                entries[path.relative_to(root).as_posix()] = file_sha256(path)
    names = ["pyproject.toml", "docs/原始实验方案.md", "docs/补充协议与实现选择.md"]
    names += [p.name for p in sorted(root.glob("run_modelmate*.py"))]
    for name in names:
        path = root / name
        if path.is_file():
            entries[name] = file_sha256(path)
    return entries


def copy_snapshot(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError(f"Existing snapshot is preserved: {target}")
    if not source.is_dir():
        raise ValueError(f"Missing source: {source}")
    folders = {"src", "configs", "deploy", "third_party", "ops", "tests", "docs"}
    skipped = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    files = []
    for item in sorted(source.iterdir()):
        if item.is_file() and (item.suffix in {".md", ".toml", ".py", ".sh", ".xml", ".json", ".txt"}
                               or item.name in {".gitignore", ".gitattributes"}):
            files.append(item)
        elif item.is_dir() and item.name in folders:
            files.extend(path for path in sorted(item.rglob("*")) if path.is_file()
                         and not skipped.intersection(path.relative_to(source).parts)
                         and not any(part.endswith(".egg-info") for part in path.parts))
    copied = {}
    total = 0
    for path in files:
        relative = path.relative_to(source)
        # Actual local credentials/settings are not publication examples.
        if relative.name in {"modelmate.local.yaml", "settings.local.json", ".env"}:
            raise ValueError(f"Local settings require explicit curation: {relative}")
        # Nested original attributes requested LF normalization, which would
        # silently change the very code bytes whose hashes prove provenance.
        # Keep their bytes as evidence and use a publication-only override.
        destination_relative = Path(".gitattributes.snapshot-original") if relative.name == ".gitattributes" else relative
        destination = target / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        digest = file_sha256(path)
        if digest != file_sha256(destination):
            raise ValueError(f"Copied bytes differ: {relative}")
        copied[destination_relative.as_posix()] = digest
        total += path.stat().st_size
    (target / ".gitattributes").write_bytes(b"* -text\n")
    copied[".gitattributes"] = file_sha256(target / ".gitattributes")
    total += len(b"* -text\n")
    scientific = source_manifest(source)
    if scientific != source_manifest(target):
        raise ValueError("Scientific source_manifest differs after copy")
    training = {path.name: file_sha256(path) for path in sorted((source / "src/task_vector").glob("*.py"))}
    return {"source_directory_name": source.name, "copied_file_count": len(copied),
            "copied_bytes": total, "copied_files": copied,
            "sources": scientific, "source_manifest_sha256": object_sha256(scientific),
            "training_package_sources": training, "training_package_sha256": object_sha256(training),
            "byte_preserving_copy_verified": True,
            "publication_only_change": {".gitattributes": "* -text; original bytes retained in .gitattributes.snapshot-original"},
            "actual_worker_source_equivalence": "not_verified_remote_local_settings_not_available",
            "training_package_keys": "basenames under src/task_vector, matching training._checkpoint_provenance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--accelerated", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    snapshots = {name: copy_snapshot(source, destination / name)
                 for name, source in (("baseline", args.baseline), ("accelerated", args.accelerated))}
    payload = {"schema_version": 1, "snapshots": snapshots,
               "hash_algorithm": "sha256", "object_hash_encoding": "UTF-8 JSON ensure_ascii=False sort_keys=True separators=(',', ':') default=str",
               "boundary": "Local code snapshots and local engineering validation; not a proof of remote training success or byte-identical worker configuration."}
    output = destination / "provenance/source-snapshots.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Preserve existing manifest: {output}")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in entry.items() if key not in {"copied_files", "sources", "training_package_sources"}}
                      for name, entry in snapshots.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
