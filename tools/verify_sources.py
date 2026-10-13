#!/usr/bin/env python3
"""Check frozen scientific source snapshots without torch or model/data assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root):
    root = Path(root)
    files = {}
    for folder in ("src", "configs", "deploy", "third_party", "ops"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
                files[path.relative_to(root).as_posix()] = file_hash(path)
    names = ["pyproject.toml", "docs/原始实验方案.md", "docs/补充协议与实现选择.md"]
    names += [path.name for path in sorted(root.glob("run_modelmate*.py"))]
    for name in names:
        if (root / name).is_file():
            files[name] = file_hash(root / name)
    return files


def verify(repo, selection="all"):
    repo = Path(repo).resolve()
    index = json.loads((repo / "provenance/source-snapshots.json").read_text(encoding="utf-8"))
    selected = list(index["snapshots"]) if selection == "all" else [selection]
    results = {}
    for name in selected:
        expected = index["snapshots"][name]
        source = repo / name
        actual = source_manifest(source)
        if actual != expected["sources"] or object_hash(actual) != expected["source_manifest_sha256"]:
            changed = sorted(k for k in actual.keys() | expected["sources"].keys() if actual.get(k) != expected["sources"].get(k))
            raise ValueError(f"Scientific snapshot differs: {name}: {changed}")
        package = {path.name: file_hash(path) for path in sorted((source / "src/task_vector").glob("*.py"))}
        if object_hash(package) != expected["training_package_sha256"]:
            raise ValueError(f"Training package digest differs: {name}")
        results[name] = {"verified": True, "source_files": len(actual),
                         "source_manifest_sha256": object_hash(actual),
                         "training_package_sha256": object_hash(package)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot", choices=("all", "baseline", "accelerated"), default="all")
    args = parser.parse_args()
    print(json.dumps(verify(args.repo, args.snapshot), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
