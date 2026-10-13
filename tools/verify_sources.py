"""Verify the frozen experiment source and protocol files using only stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def source_manifest(root):
    # Exact scope of src/task_vector/config.py's original source_manifest.
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


def verify(root, manifest_path=None):
    root = Path(root).resolve()
    path = Path(manifest_path).resolve() if manifest_path else root / "provenance/source-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("implementation") != "accelerated":
        raise ValueError("Unsupported single-implementation source manifest")
    expected = manifest["source_manifest"]
    if object_sha256(expected) != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest's own mapping digest differs")
    actual = source_manifest(root)
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(name for name in expected.keys() & actual.keys() if actual[name] != expected[name])
    if missing or added or changed:
        raise ValueError(json.dumps({"missing": missing, "added": added, "changed": changed}, ensure_ascii=False))
    training = {p.name: file_sha256(p) for p in sorted((root / "src/task_vector").glob("*.py"))}
    if (training != manifest["training_package_sources"] or
            object_sha256(training) != manifest["training_package_sha256"]):
        raise ValueError("Training Python package content differs")
    for name, expected_hash in manifest["protocols"].items():
        if actual.get(name) != expected_hash:
            raise ValueError("Protocol content differs: " + name)
    original = manifest.get("original_source_manifest")
    if original is not None:
        if object_sha256(original) != manifest.get("original_source_manifest_sha256"):
            raise ValueError("Original source mapping digest differs")
        for name, change in manifest.get("publication_changes", {}).items():
            if (original.get(name) != change.get("before_sha256") or
                    actual.get(name) != change.get("after_sha256")):
                raise ValueError("Recorded publication change differs: " + name)
    return {"status": "PASS", "implementation": "accelerated", "source_files": len(actual),
            "training_python_files": len(training),
            "source_manifest_sha256": object_sha256(actual),
            "training_package_sha256": object_sha256(training),
            "scope": "Frozen experiment src/configs/deploy/ops/third_party, entry points, pyproject, original plan and supplement; no model import or hardware access",
            "interpretation": "Content integrity of this published snapshot; compare exported training provenance separately."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root, args.manifest)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, "SOURCE_VERIFICATION=FAIL " + str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
