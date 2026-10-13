"""Check tracked publication files; no model, data or runtime dependency needed."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def check(root):
    root = Path(root).resolve()
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    paths = [name for name in paths if name]
    if not paths:
        raise ValueError("No tracked publication files")
    credentials = [re.compile(pattern) for pattern in (
        r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
        r"hf_[A-Za-z0-9]{30,}", r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        r"https?://[^\s/@:]+:[^\s/@]+@",
    )]
    forbidden = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".zip", ".pem", ".key"}
    parsed = 0
    total_bytes = 0
    for name in paths:
        path = root / name
        if path.is_symlink() or path.suffix in forbidden or path.stat().st_size > 5 * 1024**2:
            raise ValueError(f"Unexpected artifact or oversized file: {name}")
        raw = path.read_bytes()
        total_bytes += len(raw)
        content = raw.decode("utf-8-sig")
        if any(pattern.search(content) for pattern in credentials):
            raise ValueError(f"Potential credential in tracked file: {name}")
        if path.suffix == ".py":
            ast.parse(content, filename=name)
            parsed += 1
    snapshots = json.loads((root / "provenance/source-snapshots.json").read_text(encoding="utf-8"))
    verified_files = {}
    for version, info in snapshots["snapshots"].items():
        for name, expected in info["copied_files"].items():
            actual = hashlib.sha256((root / version / name).read_bytes()).hexdigest()
            if actual != expected or f"{version}/{name}" not in paths:
                raise ValueError(f"Snapshot copied-file mismatch: {version}/{name}")
        verified_files[version] = len(info["copied_files"])
    return {"status": "PASS", "tracked_files": len(paths), "tracked_bytes": total_bytes,
            "python_files_parsed": parsed, "snapshot_files_verified": verified_files,
            "credential_scan_matches": 0, "weights_or_archives_tracked": 0}


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
