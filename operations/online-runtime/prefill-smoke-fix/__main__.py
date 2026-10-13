"""Install the verified prefill-semantics engineering overlay with source backups."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    if not (root / "src" / "task_vector" / "cli.py").is_file():
        raise SystemExit("PROJECT_ROOT_MISMATCH")
    archive_path = Path(sys.argv[0]).resolve()
    prepared = []
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("PATCH_MANIFEST.json"))
        for entry in manifest["files"]:
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]:
                raise SystemExit("PATCH_PATH_INVALID")
            destination = root.joinpath(*relative.parts)
            if root not in destination.resolve().parents or destination.is_symlink():
                raise SystemExit(f"PATCH_PATH_ESCAPES_PROJECT: {relative}")
            payload = archive.read("payload/" + relative.as_posix())
            if digest(payload) != entry["sha256"]:
                raise SystemExit(f"PATCH_PAYLOAD_HASH_MISMATCH: {relative}")
            current = digest(destination.read_bytes()) if destination.exists() else None
            if current not in {entry["before_sha256"], entry["sha256"]}:
                raise SystemExit(f"PREIMAGE_MISMATCH: {relative}; actual={current}")
            prepared.append((destination, relative, payload, current, entry))
    # All preimages have passed before creating backups or changing source files.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = Path(args.evidence_root).resolve() / f"prefill-source-fix-{stamp}-{os.getpid()}"
    evidence.mkdir(parents=True, exist_ok=False)
    before = evidence / "before"
    changed = []
    for destination, relative, payload, current, entry in prepared:
        if current == entry["sha256"]:
            continue
        if current is not None:
            backup = before.joinpath(*relative.parts)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        changed.append(relative.as_posix())
    receipt = {"status": "installing", "archive": str(archive_path),
               "archive_sha256": digest(archive_path.read_bytes()),
               "project_root": str(root), "changed_files": changed, "manifest": manifest}
    receipt_path = evidence / "install.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for destination, relative, payload, current, entry in prepared:
        if current != entry["sha256"]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".prefill-fix-" + stamp)
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
        if digest(destination.read_bytes()) != entry["sha256"]:
            raise SystemExit(f"INSTALLED_HASH_MISMATCH: {relative}")
    receipt["status"] = "complete"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PATCH_SOURCE_BACKUP={evidence}")
    print("PATCH_STATUS=" + ("APPLIED" if changed else "ALREADY_APPLIED"))
    print("PATCH_INSTALLED=PASS")


if __name__ == "__main__":
    main()
