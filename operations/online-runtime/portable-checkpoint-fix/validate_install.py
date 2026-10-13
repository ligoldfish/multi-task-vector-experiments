import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

DELIVERY = Path(r"C:\Users\hujin\.codex\visualizations\2026\10\10\01a1246d-d740-7b00-9bbb-fb8379c1ee94\modelmate-repaired")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
workspace = Path(r"D:\DataSpecificWork\task-vector") / (".portable-install-validation-" + stamp)
workspace.mkdir(exist_ok=False)
patch = DELIVERY / "task-vector-portable-checkpoint-fix-20261012.zip"
with zipfile.ZipFile(patch) as archive:
    manifest = json.loads(archive.read("PATCH_MANIFEST.json"))
with zipfile.ZipFile(DELIVERY / "task-vector-modelmate-repaired-20261010.zip") as original:
    def checkout(name):
        root = workspace / name
        for entry in manifest["files"] + [{"path": "src/task_vector/cli.py", "before_sha256": True}]:
            if entry["before_sha256"] is not None:
                path = root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original.read(entry["path"]))
        return root
    clean, conflict = checkout("clean"), checkout("conflict")

def run(root):
    return subprocess.run([sys.executable, str(patch), "--project-root", str(root),
                          "--evidence-root", str(root / "environment")],
                         capture_output=True, text=True, encoding="utf-8")

first = run(clean)
assert first.returncode == 0, first.stdout + first.stderr
assert "PATCH_STATUS=APPLIED" in first.stdout
for entry in manifest["files"]:
    assert hashlib.sha256((clean / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
backup, = (clean / "environment").glob("portable-source-fix-*")
for entry in manifest["files"]:
    if entry["before_sha256"] is not None:
        assert hashlib.sha256((backup / "before" / entry["path"]).read_bytes()).hexdigest() == entry["before_sha256"]
second = run(clean)
assert second.returncode == 0 and "PATCH_STATUS=ALREADY_APPLIED" in second.stdout, second.stdout + second.stderr
bad = conflict / "src/task_vector/io.py"
bad.write_bytes(bad.read_bytes() + b"\n# unexpected remote edit\n")
failed = run(conflict)
assert failed.returncode != 0 and "PREIMAGE_MISMATCH" in failed.stderr, failed.stdout + failed.stderr
assert not (conflict / "ops" / "recover_portable_checkpoint.py").exists()
assert not (conflict / "environment").exists()
print("PATCH_INSTALL_VALIDATION=PASS (hashes, backups, repeat, conflict before writes)")
print("VALIDATION_DIRECTORY=" + str(workspace))
