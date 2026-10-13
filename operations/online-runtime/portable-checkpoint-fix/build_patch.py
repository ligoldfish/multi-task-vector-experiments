import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(r"D:\DataSpecificWork\task-vector")
DELIVERY = Path(r"C:\Users\hujin\.codex\visualizations\2026\10\10\01a1246d-d740-7b00-9bbb-fb8379c1ee94\modelmate-repaired")
HERE = Path(__file__).resolve().parent
NAMES = ("src/task_vector/io.py", "ops/recover_portable_checkpoint.py",
         "tests/test_portable_checkpoint.py", "tests/test_portable_recovery.py",
         "docs/工程检查点兼容修订-20261012.md")
manifest = {"schema_version": 1, "reason": "Portable CPU checkpoint serialization; exact engineering calibration reuse",
            "scientific_changes": [], "source_only_install": True, "files": []}
output = DELIVERY / "task-vector-portable-checkpoint-fix-20261012.zip"
with zipfile.ZipFile(DELIVERY / "task-vector-modelmate-repaired-20261010.zip") as original:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        installer = (HERE / "__main__.py").read_bytes()
        ast.parse(installer)
        archive.writestr("__main__.py", installer)
        for name in NAMES:
            payload = (ROOT / name).read_bytes()
            if name.endswith(".py"):
                ast.parse(payload)
            before = hashlib.sha256(original.read(name)).hexdigest() if name in original.namelist() else None
            manifest["files"].append({"path": name, "before_sha256": before,
                                      "sha256": hashlib.sha256(payload).hexdigest()})
            archive.writestr("payload/" + name, payload)
        archive.writestr("PATCH_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
digest = hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix(output.suffix + ".sha256.txt").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
(HERE / "PATCH_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"path": str(output), "sha256": digest, "bytes": output.stat().st_size, "files": len(NAMES)}, indent=2))
