import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(r"D:\DataSpecificWork\task-vector")
DELIVERY = Path(r"C:\Users\hujin\.codex\visualizations\2026\10\10\01a1246d-d740-7b00-9bbb-fb8379c1ee94\modelmate-repaired")
HERE = Path(__file__).resolve().parent
NAMES = ("src/task_vector/smoke.py", "ops/recover_prefill_smoke.py",
         "tests/test_smoke_prefill.py", "tests/test_smoke_recovery.py",
         "docs/工程冒烟数值比较修订-20261011.md")
manifest = {"schema_version": 1, "smoke_protocol": "same_shape_causal_prefill_v2",
    "reason": "Same-shape causal checks separate BF16 shape drift from intervention correctness",
    "scientific_changes": [], "files": []}
output = DELIVERY / "task-vector-prefill-smoke-fix-20261011.zip"
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
