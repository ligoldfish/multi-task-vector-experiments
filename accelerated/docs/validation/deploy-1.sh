bash <<'BASH'
set -eo pipefail
cd /home/work/user-job-dir/app
python - <<'PY'
import hashlib, json, stat, zipfile
from pathlib import Path, PurePosixPath
source = Path("task-vector-accelerated-20261013.zip")
checksum = Path("task-vector-accelerated-20261013.zip.sha256.txt")
target = Path("task-vector-accelerated-20261013")
expected = checksum.read_text(encoding="utf-8-sig").split()[0].lower()
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise SystemExit("ZIP SHA256 不匹配，停止")
if target.exists():
    raise SystemExit(f"目录已存在，保留并停止：{target}")
with zipfile.ZipFile(source) as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise SystemExit("ZIP 有重复路径，停止")
    for item in archive.infolist():
        relative = PurePosixPath(item.filename)
        if relative.is_absolute() or ".." in relative.parts or "\\" in item.filename or stat.S_ISLNK(item.external_attr >> 16):
            raise SystemExit(f"ZIP 路径不合规：{item.filename}")
    manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
    if set(names) != set(manifest["files"]) | {"PACKAGE_MANIFEST.json"}:
        raise SystemExit("ZIP 文件清单不匹配，停止")
    for name, digest in manifest["files"].items():
        if hashlib.sha256(archive.read(name)).hexdigest() != digest:
            raise SystemExit(f"文件哈希不匹配：{name}")
    archive.extractall(target)
print("ZIP_SHA256=PASS")
print("VERIFIED_FILES=" + str(len(manifest["files"])))
print("CODE_DIR=" + str(target.resolve()))
PY
BASH
