"""Save pinned reference source files from the two required official repositories.

These are auditable selected-file snapshots, never presented as full clones and
never imported into the experiment implementation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


SOURCES = {
    "Learned_TV": {"repo": "HLYang2001/Learned_TV", "files": ["README.md", "LICENSE", "environment.yml", "experiments/train_ltv.py", "utils/hidden_states.py"]},
    "LoDA_ICML2026": {"repo": "HHHLF/LoDA_ICML2026", "files": ["README.md", "LICENSE", "models/decomposed_lora.py", "methods/dlora.py"]},
}


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "task-vector-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    root = Path(__file__).resolve().parents[1] / "third_party"
    root.mkdir(exist_ok=True)
    report = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "kind": "selected_official_source_snapshots", "repositories": {}}
    for name, spec in SOURCES.items():
        url = "https://github.com/" + spec["repo"]
        record = {"repository_url": url, "full_clone": False, "files": [], "failures": []}
        try:
            page = get(url).decode("utf-8")
            commit_match = re.search(r'"currentOid":"([0-9a-f]{40})"', page)
            if not commit_match:
                raise ValueError("GitHub page did not expose currentOid; refusing an unpinned snapshot")
            commit = commit_match.group(1)
            record["commit"] = commit
            for filename in spec["files"]:
                source_url = f"https://raw.githubusercontent.com/{spec['repo']}/{commit}/{filename}"
                try:
                    data = get(source_url)
                    destination = root / name / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    record["files"].append({"path": f"{name}/{filename}", "source_url": source_url,
                                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                except (OSError, ValueError) as exc:
                    record["failures"].append({"path": filename, "source_url": source_url,
                                               "error_type": type(exc).__name__, "error": str(exc)})
        except (OSError, ValueError) as exc:
            record["failures"].append({"path": "repository_commit", "error_type": type(exc).__name__, "error": str(exc)})
        record["status"] = "complete_selected_snapshot" if not record["failures"] else "partial_selected_snapshot" if record["files"] else "failed"
        report["repositories"][name] = record
        print(name, record["status"], record.get("commit"), len(record["files"]), flush=True)
    (root / "official_sources.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
