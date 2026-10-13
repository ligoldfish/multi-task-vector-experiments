"""Audit a vectors-only export and reproduce within-route geometric analysis on CPU.

No model, accelerator, original DPC path, or unsafe pickle fallback is needed.
Geometry imports the repository's frozen baseline implementation and works with
the represented sum(B @ A), never with flattened low-rank factor coordinates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

import torch


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def member_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid archive-relative file name: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Invalid archive-relative file name: {value!r}")
    return value


class Bundle:
    """Read without extracting ZIP members or following directory symlinks."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = None
        if self.path.is_dir():
            self.members = set()
            for child in self.path.rglob("*"):
                if child.is_symlink():
                    raise ValueError(f"Bundle contains a symlink: {child}")
                if child.is_file():
                    self.members.add(member_name(child.relative_to(self.path).as_posix()))
        else:
            self.archive = zipfile.ZipFile(self.path)
            names = []
            for item in self.archive.infolist():
                if item.is_dir():
                    member_name(item.filename.rstrip("/"))
                    continue
                names.append(member_name(item.filename))
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f"Bundle contains a symlink: {item.filename}")
            if len(names) != len(set(names)):
                raise ValueError("Bundle contains duplicate ZIP members")
            self.members = set(names)

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def open(self, name):
        member_name(name)
        if name not in self.members:
            raise ValueError(f"Required bundle member is missing: {name}")
        return self.archive.open(name) if self.archive else (self.path / name).open("rb")

    def json(self, name):
        with self.open(name) as handle:
            return json.load(handle)

    def sha256(self, name):
        digest = hashlib.sha256()
        with self.open(name) as handle:
            for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def tensor_payload(self, name):
        # ZipExtFile seek is expensive; only the selected representation is
        # buffered. Every other archive member was hashed in a bounded stream.
        with self.open(name) as handle:
            return torch.load(io.BytesIO(handle.read()), map_location="cpu", weights_only=True)


def verify_manifest(bundle):
    manifest = bundle.json("EXPORT_MANIFEST.json")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported EXPORT_MANIFEST schema")
    expected = manifest["files"]
    if set(expected) | {"EXPORT_MANIFEST.json"} != bundle.members or "EXPORT_MANIFEST.json" in expected:
        raise ValueError("Manifest does not describe exactly the bundle files")
    for name, sha in expected.items():
        member_name(name)
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"Invalid SHA256 for {name}")
        if bundle.sha256(name) != sha:
            raise ValueError(f"SHA256 mismatch: {name}")
    return {"status": "PASS", "verified_files": len(expected),
            "manifest_sha256": bundle.sha256("EXPORT_MANIFEST.json"),
            "interpretation": "Internal integrity and consistency; the bundle is not an independent signature of its author."}


def tensor_info(value, label, ndim):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or value.numel() == 0:
        raise ValueError(f"Invalid tensor shape: {label}; expected nonempty rank {ndim}")
    if value.dtype != torch.float32 or value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError(f"Expected an ordinary FP32 CPU tensor: {label}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite tensor values: {label}")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "numel": value.numel(), "finite": True, "minimum": float(value.min()),
            "maximum": float(value.max()), "l2_norm": float(torch.linalg.vector_norm(value.double())),
            "linf_norm": float(value.abs().max())}


def parameter_inner(terms_a, terms_b):
    total = 0.0
    for a in terms_a:
        for b in terms_b:
            ba, aa, bb, ab = (x.double() for x in (a["B"], a["A"], b["B"], b["A"]))
            total += float(((ba.T @ bb) * (aa @ ab.T)).sum())
    return total


def payload_digest(payload):
    keys = tuple(sorted(payload["values"] if payload["space"] == "activation" else payload["factors"]))
    digest = hashlib.sha256()
    digest.update(object_hash({"space": payload["space"], "keys": keys}).encode())
    for key in keys:
        values = [payload["values"][key]] if payload["space"] == "activation" else [
            value for term in payload["factors"][key] for value in (term["B"], term["A"])]
        for value in values:
            digest.update(str(tuple(value.shape)).encode())
            tensor = value.detach().float().cpu().contiguous()
            try:
                digest.update(tensor.numpy().tobytes())
            except RuntimeError as exc:
                if "numpy" not in str(exc).lower():
                    raise
                # NumPy is optional for inspection. Preserve the exact original
                # FP32 tensor-byte digest even in a torch-only installation.
                flat = tensor.reshape(-1).view(torch.uint8)
                for start in range(0, flat.numel(), 1024**2):
                    digest.update(bytes(flat[start:start + 1024**2].tolist()))
    return digest.hexdigest()


def inspect_payload(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported task-vector payload schema")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("Representation metadata must be a dictionary")
    values, factors = payload.get("values"), payload.get("factors")
    if not isinstance(values, dict) or not isinstance(factors, dict):
        raise ValueError("Expected values and factors dictionaries")
    space = payload.get("space")
    modules, coordinate_shapes, squared_norm = {}, {}, 0.0
    if space == "activation":
        if not values or factors:
            raise ValueError("Activation representation must contain values only")
        for key, value in sorted(values.items()):
            if not isinstance(key, str) or not key:
                raise ValueError("Invalid module name")
            modules[key] = tensor_info(value, key, 1)
            coordinate_shapes[key] = list(value.shape)
            squared_norm += float(value.double().square().sum())
    elif space == "parameter":
        if not factors or values:
            raise ValueError("Parameter representation must contain factors only")
        labels = payload["metadata"].get("branches", [])
        for key, terms in sorted(factors.items()):
            if not isinstance(key, str) or not key or not isinstance(terms, list) or not terms:
                raise ValueError("Invalid parameter module or empty factors")
            shape, inspected_terms = None, []
            for number, term in enumerate(terms):
                if not isinstance(term, dict) or set(term) != {"A", "B"}:
                    raise ValueError(f"Expected A and B in {key}/{number}")
                a_info = tensor_info(term["A"], f"{key}/{number}/A", 2)
                b_info = tensor_info(term["B"], f"{key}/{number}/B", 2)
                a, b = term["A"], term["B"]
                if a.shape[0] != b.shape[1]:
                    raise ValueError(f"Factor rank mismatch: {key}/{number}")
                current = [b.shape[0], a.shape[1]]
                if shape is not None and current != shape:
                    raise ValueError(f"Branch represented shapes differ: {key}")
                shape = current
                term_norm = math.sqrt(max(parameter_inner([term], [term]), 0.0))
                inspected_terms.append({"index": number, "branch": labels[number] if number < len(labels) else None,
                                        "stored_rank": a.shape[0], "A": a_info, "B": b_info,
                                        "represented_BA_frobenius_norm": term_norm})
            module_sq = parameter_inner(terms, terms)
            squared_norm += module_sq
            coordinate_shapes[key] = shape
            modules[key] = {"represented_shape": shape, "terms": inspected_terms,
                            "represented_sum_BA_frobenius_norm": math.sqrt(max(module_sq, 0.0)),
                            "cross_terms_included": True}
    else:
        raise ValueError(f"Unknown representation space: {space!r}")
    return {"space": space, "coordinate_shapes": coordinate_shapes, "module_count": len(modules),
            "modules": modules, "represented_norm": math.sqrt(max(squared_norm, 0.0)),
            "representation_digest": payload_digest(payload), "factor_flattening": False}


def baseline_source_check(source_root, origin):
    package = source_root / "src/task_vector"
    if not package.is_dir():
        return {"status": "NOT_CHECKED", "reason": "Baseline source directory is unavailable"}
    digest = object_hash({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(package.glob("*.py"))})
    expected = origin.get("provenance", {}).get("training_sources_sha256")
    return {"status": "MATCH" if expected == digest else "DIFFERENT" if expected else "UNAVAILABLE_IN_ORIGIN",
            "baseline_training_sources_sha256": digest, "origin_training_sources_sha256": expected,
            "scope": "All Python files in src/task_vector; not the original path-dependent full source/config manifest"}


def check_origin(item, selected, payload, inspection, origin, contract):
    if origin.get("status") != "complete" or origin.get("completion_schema") != 2:
        raise ValueError(f"Training origin is not a committed completed run: {item['training_origin']}")
    provenance = origin.get("provenance")
    if not isinstance(provenance, dict) or origin.get("provenance_sha256") != object_hash(provenance):
        raise ValueError("Training origin provenance content hash differs")
    if payload.get("task") != item.get("task") or payload.get("seed") != item.get("seed"):
        raise ValueError("Representation task/seed differs from VECTOR_INDEX")
    expected_space = {"ltv": "activation", "loda": "parameter"}.get(item.get("route"))
    if inspection["space"] != expected_space:
        raise ValueError("Representation space differs from its indexed route")
    metadata = payload["metadata"]
    if metadata.get("base_fingerprint") != contract["fingerprint"] or provenance.get("base_fingerprint") != contract["fingerprint"]:
        raise ValueError("Representation/origin base fingerprint differs from base contract")
    if metadata.get("model_id") != contract.get("model_id"):
        raise ValueError("Representation model ID differs from base contract")
    signature = origin.get("request_signature")
    if not signature or metadata.get("training_request_signature") != signature or provenance.get("request_signature") != signature:
        raise ValueError("Representation training request signature differs from origin")
    expected_scope = "independent" if item["route"] == "ltv" else item["scope"]
    if metadata.get("source_scope") != expected_scope:
        raise ValueError("Representation source scope differs from VECTOR_INDEX")
    if item["route"] == "ltv":
        actual_layers = metadata.get("layers")
        if not isinstance(actual_layers, list) or not actual_layers:
            raise ValueError("LTV metadata is missing layer coordinates")
        if item.get("layers") is not None and actual_layers != item["layers"]:
            raise ValueError("LTV layers differ from VECTOR_INDEX")
        if sorted(inspection["coordinate_shapes"]) != sorted(f"model.layers.{layer}" for layer in actual_layers):
            raise ValueError("LTV layer names differ from their metadata")
    architecture = contract.get("expected_config", {})
    width, layer_count = architecture.get("hidden_size"), architecture.get("num_hidden_layers")
    for key, shape in inspection["coordinate_shapes"].items():
        layer = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", key)
        if layer_count is not None and (layer is None or int(layer.group(1)) >= layer_count):
            raise ValueError(f"Module layer lies outside base architecture: {key}")
        if width is not None and shape[-1] != width:
            raise ValueError(f"Coordinate input width differs from base architecture: {key}")
        if item["route"] == "loda":
            kind = key.rsplit(".", 1)[-1]
            if kind not in {"q_proj", "k_proj", "v_proj"}:
                raise ValueError(f"LoDA module is not a Q/K/V projection: {key}")
            heads = architecture.get("num_attention_heads" if kind == "q_proj" else "num_key_value_heads")
            if heads is not None and architecture.get("head_dim") is not None and shape[0] != heads * architecture["head_dim"]:
                raise ValueError(f"Coordinate output width differs from base architecture: {key}")
    if selected != "initial" and metadata.get("version") != selected:
        raise ValueError("Selected export version differs from representation metadata")
    for version in ("raw", "post", "initial"):
        field = version + "_path"
        indexed = item["exports"].get(field)
        recorded = origin.get("artifacts", {}).get(field)
        if not isinstance(indexed, dict) or not isinstance(recorded, dict):
            raise ValueError(f"Missing committed artifact entry: {field}")
        if indexed.get("sha256") != recorded.get("sha256") or indexed.get("representation_digest") != recorded.get("representation_digest"):
            raise ValueError(f"Export index differs from committed origin: {field}")
    if inspection["representation_digest"] != item["exports"][selected + "_path"]["representation_digest"]:
        raise ValueError("Actual representation content digest differs from committed origin")


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(part) for part in value]
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(clean_json(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def geometry_functions(source_root):
    package = source_root / "src/task_vector"
    if not (package / "analysis.py").is_file():
        raise ValueError(f"Baseline geometry source is unavailable: {package}")
    sys.path.insert(0, str(source_root / "src"))
    representation = importlib.import_module("task_vector.representations")
    analysis = importlib.import_module("task_vector.analysis")
    if Path(analysis.__file__).resolve() != (package / "analysis.py").resolve():
        raise ValueError("A different task_vector package is already imported; run this tool in a fresh Python process")
    return representation, analysis


def analyze(bundle_path, output, *, source_root=None, version="post", geometry=True, save_decomposition=False):
    if version not in {"post", "raw", "initial"}:
        raise ValueError("Version must be post, raw, or initial")
    if save_decomposition and not geometry:
        raise ValueError("--save-decomposition requires geometry")
    source_root = Path(source_root or Path(__file__).resolve().parents[1] / "baseline").resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output must be a new directory: {output}")
    bundle = Bundle(bundle_path)
    try:
        verified = verify_manifest(bundle)
        index = bundle.json("VECTOR_INDEX.json")
        if index.get("schema_version") != 1 or not isinstance(index.get("representations"), list) or not index["representations"]:
            raise ValueError("Unsupported or empty VECTOR_INDEX schema")
        contract = bundle.json(member_name(index["base_contract"]))
        if not contract.get("files") or object_hash(contract["files"]) != contract.get("fingerprint"):
            raise ValueError("Invalid base model contract fingerprint")
        outputs, groups, seen = [], defaultdict(list), set()
        manifest = bundle.json("EXPORT_MANIFEST.json")["files"]
        for number, item in enumerate(index["representations"]):
            for field in ("raw_path", "post_path", "initial_path"):
                exported = item.get("exports", {}).get(field, {})
                path = member_name(exported.get("path"))
                if path in seen:
                    raise ValueError(f"Export file occurs more than once: {path}")
                seen.add(path)
                if manifest.get(path) != exported.get("sha256"):
                    raise ValueError(f"Vector index SHA256 differs from manifest: {path}")
            if item.get("primary") != item["exports"]["post_path"]["path"]:
                raise ValueError("Primary export must be the indexed post version")
            path = item["exports"][version + "_path"]["path"]
            payload = bundle.tensor_payload(path)
            inspection = inspect_payload(payload)
            origin = bundle.json(member_name(item["training_origin"]))
            check_origin(item, version, payload, inspection, origin, contract)
            group = {"route": item["route"], "scope": "independent" if item["route"] == "ltv" else item["scope"],
                     "seed": item["seed"], "layers": payload["metadata"].get("layers", item.get("layers")),
                     "space": inspection["space"], "coordinate_shapes": inspection["coordinate_shapes"],
                     "base_fingerprint": contract["fingerprint"]}
            record = {"number": number, "path": path, "route": item["route"], "scope": item["scope"],
                      "task": item["task"], "seed": item["seed"], "selected_version": version,
                      "is_untrained_control": version == "initial", "inspection": inspection,
                      "representation_metadata": payload["metadata"], "training_origin": origin,
                      "training_source_comparison": baseline_source_check(source_root, origin),
                      "provenance_consistency": "PASS", "geometry_group": group}
            outputs.append(record)
            groups[object_hash(group)].append((number, payload))
        # Validation is complete before creating output; input is never changed.
        output.mkdir(parents=True, exist_ok=False)
        geometry_reports = []
        if geometry:
            rep_module, analysis_module = geometry_functions(source_root)
            for group_hash, members in groups.items():
                description = outputs[members[0][0]]["geometry_group"]
                representations = [rep_module.Representation(
                    p["space"], p["task"], p["seed"], p["values"],
                    {key: [rep_module.LowRankFactor(term["B"], term["A"]) for term in terms]
                     for key, terms in p["factors"].items()}, p["metadata"]) for _, p in members]
                tasks = [rep.task for rep in representations]
                if len(tasks) != len(set(tasks)):
                    raise ValueError("A geometry group contains duplicate source tasks")
                report = {"group_id": group_hash, "group": description, "tasks": tasks,
                          "represented_gram": analysis_module.gram_matrix(representations).tolist(),
                          "represented_cosines": analysis_module.cosine_matrix(representations).tolist(),
                          "undefined_cosines": "null when at least one represented norm is zero",
                          "factor_flattening": False, "dense_parameter_expansion": False,
                          "decompositions": {}}
                for label, method, q in (("mean", "mean", 0), ("svd-q0", "svd", 0), ("svd-q1", "svd", 1), ("svd-q2", "svd", 2)):
                    if method == "svd" and q > len(representations):
                        report["decompositions"][label] = {"status": "SKIPPED", "reason": "Requested rank exceeds source task count"}
                        continue
                    decomposed = analysis_module.decompose(representations, method, q)
                    entry = {"status": "COMPLETE", "weights": decomposed.weights.tolist(),
                             "metadata": decomposed.metadata,
                             "shared_norms": [rep_module.norm(rep) for rep in decomposed.shared],
                             "residual_norms": [rep_module.norm(rep) for rep in decomposed.residual]}
                    if save_decomposition:
                        destination = output / "decompositions" / group_hash / label
                        destination.mkdir(parents=True, exist_ok=False)
                        entry["saved"] = []
                        for n, task in enumerate(tasks):
                            for kind in ("shared", "residual"):
                                relative = destination.relative_to(output) / f"{n:02d}-{kind}.pt"
                                rep = getattr(decomposed, kind)[n]
                                rep.metadata.update(analysis_only=True, source_export_version=version,
                                                    source_bundle_manifest_sha256=verified["manifest_sha256"],
                                                    source_task=task, decomposition_method=method,
                                                    decomposition_requested_q=q if method == "svd" else None)
                                with (output / relative).open("xb") as handle:
                                    torch.save(rep.payload(), handle)
                                entry["saved"].append({"source_task": task, "kind": kind, "path": relative.as_posix()})
                    report["decompositions"][label] = entry
                geometry_reports.append(report)
        final = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "bundle_integrity": verified, "base_contract": contract, "version": version,
                 "representation_count": len(outputs), "representations": outputs,
                 "geometry": geometry_reports, "geometry_enabled": geometry,
                 "limitations": ["Hash/provenance consistency does not independently prove which training actually ran.",
                                  "Geometry and reconstruction do not establish functional usefulness or transfer; rerun controlled evaluation on the target base.",
                                  "Vectors-only has no datasets or optimizer/RNG checkpoints; it supports reuse and inspection, not exact optimizer resume.",
                                  "Source absolute paths in training_origin are historical evidence and are never accessed."]}
        write_json(output / "analysis.json", final)
        return final
    finally:
        bundle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="vectors-only ZIP or its extracted directory")
    parser.add_argument("--output", required=True, help="new, nonexisting output directory")
    parser.add_argument("--source-root", default=None, help="frozen baseline source root; defaults to this repository's baseline/")
    parser.add_argument("--version", choices=("post", "raw", "initial"), default="post")
    parser.add_argument("--skip-geometry", action="store_true", help="inspect and audit with PyTorch only")
    parser.add_argument("--save-decomposition", action="store_true", help="also save shared/residual plain tensor dictionaries")
    args = parser.parse_args(argv)
    result = analyze(args.bundle, args.output, source_root=args.source_root, version=args.version,
                     geometry=not args.skip_geometry, save_decomposition=args.save_decomposition)
    print(json.dumps({"status": "PASS", "representations": result["representation_count"],
                      "geometry_groups": len(result["geometry"]),
                      "analysis": str(Path(args.output).resolve() / "analysis.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
