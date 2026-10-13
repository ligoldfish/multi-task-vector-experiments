"""Official-source task adapters and deterministic, group-disjoint manifests.

No fallback examples are generated when an asset is missing. Length filtering
keeps complete targets, and the same materialized manifest serves both routes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import html
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable
import unicodedata
import xml.etree.ElementTree as ET

from .code_execution import EXECUTION_PROTOCOL_VERSION, make_verified_bug
from .prompts import canonical_task, encode_sample


@dataclass
class Sample:
    id: str
    task: str
    prompt: str
    target: str
    group_id: str
    official_split: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.task = canonical_task(self.task)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "Sample":
        return cls(**{key: row[key] for key in ("id", "task", "prompt", "target", "group_id", "official_split")}, metadata=dict(row.get("metadata", {})))


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _decode_tabular_fields(row: dict, path: Path, row_number: int) -> dict:
    """Nested CSV/TSV cells use JSON, never Python repr or character splitting."""
    output = dict(row)
    optional_empty = {"challenge_test_list", "triples", "rules"}
    for field_name in ("translation", "test_list", "challenge_test_list", "questions", "triples", "rules"):
        if field_name not in row:
            continue
        value = row[field_name]
        label = f"{path}:row {row_number}:field {field_name}"
        if isinstance(value, str):
            if not value.strip() and field_name in optional_empty:
                value = [] if field_name == "challenge_test_list" else {}
            else:
                try:
                    value = json.loads(value)
                except ValueError as exc:
                    raise ValueError(f"{label}: nested cell requires valid JSON") from exc
        if field_name in {"test_list", "challenge_test_list"}:
            valid = isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)
        elif field_name == "translation":
            valid = isinstance(value, dict) and all(isinstance(value.get(language), str) for language in ("en", "de"))
        elif field_name == "questions":
            valid = isinstance(value, (dict, list)) and all(isinstance(item, dict) for item in (value.values() if isinstance(value, dict) else value))
        else:
            valid = isinstance(value, dict) and all(isinstance(item, dict) and isinstance(item.get("text"), str) for item in value.values())
        if not valid:
            expected = "JSON array of nonempty test strings" if field_name in {"test_list", "challenge_test_list"} else "JSON object with en/de strings" if field_name == "translation" else "JSON object/list of query objects" if field_name == "questions" else "JSON object of objects with text strings"
            raise ValueError(f"{label}: expected {expected}")
        output[field_name] = value
    return output


def _read_file(path: Path) -> list[dict]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            return value["data"]
        raise ValueError(f"{path}: expected a JSON array or {{data: array}}")
    if path.suffix.lower() in {".tsv", ".csv"}:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            return [_decode_tabular_fields(row, path, number) for number, row in enumerate(reader, 2)]
    if path.suffix.lower() == ".parquet":
        from datasets import load_dataset
        return list(load_dataset("parquet", data_files=str(path), split="train"))
    raise ValueError(f"Unsupported raw file: {path}")


def _official_mbpp_split(task_id: int) -> str:
    # Official complete MBPP distribution ranges, not the sanitized configuration.
    if 1 <= task_id <= 10:
        return "prompt"
    if 11 <= task_id <= 510:
        return "test"
    if 511 <= task_id <= 600:
        return "validation"
    if 601 <= task_id <= 974:
        return "train"
    raise ValueError(f"MBPP complete task ID out of official range: {task_id}")


def load_official_source(task: str, source: str | dict) -> tuple[dict[str, list[dict]], dict]:
    """Load HF saved datasets, declared raw splits, or original official files."""
    task = canonical_task(task)
    source = {"path": source} if isinstance(source, str) else dict(source)
    datasets: dict[str, list[dict]] = defaultdict(list)
    provenance = {key: source.get(key) for key in ("dataset_id", "config", "revision", "path", "train_split", "eval_split")}
    file_evidence = []

    def add_file(split: str, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _read_file(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_evidence.append({"path": str(path.resolve()), "sha256": digest, "official_split": split, "rows": len(rows)})
        for row in rows:
            row = dict(row)
            row["_source_file"] = str(path)
            row["_source_sha256"] = digest
            datasets[split].append(row)

    if source.get("files"):
        for split, paths in source["files"].items():
            for path in ([paths] if isinstance(paths, str) else paths):
                add_file(split, Path(path))
    elif source.get("path"):
        path = Path(source["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing official {task} asset: {path}")
        if path.is_dir() and ((path / "dataset_dict.json").exists() or (path / "state.json").exists()):
            from datasets import DatasetDict, load_from_disk
            loaded = load_from_disk(str(path))
            if isinstance(loaded, DatasetDict):
                datasets.update({split: list(rows) for split, rows in loaded.items()})
                provenance["fingerprints"] = {split: rows._fingerprint for split, rows in loaded.items()}
            else:
                split = source.get("split")
                if not split:
                    raise ValueError("Single saved HF Dataset requires source.split (official source split)")
                datasets[split] = list(loaded)
                provenance["fingerprints"] = {split: loaded._fingerprint}
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    digest = hashlib.sha256()
                    with file.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    file_evidence.append({"path": str(file.resolve()), "sha256": digest.hexdigest(), "size": file.stat().st_size})
        elif path.is_file():
            rows = _read_file(path)
            file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in rows:
                row["_source_file"] = str(path)
                row["_source_sha256"] = file_digest
            if rows and all("official_split" in row or "split" in row for row in rows):
                for row in rows:
                    datasets[row.get("official_split", row.get("split"))].append(row)
            elif task in {"code_generation", "code_repair"} and rows and all("task_id" in row for row in rows):
                for row in rows:
                    datasets[_official_mbpp_split(int(row["task_id"]))].append(row)
                provenance["split_reconstruction"] = "complete_MBPP_official_task_id_ranges"
            elif source.get("split"):
                datasets[source["split"]] = rows
            else:
                raise ValueError(f"{path}: split is undeclared; refusing to mix official sources")
            file_evidence.append({"path": str(path.resolve()), "sha256": file_digest, "rows": len(rows)})
        elif task == "translation" and list(path.rglob("train.tags.*.en")):
            datasets.update(_load_iwslt_original(path))
            for file in path.rglob("*"):
                if file.is_file() and file.suffix in {".en", ".de", ".xml"}:
                    file_evidence.append({"path": str(file.resolve()), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        else:
            files = sorted(file for file in path.rglob("*") if file.suffix.lower() in {".jsonl", ".json", ".tsv", ".csv", ".parquet"})
            for file in files:
                if task == "logic":
                    if "cwa" in str(file).lower():
                        continue
                    depth_match = re.search(r"(?:depth[-_]?|[-_/]D)(\d+)", str(file), re.I)
                    if depth_match and int(depth_match.group(1)) > int(source.get("max_depth", 3)):
                        continue
                    if "meta-" not in file.name and any("meta-" in candidate.name for candidate in files):
                        continue  # No iterative proof-generation expansion.
                name = file.stem.lower()
                split = next((canonical for token, canonical in (("train", "train"), ("test", "test"), ("valid", "validation"), ("dev", "validation"), ("eval", "test")) if token in name), None)
                if split:
                    add_file(split, file)
            if not datasets:
                raise ValueError(f"No declared official split files found in {path}")
    elif source.get("allow_remote", False) and source.get("dataset_id"):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(source.get("revision", ""))):
            raise ValueError("Remote datasets require an explicit immutable revision")
        from datasets import load_dataset
        loaded = load_dataset(source["dataset_id"], source.get("config"), revision=source["revision"], trust_remote_code=bool(source.get("trust_remote_code", False)))
        datasets.update({split: list(rows) for split, rows in loaded.items()})
        provenance["fingerprints"] = {split: rows._fingerprint for split, rows in loaded.items()}
    else:
        raise ValueError(f"Provide local official {task} data at source.path/files, or explicitly enable remote loading")
    provenance["files"] = file_evidence
    provenance["source_counts"] = {split: len(rows) for split, rows in datasets.items()}
    return dict(datasets), provenance


def _load_iwslt_original(path: Path) -> dict[str, list[dict]]:
    """Read official TED train tags and paired XML test/dev segments."""
    output: dict[str, list[dict]] = defaultdict(list)
    train_en = sorted(path.rglob("train.tags.*.en"))
    if len(train_en) != 1:
        raise ValueError("Point IWSLT source.path at exactly one en-de corpus directory")
    en_path = train_en[0]
    de_path = en_path.with_suffix(".de")
    lines_en = [html.unescape(line.strip()) for line in en_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("<")]
    lines_de = [html.unescape(line.strip()) for line in de_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("<")]
    if len(lines_en) != len(lines_de):
        raise ValueError("IWSLT paired train line counts differ")
    output["train"] = [{"id": f"train-{index}", "translation": {"en": en, "de": de}} for index, (en, de) in enumerate(zip(lines_en, lines_de))]
    xml_en = sorted(file for file in path.rglob("*.xml") if file.name.endswith(".en.xml"))
    for en_file in xml_en:
        de_file = en_file.with_name(en_file.name.replace(".en.xml", ".de.xml"))
        if not de_file.exists():
            raise FileNotFoundError(de_file)
        en_root, de_root = ET.parse(en_file).getroot(), ET.parse(de_file).getroot()
        en_segments = ["".join(seg.itertext()).strip() for seg in en_root.iter("seg")]
        de_segments = ["".join(seg.itertext()).strip() for seg in de_root.iter("seg")]
        if len(en_segments) != len(de_segments):
            raise ValueError(f"IWSLT paired XML segment counts differ: {en_file}")
        split = "validation" if "dev" in en_file.name.lower() else "test"
        output[split].extend({"id": f"{en_file.stem}:{index}", "translation": {"en": en, "de": de}} for index, (en, de) in enumerate(zip(en_segments, de_segments)))
    return dict(output)


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while value != root:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, a: str, b: str):
        a, b = self.find(a), self.find(b)
        if a != b:
            low, high = sorted((a, b))
            self.parent[high] = low


def _proof_depth(row: dict, question: dict) -> tuple[int | None, int | None]:
    raw = question.get("QDep", question.get("depth", question.get("proof_depth")))
    depth = int(raw) if raw is not None else None
    maximum = row.get("maxD", row.get("source_depth"))
    if maximum is None:
        match = re.search(r"(?:depth[-_]?|[-_/]D)(\d+)", str(row.get("_source_file", "")) + "/" + str(row.get("id", "")), re.I)
        maximum = int(match.group(1)) if match else None
    return depth, int(maximum) if maximum is not None else None


def adapt_source(task: str, raw_splits: dict[str, list[dict]], source: dict | None = None) -> dict[str, list[Sample]]:
    task = canonical_task(task)
    source = source or {}
    adapted: dict[str, list[Sample]] = defaultdict(list)
    qqp = _UnionFind()
    if task == "paraphrase":
        for rows in raw_splits.values():
            for row in rows:
                if int(row.get("label", row.get("is_duplicate", -1))) == 1:
                    qqp.union(_normalize(row["question1"]), _normalize(row["question2"]))
    for split, rows in raw_splits.items():
        for index, row in enumerate(rows):
            identifier = str(row.get("id", row.get("idx", row.get("task_id", index))))
            metadata = {"source_index": index, "source_id": identifier}
            if row.get("_source_sha256"):
                metadata["source_file_sha256"] = row["_source_sha256"]
            if task == "math":
                prompt, target = str(row["question"]), str(row["answer"])
                if "####" not in target:
                    raise ValueError("GSM8K training/evaluation requires the official complete answer with #### final result")
                group = "gsm8k:" + _hash(_normalize(prompt))
                metadata["final_answer"] = target.rsplit("####", 1)[-1].strip()
            elif task == "logic":
                world = str(source.get("world", source.get("proofwriter_world", ""))).upper()
                if world != "OWA":
                    raise ValueError("ProofWriter source must explicitly specify world: OWA; CWA loses Unknown")
                theory = row.get("theory") or " ".join(str(item["text"]) for field_name in ("triples", "rules") for item in row.get(field_name, {}).values())
                if not theory:
                    raise ValueError("ProofWriter theory text missing")
                group = "proofwriter:" + _hash(_normalize(theory))
                questions = row.get("questions")
                if questions is None:
                    questions = {str(row.get("question_id", "Q0")): row}
                if isinstance(questions, list):
                    questions = {str(q.get("id", i)): q for i, q in enumerate(questions)}
                for qid, question in questions.items():
                    answer = question.get("answer", question.get("label"))
                    if isinstance(answer, bool):
                        answer = "True" if answer else "False"
                    target = str(answer).capitalize()
                    if target not in {"True", "False", "Unknown"}:
                        raise ValueError(f"Invalid or unlabeled OWA answer: {answer!r}")
                    depth, maximum = _proof_depth(row, question)
                    max_allowed = int(source.get("max_depth", 3))
                    if target == "Unknown":
                        # Unknown commonly has QDep=-1, which must not be discarded.
                        if maximum is None:
                            maximum = source.get("source_depth")
                        if maximum is None:
                            raise ValueError("Unknown needs a declared source_depth/maxD to establish the depth-0..3 corpus")
                        if int(maximum) > max_allowed:
                            continue
                    elif depth is None:
                        raise ValueError("ProofWriter labeled query has no proof depth")
                    elif depth < 0 or depth > max_allowed:
                        continue
                    query = question.get("question", question.get("text"))
                    if not isinstance(query, str):
                        raise ValueError("ProofWriter query text missing")
                    adapted[split].append(Sample(f"logic:{split}:{identifier}:{qid}", task,
                                                f"Facts and rules:\n{theory}\n\nQuery:\n{query}", target, group, split,
                                                {**metadata, "theory_id": identifier, "question_id": str(qid), "proof_depth": depth, "source_depth": maximum, "world": "OWA", "label": target}))
                continue
            elif task == "translation":
                value = row["translation"]
                value = json.loads(value) if isinstance(value, str) else value
                prompt, target = str(value["en"]), str(value["de"])
                group = "iwslt:" + _hash(_normalize(prompt))
                metadata.update({"source_language": "en", "target_language": "de"})
            elif task == "paraphrase":
                if int(row.get("label", row.get("is_duplicate", -1))) != 1:
                    continue
                prompt, target = str(row["question1"]), str(row["question2"])
                group = "qqp:" + _hash(qqp.find(_normalize(prompt)))
                metadata.update({"source_question": prompt, "qid1": row.get("qid1"), "qid2": row.get("qid2"), "label": 1})
            elif task in {"code_generation", "code_repair"}:
                prompt, target = str(row.get("text", row.get("prompt", ""))), str(row["code"])
                group = f"mbpp:{int(row['task_id'])}"
                test_list, challenge = row["test_list"], row.get("challenge_test_list", [])
                if not isinstance(test_list, list) or not isinstance(challenge, list) or not all(isinstance(test, str) and test.strip() for test in test_list + challenge):
                    raise ValueError(f"MBPP {identifier}: test_list/challenge_test_list must be arrays of nonempty strings; CSV/TSV cells require JSON arrays")
                tests = test_list + challenge
                if not tests or not prompt:
                    raise ValueError("MBPP requires text and official tests")
                metadata.update({"program_id": str(row["task_id"]), "test_list": tests, "test_setup_code": row.get("test_setup_code", ""), "requirements": prompt})
                # Both prompts are verified and length-filtered together before shared-ID assignment.
            elif task == "summary":
                prompt, target = str(row["dialogue"]), str(row["summary"])
                group = "samsum:" + _hash(_normalize(prompt))
            else:
                raise ValueError(task)
            if not prompt.strip() or not target.strip():
                continue
            adapted[split].append(Sample(f"{task}:{split}:{identifier}", task, prompt, target, group, split, metadata))
    return dict(adapted)


def _group_select(samples: list[Sample], counts: dict[str, int | str], seed: int) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    output: dict[str, list[Sample]] = {}
    position = 0
    for split, count in counts.items():
        if count == "all":
            output[split] = [sample for key in keys[position:] for sample in groups[key]]
            position = len(keys)
            continue
        selected = []
        while len(selected) < int(count) and position < len(keys):
            group = groups[keys[position]]
            position += 1
            needed = int(count) - len(selected)
            # Unused members are discarded; their group never enters a later split.
            selected.extend(sorted(group, key=lambda item: item.id)[:needed])
        if len(selected) != int(count):
            raise ValueError(f"Not enough group-disjoint eligible examples for {split}: requested {count}, available {len(selected)}")
        output[split] = selected
    return output


def audit_splits(splits: dict[str, list[Sample]]) -> dict:
    evidence = {"groups": {}, "ids": {}, "intersections": {}}
    for split, samples in splits.items():
        ids = [sample.id for sample in samples]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {split}")
        evidence["groups"][split] = len({sample.group_id for sample in samples})
        evidence["ids"][split] = len(samples)
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if "test_formal" in {a, b} and ({a, b} - {"test_formal"}) <= {"test", "test_confirmation"}:
                continue  # Formal expansion can contain both initial and reserved samples.
            overlap = {sample.group_id for sample in splits[a]} & {sample.group_id for sample in splits[b]}
            evidence["intersections"][f"{a}:{b}"] = len(overlap)
            if overlap:
                raise ValueError(f"Group leakage between {a}/{b}: {len(overlap)}")
    return evidence


def _write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_dataset(config: dict, tokenizer: Any, output_dir: str | Path) -> dict:
    settings = config.get("data", config)
    seed = int(settings.get("split_seed", 17))
    tasks = [canonical_task(task) for task in settings.get("tasks", ["math", "logic", "translation", "paraphrase"])]
    sources = {canonical_task(task): value for task, value in settings.get("sources", {}).items()}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema_version": 1, "status": "preparing", "split_seed": seed, "tasks": {}, "length_policy": "filter_complete_answer_never_truncate", "target_eos_supervised": True}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    code_assignment: dict[str, list[str]] | None = None
    for task in tasks:
        source = sources.get(task, sources.get("code_generation") if task == "code_repair" else None)
        if source is None:
            raise ValueError(f"Official source configuration missing for {task}")
        source_cfg = dict({"path": source} if isinstance(source, str) else source)
        if task == "logic":
            source_cfg.setdefault("world", settings.get("proofwriter_world", ""))
            source_cfg.setdefault("max_depth", settings.get("proofwriter_max_depth", 3))
        raw, provenance = load_official_source(task, source_cfg)
        adapted = adapt_source(task, raw, source_cfg)
        train_source = source_cfg.get("train_split", "train")
        eval_source = source_cfg.get("eval_split", "validation" if task == "paraphrase" else "test")
        if train_source == eval_source:
            raise ValueError("Final evaluation requires a distinct official labeled split")
        if train_source not in adapted or eval_source not in adapted:
            raise ValueError(f"Required official splits {train_source}/{eval_source} absent for {task}; found {list(adapted)}")
        rejected: list[dict] = []
        def eligible(samples: list[Sample]) -> list[Sample]:
            output = []
            seen = set()
            for sample in samples:
                identity = (sample.group_id, sample.prompt, sample.target)
                if identity in seen:
                    rejected.append({"id": sample.id, "reason": "duplicate_content"})
                    continue
                seen.add(identity)
                code_pair = None
                if task in {"code_generation", "code_repair"}:
                    cached_path = root / "verification" / f"mbpp-{sample.metadata['program_id']}.json"
                    fingerprint = _hash([EXECUTION_PROTOCOL_VERSION, sample.target, sample.metadata["test_list"], sample.metadata["test_setup_code"], config.get("code_execution", {})])
                    cached = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else None
                    if cached and cached.get("fingerprint") == fingerprint:
                        verified = cached.get("verification")
                        verification_error = cached.get("error")
                    else:
                        try:
                            verified = make_verified_bug(sample.target, sample.metadata["test_list"], sample.metadata["test_setup_code"], config)
                            verification_error = None
                        except (ValueError, SyntaxError) as exc:
                            verified, verification_error = None, str(exc)
                        cached_path.parent.mkdir(parents=True, exist_ok=True)
                        cached_path.write_text(json.dumps({"fingerprint": fingerprint, "verification": verified, "error": verification_error}, ensure_ascii=False, indent=2), encoding="utf-8")
                    if verified is None:
                        rejected.append({"id": sample.id, "group_id": sample.group_id, "reason": "code_verification_failed", "detail": verification_error})
                        continue
                    sample.metadata["verified_bug"] = verified
                    requirement = sample.metadata["requirements"]
                    repair_prompt = f"Task:\n{requirement}\n\nFaulty program:\n```python\n{verified['buggy_code']}\n```"
                    generation_sample = Sample(sample.id, "code_generation", requirement, sample.target, sample.group_id, sample.official_split, dict(sample.metadata))
                    repair_sample = Sample(sample.id, "code_repair", repair_prompt, sample.target, sample.group_id, sample.official_split, dict(sample.metadata))
                    code_pair = (generation_sample, repair_sample)
                    # The SAME program pool satisfies both tasks before the split.
                    if task == "code_repair":
                        sample.prompt = repair_prompt
                try:
                    max_prompt = int(config.get("training", {}).get("max_prompt_tokens", 512))
                    max_target = int(config.get("training", {}).get("max_target_tokens", 512))
                    variants = settings.get("eligibility_variants", ["canonical", "prompt_paraphrase", "output_format", "label_recode"])
                    encoded_variants = {variant: encode_sample(tokenizer, sample, variant=variant, max_prompt=max_prompt, max_target=max_target) for variant in variants}
                    if code_pair is not None:
                        paired_lengths = {}
                        for member in code_pair:
                            member_encoded = {variant: encode_sample(tokenizer, member, variant=variant, max_prompt=max_prompt, max_target=max_target) for variant in variants}
                            paired_lengths[member.task] = {variant: {"prompt": value["prompt_length"], "target_with_eos": value["target_length"]} for variant, value in member_encoded.items()}
                        sample.metadata["joint_program_eligibility"] = paired_lengths
                    encoded = encoded_variants.get("canonical") or encode_sample(tokenizer, sample, max_prompt=max_prompt, max_target=max_target)
                except ValueError as exc:
                    rejected.append({"id": sample.id, "group_id": sample.group_id, "reason": str(exc)})
                    continue
                sample.metadata["token_lengths"] = {"prompt": encoded["prompt_length"], "target_with_eos": encoded["target_length"]}
                sample.metadata["variant_token_lengths"] = {variant: {"prompt": value["prompt_length"], "target_with_eos": value["target_length"]} for variant, value in encoded_variants.items()}
                output.append(sample)
            return output
        train, evaluation = eligible(adapted[train_source]), eligible(adapted[eval_source])
        evaluation_groups = {sample.group_id for sample in evaluation}
        # Evaluation has priority when a group occurs in both official sources.
        removed = [sample for sample in train if sample.group_id in evaluation_groups]
        rejected.extend({"id": sample.id, "reason": "group_also_in_official_evaluation"} for sample in removed)
        train = [sample for sample in train if sample.group_id not in evaluation_groups]
        defaults = {"train": 300, "tune": 37, "discovery": 37, "test": 256} if task in {"code_generation", "code_repair"} else {"train": 512, "tune": 128, "discovery": 128, "test": 256}
        counts_setting = settings.get("counts", {})
        overrides = {key: value for key, value in counts_setting.items() if key in defaults}
        # MBPP retains its separately approved capacities instead of inheriting 512.
        if task in {"code_generation", "code_repair"}:
            overrides = {}
        overrides.update(counts_setting.get(task, counts_setting.get("default", {})))
        overrides.update(settings.get("counts_by_task", {}).get(task, {}))
        counts = {**defaults, **overrides}
        # Capacity failures must retain the IDs and reasons, rather than only raising.
        _write_jsonl(root / task / "rejected.jsonl", rejected)
        (root / task / "capacity_preflight.json").write_text(json.dumps({"source": provenance, "eligible_train": len(train), "eligible_evaluation": len(evaluation), "requested": counts, "cross_official_groups_removed": len(removed)}, ensure_ascii=False, indent=2), encoding="utf-8")
        train_splits = _group_select(train, {key: counts[key] for key in ("train", "tune", "discovery")}, seed)
        formal = settings.get("formal_test_size", 1000)
        formal_count = len(evaluation) if formal == "all" else min(int(formal), len(evaluation))
        if int(counts["test"]) > len(evaluation):
            raise ValueError(f"{task}: not enough independently labeled eligible official evaluation data")
        all_eval = _group_select(evaluation, {"test_formal": max(formal_count, int(counts["test"]))}, seed + 1)["test_formal"]
        splits = {**train_splits, "test": all_eval[:int(counts["test"])], "test_formal": all_eval}
        initial_test_groups = {sample.group_id for sample in splits["test"]}
        # A revision after unsealing the initial test can use this untouched,
        # whole-group-disjoint reserve. Formal itself contains the initial test.
        confirmation_pool = [sample for sample in evaluation if sample.group_id not in initial_test_groups]
        reserve_size = settings.get("confirmation_test_size", settings.get("formal_test_size", 1000))
        reserve_count = len(confirmation_pool) if reserve_size == "all" else min(int(reserve_size), len(confirmation_pool))
        splits["test_confirmation"] = _group_select(confirmation_pool, {"test_confirmation": reserve_count}, seed + 2)["test_confirmation"] if reserve_count else []
        if task in {"code_generation", "code_repair"}:
            assignment = {split: [sample.metadata["program_id"] for sample in rows] for split, rows in splits.items()}
            if code_assignment is None:
                code_assignment = assignment
            elif code_assignment != assignment:
                raise ValueError("Code generation/repair program assignments differ; refusing cross-task leakage")
        audit = audit_splits(splits)
        files = {}
        for split, samples in splits.items():
            path = root / task / f"{split}.jsonl"
            _write_jsonl(path, (sample.to_dict() for sample in samples))
            files[split] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": len(samples), "ids_sha256": _hash([sample.id for sample in samples])}
        _write_jsonl(root / task / "rejected.jsonl", rejected)
        manifest["tasks"][task] = {"source": provenance, "files": files, "audit": audit,
                                     "rejected_reasons": dict(Counter(row["reason"].split(":")[0] for row in rejected)),
                                     "formal_requested": formal, "formal_available": len(all_eval),
                                     "confirmation_reserve": {"count": reserve_count, "group_disjoint_from_initial_test": True, "may_overlap_formal_expansion": True, "use": "new_confirmation_only_if_never_previously_unsealed"},
                                     "label_distribution": {split: dict(Counter(sample.target for sample in rows)) for split, rows in splits.items()} if task == "logic" else None}
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest["status"] = "complete"
    manifest["manifest_content_sha256"] = _hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_samples(data_root: str | Path, task: str, split: str) -> list[Sample]:
    root, task = Path(data_root), canonical_task(task)
    path = root / task / f"{split}.jsonl"
    manifest_path = root / "manifest.json"
    expected = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("Prepared dataset is incomplete; inspect capacity_preflight.json/rejected.jsonl")
        expected = manifest["tasks"][task]["files"][split]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Prepared dataset hash mismatch for {task}/{split}")
    samples = [Sample.from_dict(row) for row in _read_file(path)]
    if expected and len(samples) != expected["count"]:
        raise ValueError(f"Prepared dataset count mismatch for {task}/{split}")
    return samples
