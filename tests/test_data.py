import csv
import json

import pytest

from task_vector.data import Sample, _group_select, adapt_source, audit_splits, load_official_source, load_samples, prepare_dataset
from task_vector.prompts import encode_sample, render_prompt, target_text, unwrap_prediction


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return list(range(3, 3 + len(text.split())))

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False and add_generation_prompt and not tokenize
        return "<system> " + messages[0]["content"] + " <user> " + messages[1]["content"] + " <assistant> <think> </think>"


def test_complete_answer_and_fixed_prompt_position():
    sample = Sample("x", "math", "Question?", "Reasoning with several tokens. #### 4", "g", "train")
    tokenizer = TinyTokenizer()
    encoded = encode_sample(tokenizer, sample)
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id
    assert all(value == -100 for value in encoded["labels"][:encoded["prompt_length"]])
    assert encoded["labels"][encoded["prompt_length"]:] == encoded["input_ids"][encoded["prompt_length"]:]
    with pytest.raises(ValueError, match="no truncation"):
        encode_sample(tokenizer, sample, max_target=2)


def test_owa_unknown_not_removed_by_negative_question_depth():
    row = {"id": "theory-D3-1", "maxD": 3, "theory": "Bob is red.", "questions": {
        "Q1": {"question": "Bob is red.", "answer": True, "QDep": 0},
        "Q2": {"question": "Bob is blue.", "answer": "Unknown", "QDep": -1},
        "Q3": {"question": "Bob is tall.", "answer": False, "QDep": 4},
    }}
    samples = adapt_source("logic", {"train": [row]}, {"world": "OWA"})["train"]
    assert [sample.target for sample in samples] == ["True", "Unknown"]
    assert len({sample.group_id for sample in samples}) == 1
    with pytest.raises(ValueError, match="OWA"):
        adapt_source("logic", {"train": [row]}, {"world": "CWA"})


def test_qqp_connected_groups_across_official_splits():
    raw = {"train": [{"id": 1, "question1": "Question A", "question2": "Question B", "label": 1}],
           "validation": [{"id": 2, "question1": " question b ", "question2": "Question C", "label": 1}]}
    adapted = adapt_source("qqp", raw)
    assert adapted["train"][0].group_id == adapted["validation"][0].group_id
    with pytest.raises(ValueError, match="leakage"):
        audit_splits({"train": adapted["train"], "test": adapted["validation"]})


def test_group_assignment_never_reuses_unused_members():
    samples = [Sample(f"{group}-{i}", "logic", "p", "True", group, "train") for group in ("a", "b", "c") for i in range(5)]
    selected = _group_select(samples, {"train": 3, "tune": 3, "discovery": 3}, 17)
    audit_splits(selected)
    assert all(len(rows) == 3 for rows in selected.values())


def test_mbpp_official_ranges_and_undeclared_other_sources(tmp_path):
    path = tmp_path / "mbpp.jsonl"
    path.write_text("\n".join(json.dumps({"task_id": value}) for value in (1, 11, 511, 601)), encoding="utf-8")
    source, evidence = load_official_source("mbpp", str(path))
    assert set(source) == {"prompt", "test", "validation", "train"}
    assert evidence["split_reconstruction"].startswith("complete_MBPP")
    with pytest.raises(ValueError, match="split is undeclared"):
        load_official_source("math", str(path))


def test_surface_controls_modify_only_protocol_and_parse_exactly():
    sample = Sample("l", "logic", "Facts\nQuery", "Unknown", "g", "test")
    assert target_text(sample, "label_recode") == "C"
    assert unwrap_prediction("C", "label_recode", "logic") == "Unknown"
    assert unwrap_prediction('{"answer":"Unknown"}', "output_format", "logic") == "Unknown"
    assert unwrap_prediction("Unknown", "output_format", "logic") == ""
    assert unwrap_prediction('{"answer":"Unknown","extra":"ignored"}', "output_format", "logic") == ""
    assert unwrap_prediction('{"answer":true}', "output_format", "logic") == ""
    assert unwrap_prediction('{"answer":"True","answer":"Unknown"}', "output_format", "logic") == ""
    assert "Facts\nQuery" in render_prompt(TinyTokenizer(), sample, "prompt_paraphrase")


def test_capacity_failure_persists_rejected_sample_evidence(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"id": "long", "official_split": "train", "question": " ".join(["x"] * 1000), "answer": "#### 1"},
                                  {"id": "eval", "official_split": "test", "question": "short", "answer": "#### 1"}]), encoding="utf-8")
    config = {"data": {"tasks": ["math"], "sources": {"math": {"path": str(source)}}, "counts": {"math": {"train": 1, "tune": 1, "discovery": 1, "test": 1}}}}
    with pytest.raises(ValueError, match="Not enough"):
        prepare_dataset(config, TinyTokenizer(), tmp_path / "prepared")
    evidence = (tmp_path / "prepared" / "math" / "rejected.jsonl").read_text(encoding="utf-8")
    assert "math:train:long" in evidence and "length_exceeded" in evidence


def test_code_tasks_jointly_verify_filter_then_share_program_assignments(tmp_path):
    path = tmp_path / "mbpp.jsonl"
    rows = [{"task_id": value, "text": f"Implement add_{value}(a,b) to add two numbers.", "code": f"def add_{value}(a,b):\n    return a+b", "test_list": [f"assert add_{value}(1,2)==3"]} for value in (601, 602, 603, 604, 605, 11, 12, 13)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    counts = {"train": 2, "tune": 1, "discovery": 1, "test": 2}
    config = {"data": {"tasks": ["code_generation", "code_repair"], "sources": {"code_generation": {"path": str(path)}}, "counts_by_task": {"code_generation": counts, "code_repair": counts}, "formal_test_size": "all"}}
    manifest = prepare_dataset(config, TinyTokenizer(), tmp_path / "prepared")
    assert manifest["status"] == "complete"
    for split in ("train", "tune", "discovery", "test"):
        generation = load_samples(tmp_path / "prepared", "code_generation", split)
        repair = load_samples(tmp_path / "prepared", "code_repair", split)
        assert [sample.group_id for sample in generation] == [sample.group_id for sample in repair]
        for sample in repair:
            verified = sample.metadata["verified_bug"]
            assert verified["original_execution"]["pass_all"] and not verified["bug_execution"]["pass_all"]
            assert "Faulty program:" in sample.prompt
            assert {"code_generation", "code_repair"} == set(sample.metadata["joint_program_eligibility"])
    initial = load_samples(tmp_path / "prepared", "code_generation", "test")
    reserve = load_samples(tmp_path / "prepared", "code_generation", "test_confirmation")
    assert {sample.group_id for sample in initial}.isdisjoint(sample.group_id for sample in reserve)
    train_path = tmp_path / "prepared" / "code_generation" / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_samples(tmp_path / "prepared", "code_generation", "train")


def test_old_setup_order_failure_cache_is_reverified(tmp_path, monkeypatch):
    from pathlib import Path
    import task_vector.data as data
    from task_vector.code_execution import EXECUTION_PROTOCOL_VERSION

    official = json.loads((Path(__file__).parent / "fixtures" / "mbpp_927.json").read_text(encoding="utf-8"))
    rows = [official] + [{"task_id": value, "text": f"Implement add_{value}(a,b).", "code": f"def add_{value}(a,b):\n    return a+b", "test_list": [f"assert add_{value}(1,2)==3"]} for value in (601, 602, 11)]
    source = tmp_path / "mbpp.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    counts = {"train": 1, "tune": 1, "discovery": 1, "test": 1}
    config = {"data": {"tasks": ["code_generation", "code_repair"], "sources": {"code_generation": {"path": str(source)}}, "counts_by_task": {task: counts for task in ("code_generation", "code_repair")}, "formal_test_size": "all"}}
    tests = official["test_list"] + official["challenge_test_list"]
    old_fingerprint = data._hash([official["code"], tests, official["test_setup_code"], {}])
    cache_path = tmp_path / "prepared" / "verification" / "mbpp-927.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"fingerprint": old_fingerprint, "verification": None, "error": "NameError: Node is not defined"}), encoding="utf-8")
    calls = []
    real_verify = data.make_verified_bug

    def tracked_verify(code, tests, setup, config):
        calls.append(code)
        return real_verify(code, tests, setup, config)

    monkeypatch.setattr(data, "make_verified_bug", tracked_verify)
    manifest = prepare_dataset(config, TinyTokenizer(), tmp_path / "prepared")
    assert manifest["status"] == "complete"
    assert calls.count(official["code"]) == 1
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["fingerprint"] == data._hash([EXECUTION_PROTOCOL_VERSION, official["code"], tests, official["test_setup_code"], {}])
    assert cached["error"] is None
    assert cached["verification"]["original_execution"]["passed"] == [True, True, True]
    for split in ("train", "tune", "discovery", "test"):
        generation = load_samples(tmp_path / "prepared", "code_generation", split)
        repair = load_samples(tmp_path / "prepared", "code_repair", split)
        assert [sample.group_id for sample in generation] == [sample.group_id for sample in repair]


@pytest.mark.parametrize("suffix,delimiter", [(".csv", ","), (".tsv", "\t")])
def test_tabular_mbpp_nested_tests_decode_without_character_splitting(tmp_path, suffix, delimiter):
    path = tmp_path / ("mbpp" + suffix)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "text", "code", "test_list", "challenge_test_list"], delimiter=delimiter)
        writer.writeheader()
        writer.writerow({"task_id": 601, "text": "Add numbers.", "code": "def add(a,b): return a+b", "test_list": json.dumps(["assert add(1,2)==3"]), "challenge_test_list": ""})
    raw, _ = load_official_source("code_generation", str(path))
    assert raw["train"][0]["test_list"] == ["assert add(1,2)==3"]
    adapted = adapt_source("code_generation", raw)["train"][0]
    assert adapted.metadata["test_list"] == ["assert add(1,2)==3"]


def test_tabular_proofwriter_and_translation_nested_fields(tmp_path):
    logic = tmp_path / "logic.csv"
    with logic.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "official_split", "maxD", "triples", "rules", "questions"])
        writer.writeheader()
        writer.writerow({"id": "theory1", "official_split": "train", "maxD": 3,
                         "triples": json.dumps({"t1": {"text": "Bob is red."}}), "rules": "{}",
                         "questions": json.dumps({"q1": {"question": "Bob is blue.", "answer": "Unknown", "QDep": -1}})})
    raw, _ = load_official_source("logic", str(logic))
    sample = adapt_source("logic", raw, {"world": "OWA"})["train"][0]
    assert sample.target == "Unknown" and "Bob is red." in sample.prompt
    translation = tmp_path / "translation.tsv"
    with translation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["official_split", "translation"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"official_split": "test", "translation": json.dumps({"en": "Cat", "de": "Katze"})})
    raw, _ = load_official_source("translation", str(translation))
    sample = adapt_source("translation", raw)["test"][0]
    assert (sample.prompt, sample.target) == ("Cat", "Katze")


@pytest.mark.parametrize("value", ["['assert True']", '"assert True"', '[42]'])
def test_tabular_nested_field_errors_identify_file_row_and_field(tmp_path, value):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "test_list"])
        writer.writeheader()
        writer.writerow({"task_id": 601, "test_list": value})
    with pytest.raises(ValueError, match=r"bad\.csv:row 2:field test_list"):
        load_official_source("code_generation", str(path))


def test_non_tabular_mbpp_requires_typed_test_arrays():
    with pytest.raises(ValueError, match="must be arrays"):
        adapt_source("code_generation", {"train": [{"task_id": 601, "text": "Do it.", "code": "pass", "test_list": "assert True"}]})
