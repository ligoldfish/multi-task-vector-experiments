"""Independent MBPP execution regressions; no model or metrics dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from task_vector.code_execution import (
    EXECUTION_PROTOCOL_VERSION,
    execute_code,
    make_verified_bug,
)


@pytest.fixture
def mbpp_927():
    # Original Google MBPP fixture, not a corrected reference or modified test.
    # https://github.com/google-research/google-research/blob/master/mbpp/mbpp.jsonl
    return json.loads((Path(__file__).parent / "fixtures" / "mbpp_927.json").read_text(encoding="utf-8"))


def test_official_927_fixture_matches_failed_remote_input(mbpp_927):
    assert mbpp_927["task_id"] == 927
    assert hashlib.sha256(mbpp_927["code"].encode()).hexdigest() == "835756b17fcf98ab1628aab9e152235f508986f12be03b77ba97571e7969b946"
    tests = mbpp_927["test_list"] + mbpp_927["challenge_test_list"]
    assert hashlib.sha256(json.dumps(tests, ensure_ascii=False).encode()).hexdigest() == "63b4c49623661764af254223b5d20d39632b61c472189c1ef3eb456a962a0db3"


def test_official_927_candidate_defines_class_before_fixture(mbpp_927):
    result = execute_code(mbpp_927["code"], mbpp_927["test_list"], mbpp_927["test_setup_code"])
    assert result["pass_all"]
    assert result["passed"] == [True, True, True]
    assert result["load_error"] is None
    assert result["execution_protocol"] == EXECUTION_PROTOCOL_VERSION
    assert result["setup_sha256"] == hashlib.sha256(mbpp_927["test_setup_code"].encode()).hexdigest()


def test_candidate_and_setup_each_execute_once_in_declared_order():
    code = "candidate_calls = globals().get('candidate_calls', 0) + 1\nevents = globals().setdefault('events', [])\nevents.append('candidate')\nclass Node:\n    pass"
    setup = "setup_calls = globals().get('setup_calls', 0) + 1\nevents.append('setup')\nroot = Node()"
    result = execute_code(code, ["assert candidate_calls == setup_calls == 1", "assert events == ['candidate', 'setup']", "assert isinstance(root, Node)"], setup)
    assert result["pass_all"]


@pytest.mark.parametrize(
    ("code", "setup", "error"),
    [
        ("raise RuntimeError('candidate failed')", "raise RuntimeError('setup must not replace candidate failure')", "candidate failed"),
        ("class Node:\n    pass", "raise RuntimeError('setup failed')", "setup failed"),
        ("this is invalid Python !", "", "SyntaxError"),
    ],
)
def test_candidate_or_setup_failure_cannot_pass(code, setup, error):
    result = execute_code(code, ["assert True"], setup)
    assert not result["pass_all"]
    assert result["passed"] == []
    assert error in result["load_error"]


def test_official_927_verified_bug_keeps_reference_and_fault_evidence(mbpp_927):
    verified = make_verified_bug(mbpp_927["code"], mbpp_927["test_list"], mbpp_927["test_setup_code"])
    original, failure = verified["original_execution"], verified["bug_execution"]
    assert original["pass_all"]
    assert original["passed"] == [True, True, True]
    assert not failure["pass_all"]
    assert verified["buggy_code"] != mbpp_927["code"]
    assert original["execution_protocol"] == failure["execution_protocol"] == EXECUTION_PROTOCOL_VERSION
    assert original["setup_sha256"] == failure["setup_sha256"]
    assert original["tests_sha256"] == failure["tests_sha256"]
