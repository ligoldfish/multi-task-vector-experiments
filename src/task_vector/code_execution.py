"""Actual MBPP test execution and verified deterministic bug construction.

The subprocess backend supplies resource limits, NOT a security boundary.
Use the Docker backend or a dedicated restricted worker for untrusted programs.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


# The fixture may instantiate classes defined by the submitted program. Keep
# this version in prepared-data cache identities when execution semantics change.
EXECUTION_PROTOCOL_VERSION = "candidate_then_setup_then_tests_v1"


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    return "\n\n".join(blocks).strip() if blocks else text.strip()


_WORKER = r'''
import contextlib, json, os, sys, traceback
payload = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    import resource
    limits = payload["limits"]
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_mb"] * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
except ImportError:
    pass
namespace = {"__name__": "__main__"}
result = {"passed": [], "errors": [], "load_error": None}
with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    try:
        exec(compile(payload["code"], "<candidate>", "exec"), namespace)
        exec(compile(payload.get("setup", ""), "<setup>", "exec"), namespace)
    except BaseException as exc:
        result["load_error"] = type(exc).__name__ + ": " + str(exc)[:2000]
    if result["load_error"] is None:
        for test in payload["tests"]:
            try:
                exec(compile(test, "<test>", "exec"), namespace)
                result["passed"].append(True)
                result["errors"].append(None)
            except BaseException as exc:
                result["passed"].append(False)
                result["errors"].append(type(exc).__name__ + ": " + str(exc)[:2000])
print(json.dumps(result))
'''


def execute_code(code: str, tests: list[str], setup: str = "", config: dict | None = None) -> dict[str, Any]:
    settings = dict(config or {})
    if "code_execution" in settings:
        settings = dict(settings["code_execution"])
    if not tests:
        raise ValueError("Code evaluation requires at least one official test")
    timeout = float(settings.get("timeout_seconds", 10))
    backend = settings.get("backend", "subprocess")
    if backend not in {"subprocess", "docker"}:
        raise ValueError(f"Unsupported code execution backend {backend}")
    payload = {"code": extract_python(code), "tests": tests, "setup": setup,
               "limits": {"cpu_seconds": max(1, int(timeout)), "memory_mb": int(settings.get("memory_mb", settings.get("memory_limit_mb", 1024)))}}
    start = time.monotonic()
    base = {"backend": backend, "isolation": "container" if backend == "docker" else "process_resource_limits",
            "security_boundary": backend == "docker", "timeout": False,
            "execution_protocol": EXECUTION_PROTOCOL_VERSION,
            "code_sha256": hashlib.sha256(payload["code"].encode()).hexdigest(),
            "setup_sha256": hashlib.sha256(setup.encode()).hexdigest(),
            "tests_sha256": hashlib.sha256(json.dumps(tests, ensure_ascii=False).encode()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="task-vector-code-") as tmp:
        directory = Path(tmp)
        (directory / "worker.py").write_text(_WORKER, encoding="utf-8")
        (directory / "payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if backend == "docker":
            image = settings.get("image", settings.get("docker_image", "python:3.11-slim"))
            container_name = f"task-vector-{os.getpid()}-{time.time_ns()}"
            command = [settings.get("docker_binary", "docker"), "run", "--rm", "--name", container_name,
                       "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                       "--pids-limit", "64", "--memory", f"{payload['limits']['memory_mb']}m", "--cpus", "1",
                       "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m",
                       "-v", f"{directory.resolve()}:/work:ro", image, "python", "-I", "/work/worker.py", "/work/payload.json"]
        else:
            command = [sys.executable, "-I", str(directory / "worker.py"), str(directory / "payload.json")]
        env = {key: os.environ[key] for key in ("PATH", "SystemRoot", "WINDIR", "LD_LIBRARY_PATH") if key in os.environ}
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
        with (directory / "stdout.txt").open("wb") as out, (directory / "stderr.txt").open("wb") as err:
            process = subprocess.Popen(command, cwd=tmp, env=env, stdout=out, stderr=err,
                                       start_new_session=os.name != "nt")
            try:
                process.wait(timeout=timeout + (15 if backend == "docker" else 1))
            except subprocess.TimeoutExpired:
                base["timeout"] = True
                if backend == "docker":
                    subprocess.run([settings.get("docker_binary", "docker"), "kill", container_name], capture_output=True, timeout=10)
                elif os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
        output = (directory / "stdout.txt").read_bytes()[:1024 * 1024].decode("utf-8", errors="replace")
        error = (directory / "stderr.txt").read_bytes()[:4000].decode("utf-8", errors="replace")
        try:
            detail = json.loads(output.strip().splitlines()[-1]) if output.strip() else {}
        except (ValueError, IndexError):
            detail = {}
        passed = detail.get("passed", [])
        base.update(detail)
        base.update({"pass_all": bool(len(passed) == len(tests) and all(passed) and not base["timeout"] and process.returncode == 0),
                     "test_count": len(tests), "returncode": process.returncode, "stderr": error,
                     "runtime_seconds": time.monotonic() - start})
        if not detail and not base["timeout"]:
            base["load_error"] = f"Worker exited without test results (exit {process.returncode})"
        return base


def _mutation_candidates(code: str):
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    substitutions = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.Div: ast.Mult,
                     ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv, ast.Eq: ast.NotEq,
                     ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt}
    for index, node in enumerate(nodes):
        mutations = []
        if isinstance(node, (ast.BinOp, ast.Compare)):
            operators = [node.op] if isinstance(node, ast.BinOp) else node.ops
            for op_index, operator in enumerate(operators):
                if type(operator) in substitutions:
                    mutations.append(("operator", op_index, substitutions[type(operator)]))
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            mutations.append(("constant", None, node.value + 1))
        if isinstance(node, ast.Return):
            mutations.append(("return_none", None, None))
        for kind, op_index, value in mutations:
            mutated = copy.deepcopy(tree)
            target = list(ast.walk(mutated))[index]
            if kind == "operator":
                if isinstance(target, ast.BinOp):
                    target.op = value()
                else:
                    target.ops[op_index] = value()
            elif kind == "constant":
                target.value = value
            else:
                target.value = ast.Constant(None)
            yield ast.unparse(ast.fix_missing_locations(mutated)), {"kind": kind, "node": index, "line": getattr(node, "lineno", None)}
    # A deterministic runtime fault provides coverage when subtle mutations are equivalent.
    for index, node in enumerate(nodes):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mutated = copy.deepcopy(tree)
            target = list(ast.walk(mutated))[index]
            offset = int(bool(target.body and isinstance(target.body[0], ast.Expr) and isinstance(target.body[0].value, ast.Constant) and isinstance(target.body[0].value.value, str)))
            target.body.insert(offset, ast.Raise(exc=ast.Call(func=ast.Name(id="RuntimeError", ctx=ast.Load()), args=[ast.Constant("injected runtime fault")], keywords=[])))
            yield ast.unparse(ast.fix_missing_locations(mutated)), {"kind": "runtime_fault", "node": index, "line": node.lineno}


def make_verified_bug(code: str, tests: list[str], setup: str = "", config: dict | None = None) -> dict:
    original = execute_code(code, tests, setup, config)
    if not original["pass_all"]:
        raise ValueError(f"Reference program does not pass official tests: {original}")
    limit = int((config or {}).get("code_execution", config or {}).get("max_mutations", 64))
    for index, (buggy, mutation) in enumerate(_mutation_candidates(extract_python(code))):
        if index >= limit:
            break
        failure = execute_code(buggy, tests, setup, config)
        if not failure["pass_all"]:
            return {"buggy_code": buggy, "mutation": mutation, "original_execution": original, "bug_execution": failure}
    # Always try an executed-function runtime fault even if the subtle mutation budget was exhausted.
    for buggy, mutation in _mutation_candidates(extract_python(code)):
        if mutation["kind"] == "runtime_fault":
            failure = execute_code(buggy, tests, setup, config)
            if not failure["pass_all"]:
                return {"buggy_code": buggy, "mutation": mutation, "original_execution": original, "bug_execution": failure}
    raise ValueError("No failing mutation verified; refusing to invent a repair example")
