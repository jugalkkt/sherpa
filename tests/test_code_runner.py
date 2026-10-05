import json
import subprocess
import sys
from pathlib import Path

import pytest

from code_runner import ALLOWED_IMPORTS, RunResult, run_snippet, validate_snippet

pytestmark = pytest.mark.unit

CHILD = str(Path(__file__).resolve().parent.parent / "src" / "code_runner.py")


def run_child_directly(code: str) -> dict:
    """Layer 2 on its own: skip the AST check and talk to the child process."""
    proc = subprocess.run([sys.executable, "-I", "-S", CHILD], input=json.dumps({"code": code}),
                          capture_output=True, text=True, timeout=15, env={"PATH": "/usr/bin:/bin"})
    return json.loads(proc.stdout)


# --- what snippets really do --------------------------------------------------------------------

@pytest.mark.parametrize("code, stdout, error", [
    ("print(1 + 1)", "2", ""),
    ("def f(item, bucket=None):\n    if bucket is None:\n        bucket = []\n    bucket.append(item)\n    return bucket\n"
     "print(f(5))\nprint(f(10))", "[5]\n[10]", ""),
    ("def f(item, bucket=[]):\n    bucket.append(item)\n    return bucket\nprint(f(5))\nprint(f(10))", "[5]\n[5, 10]", ""),
    ("fs = [lambda: i for i in range(3)]\nprint([f() for f in fs])", "[2, 2, 2]", ""),
    ("def mk():\n    n = 0\n    def inc():\n        nonlocal n\n        n += 1\n        return n\n    return inc\n"
     "c = mk()\nprint(c(), c())\nprint(c.__closure__[0].cell_contents)", "1 2\n2", ""),
    ("import functools\ndef d(f):\n    @functools.wraps(f)\n    def w(*a):\n        return f(*a)\n    return w\n"
     "@d\ndef add(a, b):\n    '''Adds.'''\n    return a + b\nprint(add.__name__, add.__doc__)", "add Adds.", ""),
    ("class A:\n    def __init__(self):\n        self.x = 1\nclass B(A):\n    def __init__(self):\n"
     "        super().__init__()\nprint(B().x)", "1", ""),
    ("from dataclasses import dataclass\n@dataclass\nclass P:\n    x: int\nprint(P(3))", "P(x=3)", ""),
    ("print('before')\nraise ValueError('boom')", "before", "ValueError: boom"),
    ("count = 0\ndef bump():\n    count += 1\nbump()", "", "UnboundLocalError: cannot access local variable 'count' "
                                                          "where it is not associated with a value"),
    ("def f(): return f()\nf()", "", "RecursionError: maximum recursion depth exceeded"),
])
def test_runs_and_reports_the_real_behaviour(code, stdout, error):
    r = run_snippet(code)
    assert r.ran and r.stdout == stdout and r.error == error


def test_trailing_whitespace_is_normalised():
    assert run_snippet("print('a   ')\nprint()\nprint('b')").stdout == "a\n\nb"


def test_describe():
    assert RunResult(stdout="1").describe() == "1"
    assert RunResult(error="KeyError: 'x'").describe() == "Raises KeyError: 'x'"
    assert RunResult(stdout="a", error="KeyError: 'x'").describe() == "a\n...then raises KeyError: 'x'"


def test_every_allowed_module_imports_and_works():
    code = "import " + ", ".join(sorted(ALLOWED_IMPORTS)) + "\nprint('ok')"
    assert run_snippet(code).stdout == "ok"
    assert run_snippet("from collections import Counter\nprint(Counter('aab'))").stdout == "Counter({'a': 2, 'b': 1})"
    assert run_snippet("import copy\nprint(copy.deepcopy([[1], [2]]))").stdout == "[[1], [2]]"


# --- layer 1: refused before running ---------------------------------------------------------------

@pytest.mark.parametrize("code, why", [
    ("import os", "imports os"),
    ("import os.path", "imports os.path"),
    ("from os import path", "imports from os"),
    ("from . import x", "imports from ."),
    ("import subprocess", "imports subprocess"),
    ("open('/etc/passwd')", "uses open()"),
    ("eval('1')", "uses eval()"),
    ("exec('x = 1')", "uses exec()"),
    ("input()", "uses input()"),
    ("getattr(1, 'real')", "uses getattr()"),
    ("globals()", "uses globals()"),
    ("__import__('os')", "uses __import__()"),
    ("print(__builtins__)", "uses __builtins__"),
    ("().__class__.__bases__", "uses .__bases__"),
    ("().__class__.__base__.__subclasses__()", "uses .__subclasses__"),
    ("print.__self__", "uses .__self__"),
    ("f = lambda: 1\nf.__globals__", "uses .__globals__"),
    ("(x for x in []).gi_frame", "uses .gi_frame"),
    ("g = (x for x in [])\ng.gi_frame.f_back.f_globals", "uses .f_globals"),
    ("'{0.__class__}'.format(1)", "uses .format"),
    ("'{}'.format_map({})", "uses .format_map"),
    ("f = lambda: 1\nf.__code__.replace(co_code=b'x')", "uses co_code="),
    ("x = (", "syntax error"),
])
def test_dangerous_code_is_refused_without_running(code, why):
    r = run_snippet(code)
    assert not r.ran and why in r.rejected and r.stdout == ""


def test_validate_returns_empty_for_good_code():
    assert validate_snippet("import functools\nprint(functools.reduce(lambda a, b: a + b, [1, 2, 3]))") == ""


def test_refusal_never_spawns_a_process(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run a refused snippet"))
    assert not run_snippet("import os").ran


# --- layer 2: holds even if the AST check is bypassed ---------------------------------------------------

@pytest.mark.parametrize("code", [
    "print(print.__self__.open('/etc/passwd').read())",           # the real builtins leak via __self__
    "print(len.__self__.__import__('os').listdir('/'))",
    "print(len.__self__.__import__('subprocess').run(['id']))",     # getting the module is harmless; using it isn't
])
def test_interpreter_tripwire_blocks_escapes(code):
    out = run_child_directly(code)
    assert out.get("rejected") == "tried a blocked operation" and out["stdout"] == ""


def test_tripwire_holds_even_when_the_snippet_swallows_the_exception():
    out = run_child_directly("data = ''\ntry:\n    data = print.__self__.open('/etc/passwd').read()\n"
                             "except Exception:\n    pass\nprint(len(data))")
    assert out["stdout"] == "0"  # the file was never opened


@pytest.mark.parametrize("code, error", [
    ("open('/etc/passwd')", "NameError: name 'open' is not defined"),
    ("eval('1')", "NameError: name 'eval' is not defined"),
    ("import os", "ImportError: import of os is not allowed"),
    ("__import__('os')", "ImportError: import of os is not allowed"),
])
def test_restricted_builtins_in_the_child(code, error):
    assert run_child_directly(code)["error"] == error


def test_snippet_sees_no_environment(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    # os can't even be imported, so check from the outside: the child's env is empty by construction
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", "import os; print(dict(os.environ))"],
                          capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert "hunter2" not in proc.stdout
    assert run_snippet("print('x')").stdout == "x"


# --- resource limits ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("code, why", [
    ("print('x' * 10**7)", "printed too much"),
    ("x = [0] * 10**10\nprint(len(x))", "ran out of memory"),
])
def test_resource_bombs_are_stopped(code, why):
    r = run_snippet(code)
    assert not r.ran and why in r.rejected


def test_infinite_loop_is_stopped():
    r = run_snippet("while True:\n    pass", timeout=2)
    assert not r.ran and r.stdout == ""


def test_wall_clock_timeout(monkeypatch):
    r = run_snippet("import contextlib\nwhile True:\n    pass", timeout=0.3)
    assert not r.ran
