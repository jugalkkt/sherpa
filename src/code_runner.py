"""Run a short Python snippet and report what it actually prints or raises.

Quiz questions about code are only trustworthy if the answer comes from running
the code, not from a 7B model's guess. The snippets are written by that model
(from your own notes), so they are treated as untrusted:

  1. validate_snippet(): an AST check refuses imports outside a small standard-
     library list, file/eval/exec-style builtins, and the dunder attributes
     used for sandbox escapes (__globals__, __subclasses__, __self__, ...).
  2. The snippet runs in a separate interpreter (-I -S) with an empty
     environment (no tokens or keys), a temp working directory, CPU, memory and
     file-size limits, a wall-clock timeout, and a capped stdout.
  3. Inside that process the snippet gets a reduced `builtins`.

This is defence in depth for model-written teaching snippets. It is not a
boundary against a determined attacker: do not feed it arbitrary user code.

The same file is the child process: `python -I -S code_runner.py` reads
{"code": ...} on stdin and writes one JSON result to stdout.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ALLOWED_IMPORTS = {"functools", "itertools", "collections", "math", "copy", "operator",
                   "typing", "dataclasses", "contextlib"}

# Builtins that read files, run code, or reach the interpreter's internals.
DENIED_NAMES = {"open", "exec", "eval", "compile", "input", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit", "memoryview"}

# Dunder attributes a teaching snippet may legitimately touch. Anything else
# starting with "__" is refused (__globals__, __subclasses__, __self__, ...).
ALLOWED_DUNDER_ATTRS = {
    "__name__", "__doc__", "__closure__", "__wrapped__", "__qualname__", "__module__", "__defaults__",
    "__class__", "__code__", "__init__", "__new__", "__call__", "__enter__", "__exit__", "__repr__",
    "__str__", "__len__", "__iter__", "__next__", "__getitem__", "__setitem__", "__contains__",
    "__eq__", "__lt__", "__add__", "__hash__", "__bool__",
}
# Frame/generator internals reachable without a dunder name (gi_frame.f_back.f_globals ...), and
# str.format, whose format string can walk attributes ("{0.__class__}") without an Attribute node.
DENIED_ATTR_PREFIXES = ("gi_", "cr_", "ag_", "f_", "tb_")
DENIED_ATTRS = {"format", "format_map"}

# Interpreter-level tripwire (sys.addaudithook), installed in the child before the snippet runs.
# It holds even if the AST check is bypassed (e.g. print.__self__ reaches the real builtins).
BLOCKED_EVENT_PREFIXES = ("os.", "subprocess.", "_posixsubprocess.", "socket.", "ctypes.", "shutil.", "pty.")
BLOCKED_PREFIX = "blocked:"

TIMEOUT_SECONDS = 5.0
CPU_SECONDS = 3
MEMORY_BYTES = 1024 * 1024 * 1024
MAX_OUTPUT_CHARS = 4000


@dataclass
class RunResult:
    rejected: str = ""   # why the snippet could not be judged (refused, timed out, ...): not its behaviour
    stdout: str = ""
    error: str = ""      # "UnboundLocalError: ..." when the snippet itself raised

    @property
    def ran(self) -> bool:
        return not self.rejected

    def describe(self) -> str:
        """What the snippet did, in words a learner would compare their answer to."""
        if self.stdout and self.error:
            return f"{self.stdout}\n...then raises {self.error}"
        if self.error:
            return f"Raises {self.error}"
        return self.stdout


# --- validation (parent side) -------------------------------------------------------------

def validate_snippet(code: str) -> str:
    """Why this snippet must not be run, or "" if it passes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"imports from {'.' * node.level}{node.module or ''}"
        elif isinstance(node, ast.Name):
            if node.id in DENIED_NAMES:
                return f"uses {node.id}()"
            if node.id.startswith("__") and node.id not in ("__name__", "__doc__"):
                return f"uses {node.id}"
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__") and attr not in ALLOWED_DUNDER_ATTRS:
                return f"uses .{attr}"
            if attr in DENIED_ATTRS or attr.startswith(DENIED_ATTR_PREFIXES):
                return f"uses .{attr}"
        elif isinstance(node, ast.keyword):
            if node.arg and node.arg.startswith("co_"):  # code.replace(co_code=...)
                return f"uses {node.arg}="
    return ""


# --- running (parent side) --------------------------------------------------------------------

def _limit_resources() -> None:  # runs in the child between fork and exec
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_snippet(code: str, timeout: float = TIMEOUT_SECONDS) -> RunResult:
    """Execute the snippet in the sandbox. Never raises."""
    reason = validate_snippet(code)
    if reason:
        return RunResult(rejected=reason)
    try:
        with tempfile.TemporaryDirectory() as workdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(Path(__file__).resolve())],
                input=json.dumps({"code": code}), capture_output=True, text=True, timeout=timeout,
                cwd=workdir, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"},
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
    except subprocess.TimeoutExpired:
        return RunResult(rejected=f"took longer than {timeout:g}s")
    except Exception as exc:
        return RunResult(rejected=f"could not run: {type(exc).__name__}")
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return RunResult(rejected="crashed (resource limit?)")
    if data.get("rejected"):
        return RunResult(rejected=data["rejected"])
    return RunResult(stdout=data.get("stdout", ""), error=data.get("error", ""))


# --- the child process ---------------------------------------------------------------------------

def _child_main() -> None:
    import builtins

    request = json.loads(sys.stdin.read())
    real_stdout = sys.stdout
    real_import = builtins.__import__

    class OutputTooLarge(Exception):
        pass

    class Capped:
        def __init__(self):
            self.parts, self.size = [], 0

        def write(self, text):
            self.size += len(text)
            if self.size > MAX_OUTPUT_CHARS:
                raise OutputTooLarge()
            self.parts.append(text)
            return len(text)

        def flush(self):
            pass

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"import of {name} is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    # Load the allowed modules now, so the snippet's imports touch no files once the hook is on.
    for module in ALLOWED_IMPORTS:
        real_import(module)

    def audit(event, args):
        if event == "open" or event.startswith(BLOCKED_EVENT_PREFIXES):
            raise RuntimeError(f"{BLOCKED_PREFIX} {event}")

    sys.addaudithook(audit)  # cannot be removed by the snippet

    safe = {k: v for k, v in vars(builtins).items() if k not in DENIED_NAMES}
    safe["__import__"] = guarded_import
    namespace = {"__builtins__": safe, "__name__": "__main__"}

    out = Capped()
    sys.stdout = out
    result: dict = {}
    try:
        exec(compile(request["code"], "<snippet>", "exec"), namespace)
    except OutputTooLarge:
        result["rejected"] = "printed too much"
    except MemoryError:
        result["rejected"] = "ran out of memory"
    except SyntaxError as exc:
        result["rejected"] = f"syntax error: {exc.msg}"
    except BaseException as exc:
        text = str(exc).splitlines()[0] if str(exc) else ""
        if text.startswith(BLOCKED_PREFIX):  # our tripwire, not the snippet's own behaviour
            result["rejected"] = "tried a blocked operation"
            text = ""
        else:
            result["error"] = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    finally:
        sys.stdout = real_stdout
    result["stdout"] = "\n".join(line.rstrip() for line in "".join(out.parts).splitlines()).strip()
    real_stdout.write(json.dumps(result))


if __name__ == "__main__":
    _child_main()
