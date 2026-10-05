"""Start the Kaggle model server and report whether it's up.

The server itself is deploy/kaggle/sherpa_ollama_server.py: Ollama + ngrok on a
Kaggle GPU, which stops itself after SERVER_IDLE_MINUTES without requests.
This module pushes that script as a Kaggle kernel (= starts it) and combines
Kaggle's kernel status with a health check of the tunnel.

There is no Stop: Kaggle's API has no usable way to cancel a run, so the
server's idle watchdog (plus Kaggle's own session timeout) ends it.

Settings (env / Streamlit secrets, read at call time):
    KAGGLE_API_TOKEN         Kaggle API token (or ~/.kaggle/access_token locally)
    KAGGLE_KERNEL_SLUG       e.g. "yourname/sherpa-ollama-server"
    KAGGLE_SECRETS_DATASET   e.g. "yourname/sherpa-server-secrets" (private)
    SERVER_IDLE_MINUTES      default 30
    SERVER_MAX_HOURS         default 3
    KAGGLE_ACCELERATOR       default "NvidiaTeslaT4"
    OLLAMA_MODEL / OLLAMA_BASE_URL / OLLAMA_BASIC_AUTH  (same as the app)

CLI:  python -m hosting.kaggle_server {upload-secrets|start|status}
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from llm import auth_headers, base_url, model_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = PROJECT_ROOT / "deploy" / "kaggle" / "sherpa_ollama_server.py"
SECRETS_FILE = "sherpa_secrets.json"
REQUIRED_SETTINGS = ("KAGGLE_KERNEL_SLUG", "KAGGLE_SECRETS_DATASET", "OLLAMA_BASE_URL", "OLLAMA_BASIC_AUTH")

# Kaggle KernelWorkerStatus names -> our states (the tunnel check can upgrade to "ready").
_KERNEL_STATES = {
    "QUEUED": "starting", "NEW_SCRIPT": "starting", "RUNNING": "starting",
    "COMPLETE": "stopped", "CANCEL_REQUESTED": "stopped", "CANCEL_ACKNOWLEDGED": "stopped",
    "ERROR": "error",
}


@dataclass
class ServerStatus:
    state: str  # not_configured | stopped | starting | ready | error
    detail: str

    @property
    def ready(self) -> bool:
        return self.state == "ready"


def idle_minutes() -> int:
    return int(os.getenv("SERVER_IDLE_MINUTES") or 30)


def max_hours() -> float:
    return float(os.getenv("SERVER_MAX_HOURS") or 3)


def missing_settings() -> list[str]:
    return [name for name in REQUIRED_SETTINGS if not (os.getenv(name) or "").strip()]


def render_server_script(model: str, idle: int, hours: float, template: str | None = None) -> str:
    """The server script with its three settings lines filled in."""
    text = template if template is not None else SERVER_SCRIPT.read_text(encoding="utf-8")
    replacements = {
        r"^MODEL = .*$": f"MODEL = {json.dumps(model)}",
        r"^IDLE_MINUTES = .*$": f"IDLE_MINUTES = {int(idle)}",
        r"^MAX_HOURS = .*$": f"MAX_HOURS = {float(hours)}",
    }
    for pattern, line in replacements.items():
        text, count = re.subn(pattern, line, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Server script has no line matching {pattern!r}")
    return text


def kernel_metadata(kernel_slug: str, dataset_slug: str) -> dict:
    return {
        "id": kernel_slug,
        "title": kernel_slug.split("/", 1)[1],  # a new kernel's title must match its slug
        "code_file": "sherpa_ollama_server.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": os.getenv("KAGGLE_ACCELERATOR") or "NvidiaTeslaT4",
        "dataset_sources": [dataset_slug],
    }


def _kaggle_api():
    """An authenticated Kaggle client. Raises RuntimeError (never SystemExit) if login fails.

    `import kaggle` itself tries to authenticate (a network call) and, with no token, prints a long
    help text; authenticate() then calls sys.exit(), which a normal `except Exception` doesn't catch
    and which would kill the web page's script run. So the import and the login both run with
    their output captured, and SystemExit becomes an ordinary error.
    """
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
    except SystemExit:
        raise RuntimeError("Kaggle login failed. Set KAGGLE_API_TOKEN (kaggle.com > Settings > API) "
                           "in the app's secrets or .env.") from None
    return api


def model_reachable(timeout: float = 5.0) -> bool:
    """Does the tunnel answer, with our credentials?"""
    try:
        r = httpx.get(f"{base_url()}/api/tags", headers=auth_headers(), timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def server_status(api=None) -> ServerStatus:
    """Never raises: every problem becomes a state with a readable detail."""
    # Reachable wins: a local Ollama or a notebook you started by hand counts too.
    if model_reachable():
        return ServerStatus("ready", f"Model server is up ({model_name()}).")
    missing = missing_settings()
    if missing:
        return ServerStatus("not_configured", f"Model not reachable, and Start needs: {', '.join(missing)}")
    try:
        api = api or _kaggle_api()
        response = api.kernels_status(os.environ["KAGGLE_KERNEL_SLUG"])
    except Exception as exc:
        text = str(exc)
        # A kernel that was never pushed answers 401/403, which the Kaggle client
        # words as "permission denied ... wrong kernel slug" (or a plain 404).
        if "404" in text or "not found" in text.lower() or "wrong kernel slug" in text:
            return ServerStatus("stopped", "Not started yet. Press Start to launch it.")
        return ServerStatus("error", f"Couldn't read the Kaggle status: {type(exc).__name__}: {text[:200]}")

    status = getattr(response.status, "name", str(response.status)).upper()
    state = _KERNEL_STATES.get(status, "error")
    if state == "starting":
        detail = ("Waiting for a Kaggle GPU..." if status in ("QUEUED", "NEW_SCRIPT")
                  else "Kaggle is installing Ollama and loading the model (a few minutes)...")
    elif state == "stopped":
        detail = "Stopped (idle shutdown or finished). Press Start to bring it back."
    else:
        detail = f"The Kaggle run failed: {getattr(response, 'failure_message', '') or 'see the notebook logs'}"
    return ServerStatus(state, detail)


def start_server(api=None) -> tuple[bool, str]:
    """Push the server script as a Kaggle kernel, which starts it. Never raises."""
    missing = missing_settings()
    if missing:
        return False, f"Missing settings: {', '.join(missing)}"
    current = server_status(api)
    if current.state in ("ready", "starting"):
        return False, f"Already {current.state}: {current.detail}"
    try:
        api = api or _kaggle_api()
        with tempfile.TemporaryDirectory() as folder:
            script = render_server_script(model_name(), idle_minutes(), max_hours())
            (Path(folder) / "sherpa_ollama_server.py").write_text(script, encoding="utf-8")
            meta = kernel_metadata(os.environ["KAGGLE_KERNEL_SLUG"], os.environ["KAGGLE_SECRETS_DATASET"])
            (Path(folder) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            # Kaggle's own hard stop: our cap plus 15 minutes of setup margin.
            response = api.kernels_push(folder, timeout=str(int(max_hours() * 3600 + 900)))
    except Exception as exc:
        return False, f"Couldn't start the Kaggle run: {type(exc).__name__}: {str(exc)[:200]}"
    if getattr(response, "error", None):
        return False, f"Kaggle rejected the run: {response.error}"
    return True, f"Started. Kaggle run: {getattr(response, 'url', '') or os.environ['KAGGLE_KERNEL_SLUG']}"


def secrets_payload() -> dict:
    """What the Kaggle server needs, built from the same settings the app uses."""
    token = (os.getenv("NGROK_AUTHTOKEN") or "").strip()
    if not token:
        raise ValueError("NGROK_AUTHTOKEN is not set")
    creds = (os.getenv("OLLAMA_BASIC_AUTH") or "").strip()
    if ":" not in creds:
        raise ValueError("OLLAMA_BASIC_AUTH must look like user:password")
    url = (os.getenv("OLLAMA_BASE_URL") or "").strip().rstrip("/")
    if not url.startswith("https://"):
        raise ValueError("OLLAMA_BASE_URL must be your https:// ngrok domain")
    return {"ngrok_authtoken": token, "ngrok_url": url, "basic_auth": creds}


def upload_secrets(api=None) -> str:
    """Create (or update) the PRIVATE Kaggle dataset holding the server's secrets.

    The secrets file only exists in a temporary folder during the upload.
    """
    slug = os.getenv("KAGGLE_SECRETS_DATASET") or ""
    if "/" not in slug:
        raise ValueError("Set KAGGLE_SECRETS_DATASET to yourname/dataset-slug")
    payload = secrets_payload()
    api = api or _kaggle_api()
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / SECRETS_FILE).write_text(json.dumps(payload), encoding="utf-8")
        meta = {"title": slug.split("/", 1)[1], "id": slug, "licenses": [{"name": "CC0-1.0"}]}
        (Path(folder) / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        result = api.dataset_create_new(folder, public=False, quiet=True)
        if getattr(result, "error", None):
            # Already exists: publish a new version, dropping old ones (they hold old secrets).
            result = api.dataset_create_version(folder, "Update Sherpa server secrets", quiet=True,
                                                delete_old_versions=True)
    if getattr(result, "error", None):
        raise RuntimeError(f"Kaggle rejected the dataset: {result.error}")
    return f"Private dataset {slug} is up to date."


def main(argv: list[str]) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    command = argv[0] if argv else "status"
    if command == "upload-secrets":
        print(upload_secrets())
    elif command == "start":
        ok, message = start_server()
        print(message)
        return 0 if ok else 1
    elif command == "status":
        status = server_status()
        print(f"{status.state}: {status.detail}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.exit(main(sys.argv[1:]))
