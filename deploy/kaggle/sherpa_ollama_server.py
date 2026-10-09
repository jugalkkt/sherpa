"""Sherpa's model server on Kaggle: Ollama behind an ngrok tunnel, stops itself when idle.

Pushed and started by `src/hosting/kaggle_server.py` (the app's Start button)
as a Kaggle *script* kernel with GPU and internet enabled. Kaggle shows this
script's output under the notebook's "Logs".

Secrets: Kaggle Secrets aren't reliably available in API-started runs, so they
come from a PRIVATE Kaggle dataset attached to the kernel. It holds one file,
sherpa_secrets.json (created by `python -m hosting.kaggle_server upload-secrets`):
    {"ngrok_authtoken": "...", "ngrok_url": "https://<your-domain>",
     "basic_auth": "user:password"}

Idle shutdown: Ollama's keep-alive is IDLE_MINUTES, so the model unloads
IDLE_MINUTES after the last request. The watchdog exits as soon as no model is
loaded, which ends the Kaggle session and frees the GPU. MAX_HOURS is a hard
cap on top (and the launcher also sets Kaggle's own session timeout).
"""

from __future__ import annotations

import base64
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --- settings: the launcher rewrites these three lines before pushing ----------
MODEL = "qwen2.5:7b"
IDLE_MINUTES = 10
MAX_HOURS = 3.0
# -------------------------------------------------------------------------------

CONTEXT_LENGTH = 8192
OLLAMA = "http://127.0.0.1:11434"
NGROK_TGZ = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
SECRETS_FILE = "sherpa_secrets.json"
WORKDIR = "/tmp/sherpa-server"
CHECK_EVERY_SECONDS = 30


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- pure helpers (unit-tested) ------------------------------------------------------

def find_secrets(root: str = "/kaggle/input") -> dict:
    """Load sherpa_secrets.json from wherever Kaggle mounted the private dataset."""
    matches = sorted(glob.glob(os.path.join(root, "**", SECRETS_FILE), recursive=True))
    if not matches:
        raise SystemExit(f"{SECRETS_FILE} not found under {root}. Attach the private secrets "
                         "dataset (run: python -m hosting.kaggle_server upload-secrets).")
    with open(matches[0], encoding="utf-8") as f:
        secrets = json.load(f)
    missing = [k for k in ("ngrok_authtoken", "ngrok_url", "basic_auth") if not secrets.get(k)]
    if missing:
        raise SystemExit(f"{SECRETS_FILE} is missing: {', '.join(missing)}")
    if ":" not in secrets["basic_auth"]:
        raise SystemExit("basic_auth must look like user:password")
    return secrets


def traffic_policy(basic_auth: str) -> str:
    """ngrok traffic policy: require basic auth, then rewrite Host so Ollama accepts
    the request. JSON is valid YAML, and json.dumps escapes any odd password characters."""
    return json.dumps({
        "on_http_request": [{
            "actions": [
                {"type": "basic-auth", "config": {"credentials": [basic_auth], "enforce": True}},
                {"type": "add-headers", "config": {"headers": {"host": "localhost"}}},
            ],
        }],
    }, indent=2)


def stop_reason(loaded_models: list, started_at: float, now: float, max_hours: float) -> str | None:
    """Why the server should stop now, or None to keep running."""
    if not loaded_models:
        return f"idle: no requests for {IDLE_MINUTES} minutes (model unloaded)"
    if now - started_at >= max_hours * 3600:
        return f"reached the {max_hours:g}-hour cap"
    return None


# --- side-effecting steps --------------------------------------------------------------

def sh(cmd: str) -> None:
    log(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def http_json(url: str, payload: dict | None = None, headers: dict | None = None, timeout: float = 10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def wait_until(check, what: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"Timed out waiting for {what}")


def start_ollama() -> subprocess.Popen:
    sh("apt-get update -qq && apt-get install -y -qq zstd pciutils lshw > /dev/null")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_KEEP_ALIVE": f"{IDLE_MINUTES}m",
           # The Explainer reads whole note files in a tool loop; Ollama's default is smaller.
           "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH)}
    proc = subprocess.Popen(["ollama", "serve"], env=env,
                            stdout=open(f"{WORKDIR}/ollama.log", "w"), stderr=subprocess.STDOUT)
    wait_until(lambda: http_json(f"{OLLAMA}/api/tags"), "Ollama to start", 120)
    sh(f"ollama pull {MODEL}")
    # An empty prompt loads the model; keep_alive starts the idle clock.
    log("Loading the model into GPU memory...")
    http_json(f"{OLLAMA}/api/generate", {"model": MODEL, "prompt": "", "keep_alive": f"{IDLE_MINUTES}m"},
              timeout=600)
    return proc


def start_ngrok(secrets: dict) -> subprocess.Popen:
    sh(f"curl -fsSL {NGROK_TGZ} | tar -xz -C {WORKDIR}")
    ngrok = f"{WORKDIR}/ngrok"
    subprocess.run([ngrok, "config", "add-authtoken", secrets["ngrok_authtoken"]], check=True,
                   stdout=subprocess.DEVNULL)
    policy = f"{WORKDIR}/policy.json"
    with open(policy, "w") as f:
        f.write(traffic_policy(secrets["basic_auth"]))
    url = secrets["ngrok_url"].rstrip("/")
    proc = subprocess.Popen([ngrok, "http", "11434", "--url", url, "--traffic-policy-file", policy,
                             "--log", "stdout"], stdout=open(f"{WORKDIR}/ngrok.log", "w"),
                            stderr=subprocess.STDOUT)
    auth = "Basic " + base64.b64encode(secrets["basic_auth"].encode()).decode()
    headers = {"Authorization": auth, "ngrok-skip-browser-warning": "true"}
    wait_until(lambda: http_json(f"{url}/api/tags", headers=headers), "the ngrok tunnel", 90)
    return proc


def main() -> None:
    os.makedirs(WORKDIR, exist_ok=True)
    started_at = time.time()
    secrets = find_secrets()
    log(f"Starting Sherpa model server: {MODEL}, idle stop after {IDLE_MINUTES} min, cap {MAX_HOURS:g} h")
    ollama = start_ollama()
    ngrok = start_ngrok(secrets)
    log(f"READY at {secrets['ngrok_url']}")

    reason = None
    while reason is None:
        time.sleep(CHECK_EVERY_SECONDS)
        if ollama.poll() is not None:
            reason = "Ollama exited (see ollama.log)"
        elif ngrok.poll() is not None:
            reason = "ngrok exited (another agent using the same domain?)"
        else:
            try:
                loaded = http_json(f"{OLLAMA}/api/ps").get("models") or []
            except (urllib.error.URLError, OSError, ValueError):
                loaded = ["(Ollama busy; treat as active)"]
            reason = stop_reason(loaded, started_at, time.time(), MAX_HOURS)

    log(f"Stopping: {reason}")
    for proc in (ngrok, ollama):
        proc.terminate()
    log("Stopped. The Kaggle session ends now and the GPU is released.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            log(f"ERROR: {exc.code}")
        raise
    sys.exit(0)
