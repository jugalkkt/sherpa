# Hosting Sherpa (free)

```
Browser ──► Streamlit Community Cloud (the web app, password-protected)
                 │  ▶ Start button: pushes a Kaggle script kernel
                 ▼
            Kaggle GPU: Ollama + ngrok  ──(stops itself after 10 idle minutes)
                 ▲
                 └── the app calls https://<your-ngrok-domain> with basic auth
```

You press **Start** in the sidebar, Kaggle brings up the model in a few
minutes, you study, and the server shuts itself down 10 minutes after the
last request.

## What you need (one time)

| Thing | Where |
|---|---|
| ngrok account with a **static domain** and an **authtoken** | ngrok dashboard |
| Kaggle account, **phone-verified** (required for GPU and internet) | kaggle.com → Settings |
| Kaggle **API token** | kaggle.com → Settings → API. Locally it lives in `~/.kaggle/access_token` |
| GitHub repo with this project | for Streamlit Community Cloud |

## 1. Fill in `.env`

```
OLLAMA_BASE_URL=https://<your-ngrok-domain>
OLLAMA_BASIC_AUTH=<user>:<password>        # pick any; the tunnel will require it
OLLAMA_MODEL=qwen2.5:7b
NGROK_AUTHTOKEN=<from the ngrok dashboard>
KAGGLE_KERNEL_SLUG=<kaggle-username>/sherpa-ollama-server
KAGGLE_SECRETS_DATASET=<kaggle-username>/sherpa-server-secrets
APP_PASSWORD=<password for the web app>
SERVER_IDLE_MINUTES=10
SERVER_MAX_HOURS=3
```

## 2. Upload the server's secrets to Kaggle

```bash
make server-secrets
```

This creates a **private** Kaggle dataset holding the ngrok token, domain and
basic-auth credentials. The Kaggle server reads them from there, because
Kaggle Secrets don't work reliably in runs started through the API. Run it
again whenever you change any of those three values; old versions are deleted.

## 3. Try it locally

```bash
make streamlit
```

Sign in, press **▶ Start model server** in the sidebar and wait for 🟢 ready.
It takes a few minutes: Kaggle finds a GPU, installs Ollama, downloads the
model and opens the tunnel. Then start a session.

Without the UI: `make server-start`, then `make server-status`. You can watch
the server's log on kaggle.com under Code → sherpa-ollama-server → Logs.

## 4. Deploy to Streamlit Community Cloud (the public URL)

1. **Push the repo to GitHub.** `.env`, `data/` and `.streamlit/secrets.toml` are
   git-ignored; nothing secret is in the repo (checked by searching every
   committable file for the real secret values). I couldn't confirm from
   Streamlit's docs whether free accounts can deploy from a *private* repo; a
   public repo works either way, and it contains no secrets.
2. Go to **share.streamlit.io**, sign in with GitHub, click **Create app** →
   **"Yup, I have an app"**.
3. Fill in: your repository, branch, and **Main file path:**
   `deploy/streamlit/streamlit_app.py`
   (this makes Cloud use the slim `deploy/streamlit/requirements.txt`).
4. Choose an **App URL**, e.g. `sherpa-study` → `https://sherpa-study.streamlit.app`.
5. **Advanced settings:** Python version **3.11**, and paste the contents of
   `.streamlit/secrets.toml` (generated for you, git-ignored) into **Secrets**.
   Then add one more line yourself, your Kaggle API token:
   `KAGGLE_API_TOKEN = "<contents of ~/.kaggle/access_token>"`
6. **Deploy.** The first build takes a few minutes.

`NGROK_AUTHTOKEN` is not needed in the cloud secrets; only `make server-secrets`
(run on your machine) uses it.

Open the URL, enter `APP_PASSWORD`, press **▶ Start model server**, wait for 🟢,
and study. Anyone with the URL *and* the password can start your Kaggle GPU, so
share the password only with people you trust.

To change a secret later: app **Settings → Secrets**. Pushing to the repo
redeploys automatically.

## How shutdown works

- Ollama keeps the model loaded for `SERVER_IDLE_MINUTES` after the last
  request. The server checks every 30 s and exits once the model has unloaded,
  which ends the Kaggle session and frees the GPU.
- Hard caps: the server exits after `SERVER_MAX_HOURS`, and Kaggle itself
  kills the run 15 minutes after that.
- There is no Stop button: Kaggle's API has no usable way to cancel a run.

## Limits to know

- **Kaggle GPU quota** is about 30 hours a week; every run counts.
- **ngrok's free plan allows one agent at a time.** Stop any notebook you
  started by hand before pressing Start, or the tunnel won't come up.
- **Sessions don't survive an app restart.** Community Cloud wipes local files
  when the app reboots, redeploys or sleeps (after 12 hours without visitors),
  so a half-finished session can't be resumed after that.
- **Kaggle's terms.** Running a tunnelled model server from a notebook is a grey
  area; the risk is to your Kaggle account.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Kaggle rejected the run" mentioning verification | Phone-verify your Kaggle account |
| Status goes 🟡 → ⚪ quickly | The run failed early: read the Logs on kaggle.com |
| Logs: `sherpa_secrets.json not found` | Run `make server-secrets`; check `KAGGLE_SECRETS_DATASET` |
| Logs: `ngrok exited` | Another ngrok agent is using the domain, or the token is wrong |
| 🟢 never comes, Logs say READY | `OLLAMA_BASE_URL` / `OLLAMA_BASIC_AUTH` in the app don't match the secrets you uploaded |
| Sidebar 🔴 "Kaggle login failed" | `KAGGLE_API_TOKEN` is missing or expired in the app's secrets |
| Sidebar 🔴 "Couldn't read the Kaggle status" | A wrong `KAGGLE_KERNEL_SLUG`, or Kaggle is unreachable |
| Sidebar flips to 🟡 "starting" for a minute mid-session | The tunnel check timed out while the model was busy; it recovers by itself |
