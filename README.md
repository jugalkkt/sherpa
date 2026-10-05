# Sherpa

A local, zero-API-key, multi-agent study companion built with LangGraph, MCP,
A2A and Ollama. Give it a learning goal and it plans a roadmap, explains each
topic from your own Markdown notes, quizzes you, and coaches you on what to
review.

> Work in progress. See [plan.md](plan.md) for the build plan.
Next step is phase 9

## Quick start

```bash
ollama pull qwen2.5:7b
make setup          # venv + deps + .env
make test           # unit tests, no Ollama needed
make help           # every target
```

Your notes live in `study_materials/sample_notes/` (set `NOTES_PATH` in `.env`
to use another folder).

## Web app and hosting

```bash
make streamlit      # the web UI locally
```

To host it for free (Streamlit Community Cloud, with a Kaggle GPU you start
from the app that stops itself when idle), see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
