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

### Your own notes

On the start screen you can upload up to 5 `.md` or `.txt` files (max 15 KB
each) related to your learning goal. They replace the sample notes for that
session, and the roadmap is built **only** from them: your notes are split
into sections (at `#` and `##` headings), every section is assigned to one of
4-6 topics, and each topic is explained from exactly its sections. The goal
is then just the roadmap's label. The approval screen shows which sections
each topic teaches. Topics whose notes have no Python code get an example in
words and concept questions only.

Without uploads, the sample notes are used and the roadmap comes from the
goal alone. Notes are fixed once the session starts; to change them, use
**Start over**.

## Limitations

- **Uploads are capped at 5 files × 15 KB.** The hosted model (`qwen2.5:7b`)
  runs with an 8,192-token context (`CONTEXT_LENGTH` in
  `deploy/kaggle/sherpa_ollama_server.py`), about 25 KB of text shared by the
  prompt, the notes and the reply. The Explainer must fit one whole file in
  it, so the limits are kept small for now. **Planned:** host a bigger model
  with a longer context on Kaggle, then raise these limits (see
  [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#limits-to-know)).
- **The planner sees each section's heading and first lines**, not the full
  text (about 6,000 characters in all).
- **A topic's notes are cut at 12,000 characters** when the Explainer reads
  them, for the same context reason; the explanation then says so.
- **"Make a new one" may give a very similar roadmap** from the same notes
  (the planner runs at a low temperature).
- **Uploads last for one browser session.** They are not saved between visits;
  a refresh or an app restart loses them.
- **Only `.md` and `.txt`.** PDFs and Word files aren't supported.
- The 7B model sometimes writes partly wrong expected answers for quiz
  questions (see [docs/DEVIATIONS.md](docs/DEVIATIONS.md)).
