PYTHON ?= venv/bin/python
PIP    ?= venv/bin/pip
GOAL   ?= Learn Python closures and decorators from scratch
LOGS   := data/logs

.PHONY: setup run streamlit run-goal resume services stop langfuse langfuse-stop \
        test eval test-all clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv, install deps, copy .env.example to .env if missing
	test -d venv || python3 -m venv venv
	$(PIP) install -r requirements.txt
	test -f .env || cp .env.example .env
	mkdir -p data

run:  ## Run a CLI session with the default goal
	$(PYTHON) main.py

run-goal:  ## Run a CLI session with a custom goal: make run-goal GOAL="..."
	$(PYTHON) main.py "$(GOAL)"

resume:  ## Resume a session: make resume ID=<session_id>
	@test -n "$(ID)" || (echo "Usage: make resume ID=<session_id>" && exit 1)
	$(PYTHON) main.py --resume $(ID)

streamlit:  ## Launch the Streamlit UI
	$(PYTHON) -m streamlit run streamlit_app.py

services:  ## Start both A2A services in the background (logs in data/logs)
	mkdir -p $(LOGS)
	nohup $(PYTHON) src/a2a_services/quiz_service.py > $(LOGS)/quiz_service.log 2>&1 &
	nohup $(PYTHON) src/crewai_agent/study_buddy.py > $(LOGS)/study_buddy.log 2>&1 &
	@echo "Quiz service  -> http://localhost:9001"
	@echo "Study buddy   -> http://localhost:9002"

stop:  ## Stop both A2A services
	-pkill -f "src/a2a_services/quiz_service.py"
	-pkill -f "src/crewai_agent/study_buddy.py"

langfuse:  ## Start the self-hosted Langfuse stack (http://localhost:3000)
	docker compose up -d

langfuse-stop:  ## Stop the Langfuse stack
	docker compose down

test:  ## Unit tests (no Ollama needed)
	$(PYTHON) -m pytest -m "not eval"

eval:  ## LLM-quality evals (needs Ollama)
	$(PYTHON) -m pytest tests/test_eval.py -m eval -s -v

test-all:  ## Unit tests, then evals
	$(MAKE) test
	$(MAKE) eval

clean:  ## Remove caches and runtime data
	find . -path ./venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache data
