# One-command entry points. Run `make setup` once.
BACKEND := cd backend &&
UV := uv run --frozen

.PHONY: setup models dev-backend dev-frontend check lint test evals evals-audio evals-stt bench synth

setup:            ## install backend (all providers) and frontend dependencies
	$(BACKEND) uv sync --extra all
	cd frontend && npm ci

models:           ## download the default local LLM (GGUF) into .local/
	./scripts/fetch_local_llm.sh

dev-backend:      ## API + WebSocket on :8000
	$(BACKEND) $(UV) python -m voicedesk

dev-frontend:     ## UI on :3000
	cd frontend && npm run dev

lint:
	$(BACKEND) $(UV) ruff check src tests && $(UV) ruff format --check src tests && $(UV) mypy src
	cd frontend && npm run typecheck

test:
	$(BACKEND) $(UV) pytest -q
	cd frontend && npm test

check: lint test  ## the full local check: lint, types, unit + integration tests

evals:            ## agent evals on scripted conversations (text in, real LLM)
	$(BACKEND) $(UV) python -m voicedesk.evals run --mode text

evals-audio:      ## full understanding path: audio fixtures -> STT -> language ID -> agent
	$(BACKEND) $(UV) python -m voicedesk.evals run --mode audio

evals-stt:        ## STT + language ID only (no LLM): cheap enough for every CI run
	$(BACKEND) $(UV) python -m voicedesk.evals run --mode audio --stage stt --min-score 0.95

synth:            ## regenerate audio fixtures with edge-tts (needs internet)
	$(BACKEND) $(UV) python -m voicedesk.evals synth --force

bench:            ## latency benchmark against a running backend (make dev-backend first)
	$(BACKEND) $(UV) python -m voicedesk.bench --repeat 2
