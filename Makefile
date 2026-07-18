.DEFAULT_GOAL := help
SERVICE_NAME := edgar-rag

.PHONY: help
help: ## Show this help.
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

# Local dev environment via uv, bootstrapped inside a stdlib venv so uv does not
# need to be pre-installed on the host. Rebuilt when pyproject.toml changes.
.venv: pyproject.toml
	python3 -m venv .venv --prompt $(SERVICE_NAME) \
	&& . .venv/bin/activate \
	&& pip install --quiet --upgrade pip uv \
	&& uv sync --extra dev

.PHONY: setup
setup: .venv ## Create/refresh the local virtualenv (uv sync --extra dev).

.PHONY: lint
lint: .venv ## Lint the codebase with ruff.
	. .venv/bin/activate && ruff check .

.PHONY: test
test: .venv ## Run local testing (lint + unit tests).
	$(MAKE) lint
	. .venv/bin/activate \
	&& set -a && { [ -f .env ] && . ./.env || true; } && set +a \
	&& pytest
