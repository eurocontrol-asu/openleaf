.PHONY: install check test test-all test-core test-performance lint format type-check docs-serve docs-build clean help

# Workspace Management

install:  ## Install all dependencies
	uv sync --all-groups

# Testing

test-all:  ## Run ALL workspace tests (per package)
	@echo "Running ALL workspace tests (per package)..."
	@for pkg in openleaf-core openleaf-performance; do \
		echo "\nTesting $$pkg..."; \
		uv run --package $$pkg --directory packages/$$pkg pytest || test $$? -eq 5 || exit 1; \
	done

test-core:  ## Run openleaf-core tests
	uv run --package openleaf-core --directory packages/openleaf-core pytest

test-performance:  ## Run openleaf-performance tests
	uv run --package openleaf-performance --directory packages/openleaf-performance pytest

# Quality Gates

lint:  ## Linter + formatter check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format code
	uv run ruff format .
	uv run ruff check --fix .

type-check:  ## Run mypy per package
	@echo "Running mypy per package..."
	uv run --package openleaf-core mypy --config-file packages/openleaf-core/pyproject.toml packages/openleaf-core/src/ packages/openleaf-core/tests/
	cd packages/openleaf-performance && uv run mypy src/ tests/

check: lint type-check test-all  ## Lint + type-check + tests

test: test-all  ## Run all workspace tests (alias)

# Documentation

docs-serve:  ## Preview docs locally
	uv run mkdocs serve

docs-build:  ## Build docs site
	uv run mkdocs build --strict

# Cleanup

clean:  ## Clean artifacts
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

help:  ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
