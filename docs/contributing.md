# Contributing to openleaf

## Development Setup

```bash
git clone https://github.com/eurocontrol-asu/openleaf.git
cd openleaf
uv sync --all-groups
uv run pre-commit install
```

## Making Changes

1. Create a branch: `git checkout -b feat/my-change`
2. Make changes in the relevant package under `packages/`
3. Run tests: `make test-all`
4. Run lint: `make lint`
5. Run type checks: `make type-check`
6. Commit with conventional commits: `feat(pkg): description`
7. Open a pull request

## Available Commands

| Command | Description |
|---|---|
| `make test-all` | Run tests for all packages |
| `make test-core` | Run tests for openleaf-core |
| `make test-performance` | Run tests for openleaf-performance |
| `make lint` | Ruff linter + formatter check |
| `make format` | Auto-format code |
| `make type-check` | Run mypy per package |
| `make check` | Lint + type-check + tests |
| `make docs-serve` | Preview docs locally |
| `make docs-build` | Build docs site |

## Adding a Package

1. Create `packages/my-pkg/` with `pyproject.toml`, `src/`, `tests/`
2. UV auto-discovers members via `packages/*` glob
3. Run `uv sync` to update the lockfile
4. Add the package to the `nav` in `mkdocs.yml` and `docs/packages/index.md`
