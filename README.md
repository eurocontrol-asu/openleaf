<p align="center">
  <img src="https://raw.githubusercontent.com/eurocontrol-asu/openleaf/main/docs/assets/logo.svg" alt="OpenLeaf Logo" width="180" />
</p>

<h1 align="center">OpenLeaf</h1>
<p align="center"><strong>Open-source aviation sustainability suite — performance modeling, emissions, and trajectory optimization.</strong></p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <a href="https://eurocontrol-asu.github.io/openleaf/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
</p>

---

## Philosophy

OpenLeaf provides **open-source tools for aviation sustainability research**. The suite is designed for researchers and aviation professionals who need accurate, reproducible, and differentiable aircraft performance models.

- **Modular Performance Models** — BADA 3/4, Poll-Schumann, OpenAP — all behind a unified interface with pluggable backends.
- **Multi-Backend Computation** — NumPy, JAX, CasADi, and PyTorch via the Array API standard. From quick analysis to GPU-accelerated optimization.
- **Emissions & Climate Impact** — Fuel burn, CO2, NOx, and contrail assessment integrated into the performance pipeline.
- **Differentiable by Design** — End-to-end differentiable models for trajectory optimization with automatic differentiation.

## Packages

| Package | Description | Quality |
|---|---|---|
| [openleaf-performance](packages/openleaf-performance/) | Modular, differentiable aircraft performance modeling framework | [![cov](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-performance/coverage.json)](https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml) |
| [openleaf-core](packages/openleaf-core/) | Shared types and interfaces for the OpenLeaf suite | — |

## Quick Start

```bash
# Clone and install
git clone https://github.com/eurocontrol-asu/openleaf.git
cd openleaf
uv sync --all-groups

# Run all tests
make test-all

# Lint + type check
make check
```

## Architecture

```mermaid
graph TD
    classDef perf fill:#B3D4F0,stroke:#023961
    classDef core fill:#C4DFF5,stroke:#15588F

    PERF["openleaf-performance<br/>Aircraft performance"]:::perf --> CORE
    CORE["openleaf-core<br/>Shared types & interfaces"]:::core
```

## Development

| Command | Description |
|---|---|
| `make test-all` | Run tests for all packages |
| `make test-performance` | Run tests for openleaf-performance |
| `make test-core` | Run tests for openleaf-core |
| `make lint` | Ruff linter + formatter check |
| `make type-check` | Run mypy per package |
| `make check` | Lint + type-check + tests |
| `make docs-serve` | Preview documentation |

## License

[EUPL-1.2](LICENSE) — © 2026 EUROCONTROL
