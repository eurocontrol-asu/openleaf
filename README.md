<h1 align="center">openleaf</h1>
<p align="center"><strong>Open-source aviation sustainability suite — performance modeling, emissions, and trajectory optimization.</strong></p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml"><img src="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/axm-audit.json" alt="axm-audit"></a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/axm-init.json" alt="axm-init"></a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/coverage.json" alt="Coverage"></a>
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
| [openleaf-performance](packages/openleaf-performance/) | Modular, differentiable aircraft performance modeling framework | [![audit](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-performance/axm-audit.json)](https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml) [![cov](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-performance/coverage.json)](https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml) |
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
    classDef perf fill:#42A5F5,stroke:#1E88E5
    classDef core fill:#78909C,stroke:#546E7A

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
