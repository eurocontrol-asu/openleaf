# openleaf-core

<p align="center">
  <strong>Shared types and interfaces for the OpenLeaf suite</strong>
</p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-core/coverage.json" alt="Coverage" /></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+" />
</p>

---

`openleaf-core` defines the common types, protocols, and interfaces shared across all OpenLeaf packages. It serves as the foundation that other packages (like `openleaf-performance`) build upon.

## Installation

```bash
uv add openleaf-core
```

## Role in the suite

```mermaid
graph TD
    classDef perf fill:#B3D4F0,stroke:#023961
    classDef core fill:#C4DFF5,stroke:#15588F

    PERF["openleaf-performance<br/>Aircraft performance"]:::perf --> CORE
    CORE["openleaf-core<br/>Shared types & interfaces"]:::core
```

All OpenLeaf packages depend on `openleaf-core` for shared definitions. If you are using `openleaf-performance`, `openleaf-core` is installed automatically as a dependency.
