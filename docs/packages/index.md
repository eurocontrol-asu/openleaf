---
hide:
  - toc
---

# Packages

<div class="grid cards" markdown>

-   :material-airplane:{ .lg .middle } **openleaf-performance**

    ---

    Modular, differentiable aircraft performance modeling with BADA, Poll-Schumann, OpenAP and more.

    [![cov](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-performance/coverage.json)](https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml)

    [:octicons-arrow-right-24: Getting Started](../openleaf-performance/)

-   :material-cube-outline:{ .lg .middle } **openleaf-core**

    ---

    Shared types, protocols, and interfaces that form the foundation of the
    OpenLeaf suite. Installed automatically as a dependency of
    `openleaf-performance`.

</div>

## Architecture

```mermaid
graph TD
    classDef perf fill:#B3D4F0,stroke:#023961
    classDef core fill:#C4DFF5,stroke:#15588F

    PERF["openleaf-performance<br/>Aircraft performance"]:::perf --> CORE
    CORE["openleaf-core<br/>Shared types & interfaces"]:::core
```
