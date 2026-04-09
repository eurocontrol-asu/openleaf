---
hide:
  - navigation
  - toc
---

# openleaf

**Open-source aviation sustainability suite** — performance modeling, emissions analysis, and trajectory optimization.

<p align="center">
  <a href="https://github.com/ramondalmau/openleaf/actions/workflows/axm-quality.yml"><img src="https://github.com/ramondalmau/openleaf/actions/workflows/axm-quality.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ramondalmau/openleaf/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ramondalmau/openleaf/gh-pages/badges/axm-audit.json" alt="axm-audit"></a>
  <a href="https://github.com/ramondalmau/openleaf/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ramondalmau/openleaf/gh-pages/badges/axm-init.json" alt="axm-init"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="EUPL-1.2">
</p>

---

<div class="grid cards" markdown>

-   :material-airplane:{ .lg .middle } **Performance Modeling**

    ---

    Modular, differentiable aircraft performance with BADA, Poll-Schumann, OpenAP and more.

    [:octicons-arrow-right-24: openleaf-performance](packages/index.md)

-   :material-cog-outline:{ .lg .middle } **Multi-Backend**

    ---

    NumPy, JAX, CasADi, and PyTorch backends via the Array API standard.

-   :material-chart-line:{ .lg .middle } **Emissions & Contrails**

    ---

    Fuel burn, CO2, NOx, and contrail climate impact assessment.

-   :material-open-source-initiative:{ .lg .middle } **Open Source**

    ---

    EUPL-1.2 licensed. Built for researchers and aviation professionals.

</div>

---

## Quick Install

```bash
uv add openleaf-performance
```

## Getting Started

```python
from openleaf.performance import PerformanceMixin
```

Head over to the [Packages](packages/index.md) section to explore what's available.
