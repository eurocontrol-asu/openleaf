<div class="hero" markdown>

<p align="center">
  <img src="assets/logo.svg" alt="OpenLeaf Logo" width="180" />
</p>

<h1 align="center">OpenLeaf</h1>
<p align="center"><strong>Open-source aviation sustainability suite — aircraft performance modeling and trajectory optimization.</strong></p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="EUPL-1.2">
</p>

</div>

---

<div class="grid cards" markdown>

-   :material-airplane:{ .lg .middle } **Performance Modeling**

    ---

    Modular, differentiable aircraft performance with BADA, Poll-Schumann, OpenAP and more.

    [:octicons-arrow-right-24: openleaf-performance](packages/index.md)

-   :material-cog-outline:{ .lg .middle } **Multi-Backend**

    ---

    NumPy, JAX, CasADi, and PyTorch backends via the Array API standard.

    [:octicons-arrow-right-24: Architecture](openleaf-performance/explanation/architecture/)

-   :material-chart-line:{ .lg .middle } **Fuel Burn & Beyond**

    ---

    Accurate fuel-flow modeling today; CO2, NOx and contrail assessment
    are on the project roadmap.

    [:octicons-arrow-right-24: Models overview](openleaf-performance/explanation/models/)

-   :material-open-source-initiative:{ .lg .middle } **Open Source**

    ---

    EUPL-1.2 licensed. Built for researchers and aviation professionals.

    [:octicons-arrow-right-24: Contributing](contributing.md)

</div>

---

## Quick Install

```bash
uv add openleaf-performance
```

## Quick Example

```python
import numpy as np
from openleaf.performance.core import ModelFactory, TrajectoryData

# Load an aircraft model (OpenAP requires no configuration)
model = ModelFactory.load("A320", model_type="openap")

# Build a trajectory
traj = TrajectoryData(
    {
        "altitude": np.array([10000.0, 10500.0]),
        "true_airspeed": np.array([230.0, 232.0]),
        "vertical_speed": np.array([5.0, 5.0]),
        "aircraft_mass": np.array([65000.0, 64900.0]),
        "acceleration": np.array([0.1, 0.1]),
    }
)

# Attach model and compute performance
perf = traj.with_aircraft(model)
print(perf.drag)  # Aerodynamic drag [N]
print(perf.thrust)  # Required thrust [N]
print(perf.fuel_flow)  # Fuel flow rate [kg/s]
```

Head over to the [Packages](packages/index.md) section to explore what's available.
