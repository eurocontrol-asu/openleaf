---
hide:
  - navigation
  - toc
---

# openleaf-performance

<p align="center">
  <strong>Aircraft performance models for trajectory optimization</strong>
</p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml">
    <img src="https://github.com/eurocontrol-asu/openleaf/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-performance/axm-init.json" alt="axm-init" />
  </a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-performance/axm-audit.json" alt="axm-audit" />
  </a>
  <a href="https://github.com/eurocontrol-asu/openleaf/actions/workflows/axm-quality.yml">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/openleaf/gh-pages/badges/openleaf-performance/coverage.json" alt="Coverage" />
  </a>
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+" />
</p>

---

`openleaf-performance` provides aircraft performance models (BADA3, BADA4, OpenAP,
Poll-Schumann) behind a unified interface. All models support NumPy arrays and
CasADi symbolic types, making them suitable for both analysis and trajectory
optimization.

## Installation

```bash
uv add openleaf-performance
```

## Quick example

```python
import numpy as np
from openleaf.performance.core import ModelFactory, TrajectoryData

# Load an aircraft model (OpenAP requires no configuration)
model = ModelFactory.load("A320", model_type="openap")

# Build a trajectory
traj = TrajectoryData({
    "altitude": np.array([10000.0, 10500.0]),       # Pressure altitude [m]
    "true_airspeed": np.array([230.0, 232.0]),       # TAS [m/s]
    "vertical_speed": np.array([5.0, 5.0]),          # dHp/dt [m/s]
    "aircraft_mass": np.array([65000.0, 64900.0]),   # [kg]
    "acceleration": np.array([0.1, 0.1]),            # dV/dt [m/s^2]
})

# Attach model and compute performance
perf = traj.with_aircraft(model)
print(perf.drag)       # Aerodynamic drag [N]
print(perf.thrust)     # Required thrust [N]
print(perf.fuel_flow)  # Fuel flow rate [kg/s]
```

## Features

- **4 performance models** -- BADA3, BADA4, OpenAP, and Poll-Schumann
- **Multi-backend dispatch** -- NumPy, JAX, CasADi, and PyTorch via a unified array API
- **Lazy trajectory container** -- automatic ISA, airspeed conversions, energy share factor
- **Flight envelope checking** -- configurable violation modes (ignore, warn, raise)
- **Unified factory** -- load any model by ICAO code with `ModelFactory.load()`

---

<div style="text-align: center; margin: 2rem 0;">
  <a href="tutorials/getting-started/" class="md-button md-button--primary">Get Started</a>
  <a href="explanation/architecture/" class="md-button">Architecture</a>
</div>
