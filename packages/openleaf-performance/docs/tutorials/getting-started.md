# Getting Started

This tutorial walks you through loading an aircraft model and computing
performance quantities (drag, thrust, fuel flow) for a flight trajectory.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
uv add openleaf-performance
```

Or with pip:

```bash
pip install openleaf-performance
```

## Step 1: Configure data paths (BADA only)

OpenAP and Poll-Schumann models work out of the box. BADA3 and BADA4 require
you to point to the data directories provided by EUROCONTROL:

```python
import openleaf.performance

# Set paths once -- they are persisted in ~/.config/openleaf/settings.toml
openleaf.performance.set_bada4_path("/path/to/bada4/data")
openleaf.performance.set_bada3_path("/path/to/bada3/data")

# Verify configuration
openleaf.performance.show_config()
```

Skip this step if you only use OpenAP or Poll-Schumann.

## Step 2: Load an aircraft model

The `ModelFactory` provides a single entry point for all model types:

```python
from openleaf.performance.core import ModelFactory

# Auto-detect best available model (tries BADA4 > BADA3 > OpenAP > PS)
model = ModelFactory.load("A320")

# Or request a specific model type
model = ModelFactory.load("A320", model_type="openap")
model = ModelFactory.load("B738", model_type="bada4")
```

You can also instantiate models directly:

```python
from openleaf.performance.models.openap import OpenAPModel

model = OpenAPModel("A320")
print(model.data)  # Aircraft data: model, WTC, engine type
print(model.limits)  # LimitsData: MTOM, VMO, MMO, HMO
```

## Step 3: Build a trajectory

`TrajectoryData` is a lazy container that holds flight state arrays and
automatically computes derived quantities (ISA atmosphere, Mach number,
calibrated airspeed, energy share factor) on first access.

```python
import numpy as np
from openleaf.performance.core import TrajectoryData

data = {
    "altitude": np.array([10000.0, 10500.0, 11000.0]),  # Pressure altitude Hp [m]
    "true_airspeed": np.array([230.0, 232.0, 234.0]),  # TAS [m/s]
    "vertical_speed": np.array([5.0, 5.0, 0.0]),  # dHp/dt [m/s]
    "aircraft_mass": np.array([65000.0, 64900.0, 64800.0]),  # [kg]
    "acceleration": np.array([0.1, 0.1, 0.0]),  # dV/dt [m/s^2]
}

traj = TrajectoryData(data)

# Derived quantities are computed lazily
print(traj.mach_number)  # Mach from TAS and ISA temperature
print(traj.calibrated_airspeed)  # CAS [m/s]
print(traj.air_temperature)  # ISA temperature [K]
print(traj.air_pressure)  # ISA pressure [Pa]
```

You can provide either `acceleration` (dV/dt) or `f` (energy share factor).
The missing one is derived automatically.

## Step 4: Compute aircraft performance

Attach the aircraft model to the trajectory to compute drag, thrust, and fuel
flow:

```python
perf = traj.with_aircraft(model)

# Performance quantities (computed lazily, cached)
print(perf.drag)  # Aerodynamic drag [N]
print(perf.thrust)  # Required thrust [N]
print(perf.fuel_flow)  # Fuel flow [kg/s]
print(perf.drag_coef)  # CD [-]
```

The `with_aircraft` method also accepts a string shorthand:

```python
perf = traj.with_aircraft("OpenAP", "A320")
perf = traj.with_aircraft("BADA4", "B738")
```

## Step 5: Check the flight envelope

By default, envelope violations are ignored. You can change this behavior:

```python
from openleaf.performance.core import EnvelopeMode

# Warn on violations (logs a warning)
perf = traj.with_aircraft(model, envelope_mode=EnvelopeMode.WARN)

# Raise on violations (raises EnvelopeViolationError)
perf = traj.with_aircraft(model, envelope_mode=EnvelopeMode.RAISE)
```

Envelope checks cover: overspeed (CAS and Mach), underspeed (stall), ceiling
altitude, and mass bounds.

## Next steps

- [Architecture](../explanation/architecture.md) -- how the layers fit together
- [Performance models](../explanation/models.md) -- BADA3 vs BADA4 vs OpenAP vs PS
- [Multi-backend dispatch](../explanation/backends.md) -- NumPy, CasADi, JAX, PyTorch
- [Switch backend](../howto/switch-backend.md) -- use CasADi for optimization
