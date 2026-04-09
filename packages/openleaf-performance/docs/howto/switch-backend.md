# Switch backend

By default, `openleaf-performance` uses NumPy arrays. You can switch to CasADi,
JAX, or PyTorch by simply passing different array types as input. No
configuration or code changes are needed in the model layer.

## Use CasADi for trajectory optimization

CasADi enables symbolic automatic differentiation, required by NLP solvers
like IPOPT.

```python
import casadi as ca
import numpy as np
from openleaf.performance.core import TrajectoryData
from openleaf.performance.models.openap import OpenAPModel

model = OpenAPModel("A320")
n = 100  # number of trajectory points

# Create CasADi symbolic variables
altitude = ca.MX.sym("altitude", n)
tas = ca.MX.sym("tas", n)
vs = ca.MX.sym("vs", n)
mass = ca.MX.sym("mass", n)
accel = ca.MX.sym("accel", n)

# TrajectoryData works with CasADi MX types unchanged
traj = TrajectoryData({
    "altitude": altitude,
    "true_airspeed": tas,
    "vertical_speed": vs,
    "aircraft_mass": mass,
    "acceleration": accel,
})

perf = traj.with_aircraft(model)

# perf.fuel_flow is now a CasADi symbolic expression
# Use it in an NLP objective or constraint
total_fuel = ca.sum1(perf.fuel_flow)
```

All internal operations (`np.where`, `np.sqrt`, interpolation) automatically
dispatch to CasADi equivalents.

## Use JAX for GPU acceleration

```python
import jax.numpy as jnp
from openleaf.performance.core import TrajectoryData
from openleaf.performance.models.openap import OpenAPModel

model = OpenAPModel("A320")

traj = TrajectoryData({
    "altitude": jnp.array([10000.0, 10500.0]),
    "true_airspeed": jnp.array([230.0, 232.0]),
    "vertical_speed": jnp.array([5.0, 5.0]),
    "aircraft_mass": jnp.array([65000.0, 64900.0]),
    "acceleration": jnp.array([0.1, 0.1]),
})

perf = traj.with_aircraft(model)
print(perf.fuel_flow)  # JAX array on GPU
```

## Use PyTorch for ML integration

```python
import torch
from openleaf.performance.core import TrajectoryData
from openleaf.performance.models.openap import OpenAPModel

model = OpenAPModel("A320")

traj = TrajectoryData({
    "altitude": torch.tensor([10000.0, 10500.0]),
    "true_airspeed": torch.tensor([230.0, 232.0]),
    "vertical_speed": torch.tensor([5.0, 5.0]),
    "aircraft_mass": torch.tensor([65000.0, 64900.0]),
    "acceleration": torch.tensor([0.1, 0.1]),
})

perf = traj.with_aircraft(model)
print(perf.fuel_flow)  # PyTorch tensor
```

## How dispatch works internally

The dispatch layer in `openleaf.performance.numpy` checks input types at
runtime:

1. If any input is a CasADi type (`MX`, `DM`, `SX`), use CasADi operations
2. Otherwise, use `array-api-compat` to get the correct namespace (NumPy, JAX,
   or PyTorch)

This means you never need to change model code or configuration. The backend
is determined entirely by the types you pass in.

## Limitations

- **CasADi**: No boolean indexing or shape inspection at build time. All
  branching must use `np.where`.
- **JAX**: Requires `jax` and `jaxlib` installed. JIT compilation adds
  overhead on first call.
- **PyTorch**: Some operations may not support all dtypes (e.g., complex).
- **Interpolation**: `bspline` method always uses CasADi regardless of input
  type (SciPy has no B-spline interpn).

See [Multi-backend dispatch](../explanation/backends.md) for a deeper
explanation of the dispatch architecture.
