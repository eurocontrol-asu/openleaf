<p align="center">
  <strong>A modular, differentiable aircraft performance modeling framework</strong>
</p>

<!-- Canonical Badge Row: Status → Stack → Tooling → Info -->
<p align="center">
  <!-- Status -->
  <a href="https://github.com/ramondalmau/openleaf/actions/workflows/ci.yml"><img src="https://github.com/ramondalmau/openleaf/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://coveralls.io/github/ramondalmau/openleaf?branch=main"><img src="https://coveralls.io/repos/github/ramondalmau/openleaf/badge.svg?branch=main" alt="Coverage"></a>
  <!-- Stack -->
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/casadi-symbolic-orange" alt="CasADi">
  <img src="https://img.shields.io/badge/pytorch-supported-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/jax-supported-green" alt="JAX">
  <img src="https://img.shields.io/badge/typed-strict-blue" alt="Typed">
  <!-- Tooling -->
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <!-- Info -->
  <a href="https://pypi.org/project/openleaf-performance/"><img src="https://img.shields.io/pypi/v/openleaf-performance" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="License">
  <a href="https://ramondalmau.github.io/openleaf/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
</p>

---

A modular, differentiable aircraft performance modeling framework supporting BADA3, BADA4, Poll-Schumann, OpenAP, and more. Designed for trajectory optimization using CasADi symbolic computation.

## Features

- **Unified ICAO Interface**: Initialize any aircraft model with just an ICAO code (e.g., `PSModel("A320")`)
- **Multi-Model Support**: BADA3, BADA4, Poll-Schumann, and OpenAP aircraft performance models
- **Multi-Backend**: Write once, run anywhere with NumPy, PyTorch, JAX, CuPy, or CasADi arrays via the [Array API Standard](https://data-apis.org/array-api/latest/). All model functions (`drag_coef`, `thrust_max`, `thrust_min`, `fuel_flow`, `fuel_flow_min`) are tested across all backends and aircraft types (jet, turboprop, piston)
- **CasADi Integration**: Full symbolic differentiation support for trajectory optimization
- **Flight Envelope Limits**: VMO, MMO, HMO, stall speeds, buffet onset, and more
- **Modular Architecture**: Clean separation of aircraft data, engine models, and aerodynamic models

## Installation

### From PyPI

```bash
uv pip install openleaf-performance
```

### From Source

```bash
git clone https://github.com/ramondalmau/openleaf.git
cd openleaf
uv sync --all-groups
```

### Optional Dependencies

```bash
# Install with OpenAP support
uv pip install openleaf-performance[openap]

# Install with PS model support (requires pycontrails)
uv pip install openleaf-performance[ps]

# Install everything
uv pip install openleaf-performance[all]
```

## Quick Start

### Unified ICAO-Based Initialization

All models now support initialization with just an ICAO aircraft type code:

```python
from openleaf.performance.models.bada3 import BADA3Model
from openleaf.performance.models.bada4 import BADA4Model
from openleaf.performance.models.openap import OpenAPModel
from openleaf.performance.models.ps import PSModel

# Initialize any model with ICAO code
model = OpenAPModel("A320")      # Works out of the box
model = PSModel("B738")          # Requires pycontrails
model = BADA3Model("A320")       # Requires configuration (see below)
model = BADA4Model("A320")       # Requires configuration (see below)

# Access aircraft properties
print(f"Aircraft: {model.model}")
print(f"MTOM: {model.mtom} kg")
print(f"MMO: {model.mmo}")
```

### Configuring BADA Paths

BADA3 and BADA4 require proprietary data from [EUROCONTROL](https://www.eurocontrol.int/model/bada). Once obtained, configure the paths using one of the three methods below (in order of priority):

**1. Environment variables** (highest priority)

```bash
export OPENLEAF_BADA3_PATH="/path/to/bada3/data"
export OPENLEAF_BADA4_PATH="/path/to/bada4/data"
```

**2. Config file** `~/.config/openleaf/settings.toml`

```toml
[bada3]
path = "/path/to/bada3/data"
# synonyms = "/path/to/SYNONYM.NEW"  # Optional

[bada4]
path = "/path/to/bada4/data"
# synonyms = "/path/to/SYNONYM.BD4"  # Optional
```

**3. Python API** (persists to settings.toml)

```python
import openleaf.performance

openleaf.performance.set_bada3_path("/path/to/bada3/data")
openleaf.performance.set_bada4_path("/path/to/bada4/data")

# View current configuration
openleaf.performance.show_config()
```

Once configured, ICAO codes work directly:

```python
from openleaf.performance.models.bada3 import BADA3Model
model = BADA3Model("A320")
```

### Using with CasADi

The power of OpenLeaf: all functions work seamlessly with CasADi symbolic types!

```python
import casadi as ca
import openleaf.performance.numpy as anp
from openleaf.performance.models.ps import PSModel
from openleaf.performance.physics import isa

# Initialize model
ps_model = PSModel("B738")

# Create symbolic variables
altitude = ca.SX.sym("h")  # meters
mach = ca.SX.sym("M")
aircraft_mass = ca.SX.sym("m")

# OpenLeaf ISA functions work with CasADi!
air_temperature = isa.air_temperature(altitude)
air_pressure = isa.air_pressure(altitude)

# PSModel methods work with CasADi!
c_lift = ps_model.lift_coefficient(aircraft_mass, air_pressure, mach)
c_drag = ps_model.aerodynamics.drag_coef_with_state(
    cl=c_lift, mach=mach,
    air_temperature=air_temperature,
    air_pressure=air_pressure,
)
thrust = ps_model.compute_thrust_required(aircraft_mass, c_lift, c_drag)
fuel_flow = ps_model.engine.fuel_flow({
    "thrust": thrust,
    "mach_number": mach,
    "air_temperature": air_temperature,
    "air_pressure": air_pressure,
    "altitude": altitude,
})

# Compute gradients for optimization
grad = ca.gradient(fuel_flow, altitude)
```

### Basic Performance Calculations

```python
from openleaf.performance.models.ps import PSModel
from openleaf.performance.physics import isa
import numpy as np

model = PSModel("A320")

# Flight conditions (FL350)
h = 10668.0       # altitude [m]
mach = 0.78       # Mach number
m = 65000.0       # aircraft mass [kg]

# Atmospheric state using OpenLeaf ISA
T = isa.air_temperature(h)  # [K]
p = isa.air_pressure(h)     # [Pa]

# Calculate lift and drag coefficients
cl = model.lift_coefficient(m, p, mach)
cd = model.aerodynamics.drag_coef_with_state(cl=cl, mach=mach, air_temperature=T, air_pressure=p)

# Thrust required for level flight
thrust = model.compute_thrust_required(m, cl, cd)
print(f"Thrust required: {thrust/1000:.1f} kN")
print(f"L/D ratio: {cl/cd:.1f}")

# Calculate fuel flow
ff = model.engine.fuel_flow({
    "thrust": thrust,
    "mach_number": mach,
    "air_temperature": T,
    "air_pressure": p,
    "altitude": h,
})
print(f"Fuel flow: {ff*3600:.0f} kg/h")
```

### Flight Envelope Limits

```python
from openleaf.performance.models.ps import PSModel

model = PSModel("A320")

# Speed limits (PS model uses Mach limits for cruise, VMO not applicable)
print(f"MMO: {model.mmo}")
print(f"HMO: {model.hmo} ft")

# Mass limits
print(f"MTOM: {model.mtom} kg")
print(f"OEM: {model.oem} kg")
```

## Architecture

```
openleaf/
├── core/
│   ├── perfos.py          # Abstract base classes and interfaces
│   ├── factory.py         # Model factory (ModelType → AircraftModel)
│   ├── trajectory.py      # AircraftTrajectoryData container
│   ├── types.py           # StandardFlightState and shared types
│   └── validation.py      # Input validation utilities
├── models/
│   ├── bada3/             # BADA3 implementation
│   ├── bada4/             # BADA4 implementation
│   ├── openap/            # OpenAP wrapper
│   └── ps/                # Poll-Schumann model
├── numpy/                 # Multi-backend NumPy operations (Array API)
│   ├── backends.py        # Backend detection (NumPy/PyTorch/JAX/CuPy/CasADi)
│   ├── determine_type.py  # Runtime type dispatch
│   └── ...                # Arithmetic, trig, conditionals, etc.
├── physics/
│   ├── aero.py            # Aerodynamic calculations (tas_to_mach, etc.)
│   ├── esf.py             # Energy share factors
│   ├── isa.py             # International Standard Atmosphere
│   ├── tem.py             # Total Energy Model
│   └── units.py           # Unit conversions and constants
├── config.py              # Configuration for BADA paths
└── utils/
    └── types.py           # Type definitions (ArrayScalarLike, KeyedData)
```

## Data Requirements

### BADA3/BADA4

BADA is proprietary and must be obtained from [EUROCONTROL](https://www.eurocontrol.int/model/bada). See [Configuring BADA Paths](#configuring-bada-paths) for setup instructions.

### OpenAP

OpenAP data is open-source and bundled with the `openap` package.

### Poll-Schumann

Requires `pycontrails` for aircraft parameters:

```bash
uv pip install pycontrails
```

## Examples

See `examples/` for complete demonstrations reproducing the results in the paper:

| Example | Description |
|---------|-------------|
| `01_mass_estimation.py` | Estimate aircraft mass from ADS-B climb data |
| `02_fuel_flow_estimation.py` | Compute fuel flow along a trajectory |
| `03_cost_index_estimation.py` | Estimate airline cost index from cruise data |
| `04_vertical_profile_optimisation.py` | Optimise altitude-speed profile for minimum fuel |

```bash
uv run python examples/01_mass_estimation.py --model ps
uv run python examples/02_fuel_flow_estimation.py --model bada4
uv run python examples/03_cost_index_estimation.py --model openap
uv run python examples/04_vertical_profile_optimisation.py --model ps
```

## Development

```bash
git clone https://github.com/ramondalmau/openleaf.git
cd openleaf
uv sync --all-groups    # Install all dependencies
uv run pytest           # Run tests
uv run ruff check       # Lint
uv run mypy src/        # Type check
```

## Multi-Backend Support

OpenLeaf uses `openleaf.performance.numpy` which provides NumPy-compatible operations that work with multiple array backends via the [Array API Standard](https://data-apis.org/array-api/latest/):

```python
import openleaf.performance.numpy as anp
import numpy as np
import torch  # Optional

# Works with NumPy arrays
x_np = np.array([1.0, 4.0, 9.0])
result_np = anp.sqrt(x_np)

# Works with PyTorch tensors (automatic dispatch)
x_torch = torch.tensor([1.0, 4.0, 9.0])
result_torch = anp.sqrt(x_torch)  # Returns torch.Tensor

# Works with CasADi symbols for optimization
import casadi as ca
x_sym = ca.SX.sym("x")
result_sym = anp.sqrt(x_sym)  # Returns CasADi expression

# All NumPy-like operations work across backends
result = anp.where(condition, value_if_true, value_if_false)
result = anp.maximum(x1, x2)
result = anp.concatenate([arr1, arr2])
```

Supported backends:
- **NumPy** (default)
- **PyTorch** (GPU acceleration, autodiff)
- **JAX** (JIT compilation, autodiff)
- **CuPy** (NVIDIA GPU)
- **CasADi** (symbolic computation, optimization)

All model functions (`drag_coef`, `thrust_max`, `thrust_min`, `fuel_flow`, `fuel_flow_min`) are tested across all backends for jet, turboprop, and piston aircraft types. This enables the same model code to work for simulation (NumPy/PyTorch/JAX) and optimization (CasADi).

## Contributing

Contributions are welcome! See the development section above for setup instructions.

## License

European Union Public Licence v. 1.2 (EUPL-1.2) — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@inproceedings{dalmau2026aetherx,
  author    = {Dalmau, Ramon and Jarry, Gabriel and Very, Philippe},
  title     = {OpenLeaf: A Unified, Vectorized and Differentiable Framework for Aircraft Performance Modelling},
  booktitle = {Proceedings of the Air Traffic and Radar Studies (ATRDS) 2026},
  year      = {2026}
}
```

## Acknowledgments

- EUROCONTROL for the BADA methodology
- The CasADi team for symbolic computation
- The OpenAP project for open-source aircraft data
- The pycontrails team for Poll-Schumann parameters
