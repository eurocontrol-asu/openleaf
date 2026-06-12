# OpenLeaf Examples

This directory contains examples that demonstrate the experiments described in the OpenLeaf research paper.

## Prerequisites

### Install Dependencies

```bash
uv sync --all-groups --all-packages
```

### Data Requirements

All examples use the sample flight data in `data/`:
- `flight_data.json` - ADS-B trajectory (B738, ~1100 km flight)
- `met.npz` - Meteorological data (temperature, wind profiles)

### BADA License (Optional)

For BADA3 and BADA4 models, you need a license from EUROCONTROL:
1. Request a license at https://www.eurocontrol.int/model/bada
2. Point the `OPENLEAF_BADA3_PATH` / `OPENLEAF_BADA4_PATH` environment
   variables at your licensed data directories (see Troubleshooting below)

Without BADA data, use `--model ps` (all examples) or `--model openap`
(examples 01 and 02; examples 03 and 04 accept `ps|bada3|bada4` only).

## Examples Overview

| Example | Paper Section | Description |
|---------|---------------|-------------|
| `01_mass_estimation.py` | IV-B | Estimate aircraft mass from climb trajectory |
| `02_fuel_flow_estimation.py` | V (Timing) | Compute fuel flow along trajectory (vectorized) |
| `03_cost_index_estimation.py` | IV-C | Infer cost index from cruise observations |
| `04_vertical_profile_optimisation.py` | IV-D | Optimise altitude-speed profile for minimum fuel |

## Reproducing Paper Results

### Sensitivity Analysis (Paper Section IV-A)

Demonstrates CasADi autodiff. The reproduction scripts are not yet part of
the public release; see [`../experiments/README.md`](../experiments/README.md)
for the experiment description. BADA-derived reference outputs are not
redistributed in this repository.

### Mass Estimation (Paper Section IV-B)

Estimates takeoff mass from observed ADS-B climb data using the Total Energy Model.

```bash
# Poll-Schumann model (open-source)
uv run python examples/01_mass_estimation.py --model ps

# BADA3 (requires EUROCONTROL license)
uv run python examples/01_mass_estimation.py --model bada3

# BADA4 (requires EUROCONTROL license)
uv run python examples/01_mass_estimation.py --model bada4
```

**Reference results:** see [`../experiments/README.md`](../experiments/README.md)
(Table II); BADA-derived outputs are not redistributed in this repository.

**Output:** `examples/mass_estimation.png`

**Note:** `--model openap` runs via the implicit-NLP formulation, but OpenAP
is excluded from the paper's comparison: its `thrust_max` requires ROCD as an
input (circular with the TEM dynamics) and reflects a different thrust
premise, making the comparison uninformative.

### Cost Index Estimation (Paper Section IV-C)

Estimates airline cost index from cruise phase using optimality conditions (dq/dV = dq/dh = 0).

```bash
# Poll-Schumann model
uv run python examples/03_cost_index_estimation.py --model ps

# BADA3
uv run python examples/03_cost_index_estimation.py --model bada3

# BADA4
uv run python examples/03_cost_index_estimation.py --model bada4
```

**Reference results:** see [`../experiments/README.md`](../experiments/README.md)
(Table III); BADA-derived outputs are not redistributed in this repository.

### Vertical Profile Optimisation (Paper Section IV-D)

Optimises the vertical profile (altitude and speed) for minimum fuel consumption.

```bash
# Poll-Schumann model
uv run python examples/04_vertical_profile_optimisation.py --model ps

# BADA3
uv run python examples/04_vertical_profile_optimisation.py --model bada3 --mass 70000
```

**Note:** This is a complex NLP problem. The solver may require tuning for some trajectories.

**Output:** `examples/vertical_profile_optimisation.png`

## Command-Line Options

All examples support:

| Option | Description |
|--------|-------------|
| `--model ps\|bada3\|bada4\|openap` | Performance model (03/04: `ps\|bada3\|bada4`) |
| `--data PATH` | Path to flight_data.json |
| `--met PATH` | Path to met.npz |
| `--mass VALUE` | Aircraft mass in kg |
| `--quiet` | Suppress solver output |
| `--save-plot PATH` | Custom path for output plot |

## Model Comparison

| Model | License | Use Case | Supports CasADi |
|-------|---------|----------|-----------------|
| PS | Open (pycontrails) | Cruise analysis, contrails | Yes |
| OpenAP | Open | General estimation | Yes |
| BADA3 | EUROCONTROL | ATM simulation | Yes |
| BADA4 | EUROCONTROL | High-fidelity analysis | Yes |

## Troubleshooting

### "BADA data file not found"

Install BADA data from EUROCONTROL and set the path:
```bash
export OPENLEAF_BADA3_PATH=/path/to/bada3/data
export OPENLEAF_BADA4_PATH=/path/to/bada4/data
```

### "Optimization failed: Maximum_Iterations_Exceeded"

The NLP solver couldn't converge. Try:
- Increasing solver iterations in the script
- Using a different initial mass value
- Simplifying the trajectory (fewer waypoints)

### OpenAP mass estimates differ from other models

OpenAP's `thrust_max` requires ROCD (the quantity being predicted), so the
example solves an implicit formulation with a different thrust premise.
Prefer PS, BADA3, or BADA4 when comparing absolute mass estimates.

## Citation

If you use these examples in your research, please cite:

```bibtex
@inproceedings{dalmau2026openleaf,
  author    = {Dalmau, Ramon and Jarry, Gabriel and Very, Philippe},
  title     = {OpenLeaf: A Unified, Vectorized and Differentiable Framework for Aircraft Performance Modelling},
  booktitle = {US-Europe Air Transportation Research \& Development Symposium},
  address   = {Delft, Netherlands},
  year      = {2026}
}
```
