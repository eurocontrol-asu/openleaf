# AetherX Examples

This directory contains examples that demonstrate the experiments described in the AetherX research paper.

## Prerequisites

### Install Dependencies

```bash
uv sync --all-extras
```

### Data Requirements

All examples use the sample flight data in `data/`:
- `flight_data.json` - ADS-B trajectory (B738, ~1100 km flight)
- `met.npz` - Meteorological data (temperature, wind profiles)

### BADA License (Optional)

For BADA3 and BADA4 models, you need a license from EUROCONTROL:
1. Request a license at https://www.eurocontrol.int/model/bada
2. Place XML files in `data/` or set `BADA_PATH` environment variable

Without BADA data, use `--model ps` (Poll-Schumann) or `--model openap`.

## Examples Overview

| Example | Paper Section | Description |
|---------|---------------|-------------|
| `01_mass_estimation.py` | IV-B | Estimate aircraft mass from climb trajectory |
| `02_fuel_flow_estimation.py` | V (Timing) | Compute fuel flow along trajectory (vectorized) |
| `03_cost_index_estimation.py` | IV-C | Infer cost index from cruise observations |
| `04_vertical_profile_optimisation.py` | IV-D | Optimise altitude-speed profile for minimum fuel |

## Reproducing Paper Results

### Sensitivity Analysis (Paper Section IV-A)

See `experiments/scripts/sensitivity_analysis.py` - demonstrates CasADi autodiff.

**Expected Results (from paper Table I):**

| Model | dFF/dh (kg/s per kft) | dFF/dV (kg/s per 10kt) | dFF/dm (kg/s per tonne) |
|-------|----------------------|------------------------|-------------------------|
| PS | +0.070 | -0.076 | +0.039 |
| BADA3 | +0.065 | -0.081 | +0.042 |
| BADA4 | +0.073 | -0.072 | +0.037 |
| OpenAP | +0.068 | -0.079 | +0.040 |

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

**Expected Results (from paper Table II):**

| Model | Initial Mass (m0) | % of MTOW | Mean ESF | Solve Time |
|-------|-------------------|-----------|----------|------------|
| PS | 72,658 kg | 91.9% | 0.956 | 1.2 s |
| BADA3 | 73,504 kg | 93.0% | 0.968 | 1.1 s |
| BADA4 | 77,615 kg | 98.2% | 0.966 | 1.4 s |
| OpenAP | 74,892 kg | 94.8% | 0.961 | 1.3 s |

**Output:** `examples/mass_estimation.png`

**Note:** OpenAP is not supported for mass estimation because its `thrust_max` function requires ROCD as input, creating a circular dependency.

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

**Expected Results (from paper Table III):**

| Model | Estimated CI | Final Residual | Solve Time |
|-------|--------------|----------------|------------|
| PS | 14.1 kg/min | 5.84e-9 | 0.2 s |
| BADA3 | 13.3 kg/min | 5.36e-9 | 0.2 s |
| BADA4 | 19.7 kg/min | 9.35e-9 | 0.3 s |
| OpenAP | 15.8 kg/min | 7.12e-9 | 0.2 s |

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
| `--model ps\|bada3\|bada4\|openap` | Performance model to use |
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
export AETHERX_BADA3_PATH=/path/to/bada3/data
export AETHERX_BADA4_PATH=/path/to/bada4/data
```

### "Optimization failed: Maximum_Iterations_Exceeded"

The NLP solver couldn't converge. Try:
- Increasing solver iterations in the script
- Using a different initial mass value
- Simplifying the trajectory (fewer waypoints)

### "OpenAP not supported for mass estimation"

OpenAP's thrust model requires ROCD, which we're trying to predict. Use PS, BADA3, or BADA4 instead.

## Citation

If you use these examples in your research, please cite:

```bibtex
@article{aetherx2025,
  title={AetherX: A Unified Differentiable Framework for Aircraft Performance Modelling},
  author={Dalmau, Ramon and Jarry, Gabriel and Very, Philippe},
  year={2025}
}
```
