# Experiments

This directory documents the experiments from the OpenLeaf paper and carries
their committed results (`results/*.json`).

> **Note:** the reproduction scripts referenced below are not yet included in
> the public release; they will be published separately. The committed
> `results/*.json` files are the authoritative reference values.

## Reproduction workflow

Run experiments in order. Each script writes a JSON file; the figure script reads all of them.

```bash
# 1. Sensitivity analysis (Table I / Fig. sensitivity)
uv run python experiments/scripts/run_sensitivity_analysis.py

# 2. Mass estimation (Table II / Fig. mass)
uv run python experiments/scripts/run_mass_estimation.py

# 3. Cost index (Table III / Fig. cost_index)
uv run python experiments/scripts/run_cost_index.py

# 4. Optimal altitude (Table IV / Fig. optimal_altitude)
uv run python experiments/scripts/run_optimal_altitude.py

# 5. Vectorization benchmark (Fig. vectorization)
uv run python experiments/scripts/run_vectorization_benchmark.py

# 6. Generate all paper figures (reads results/*.json)
uv run python experiments/scripts/generate_paper_figures.py
```

Figures are generated from `results/*.json` by the figure script.

---

## Per-experiment details

### Sensitivity Analysis — Table I

```bash
uv run python experiments/scripts/run_sensitivity_analysis.py
```

Reference condition: FL350 (10,668 m), 240 m/s TAS, 65 t, Mach ≈ 0.81.

**Expected output (all 4 models):**

```
Table I summary (FL350, 240 m/s TAS, 65 t):
Model         ∂FF/∂h       ∂FF/∂V       ∂FF/∂m
          (kg/s/kft)  (kg/s/10kt)     (kg/s/t)
PS            -0.000       +0.042       +0.009
BADA3         -0.014       +0.021       +0.005
BADA4         +0.001       +0.070       +0.008
OpenAP        -0.010       +0.012       +0.008
```

**Results saved to:** `experiments/results/sensitivity_results.json`

---

### Mass Estimation — Table II

```bash
uv run python experiments/scripts/run_mass_estimation.py
```

**Notes:**
- Analysis restricted to FL247–FL310 (6 samples) to avoid large ADS-B time gaps
  and the ADS-B position/velocity phase-shift artefact visible above FL310.
- OpenAP is excluded: its thrust_max requires ROCD as input, creating a circular
  dependency with TEM dynamics that produces a fundamentally different thrust
  premise (typical operational vs. certified maximum), making comparison uninformative.

**Expected output (PS, BADA3, BADA4):**

```
Model    m0 (kg)    % MTOW    Mean ESF    Time (s)
PS        77,366      97.9       0.896         3.6
BADA3     77,561      99.1       0.912         0.1
BADA4     79,016     100.0       0.879         0.2
```

**Results saved to:** `experiments/results/mass_estimation_results.json`

---

### Cost Index Estimation — Table III

```bash
uv run python experiments/scripts/run_cost_index.py
```

**Expected output (PS, BADA3, BADA4, OpenAP):**

```
Model    CI (kg/min)       Residual    Time (s)
PS              14.1       5.84e-09         2.0
BADA3           13.3       5.36e-09         1.6
BADA4           19.7       9.35e-09         1.2
OPENAP           6.1       2.80e-08         1.0
```

**Results saved to:** `experiments/results/cost_index_results.json`

---

### Optimal Cruise Altitude — Table IV

```bash
uv run python experiments/scripts/run_optimal_altitude.py
```

Search range: FL250–FL410 (B738 operational envelope). Values marked ≥FL410
indicate that the mathematical optimum lies above the aircraft's maximum
certified altitude.

**Expected output (all 4 models, masses 50/60/70 t):**

```
Table IV — Optimal altitude at Mach 0.78:
Mass (t)        PS   BADA3   BADA4  OpenAP
50          ≥FL410  ≥FL410  ≥FL410  ≥FL410
60           FL388  ≥FL410   FL397  ≥FL410
70           FL360  ≥FL410   FL364   FL395
```

BADA3 shows ≥FL410 at all masses because its parabolic drag polar has no
wave-drag term, so ∂FF/∂h remains negative throughout the operational
envelope — the mathematical optimum is above FL450.

**Results saved to:** `experiments/results/optimal_altitude_results.json`

---

### Vectorization Benchmark

```bash
uv run python experiments/scripts/run_vectorization_benchmark.py
```

**Expected output (indicative; hardware-dependent):**

```
SUMMARY: Average speedup across batch sizes
isa         :  274x faster with vectorization
drag        :  160x faster with vectorization
fuel_flow   :  211x faster with vectorization
```

**Results saved to:** `experiments/results/vectorization_benchmark.json`

---

## BADA licence requirements

| Experiment       | PS (open) | OpenAP (open) | BADA3 | BADA4 |
|------------------|-----------|---------------|-------|-------|
| Sensitivity      | ✓         | ✓             | ✓     | ✓     |
| Mass Estimation  | ✓         | ✓             | ✓     | ✓     |
| Cost Index       | ✓         | ✓             | ✓     | ✓     |
| Optimal Altitude | ✓         | ✓             | ✓     | ✓     |
| Vectorization    | ✓         | ✓             | —     | —     |

Request a BADA licence at <https://www.eurocontrol.int/model/bada>.
