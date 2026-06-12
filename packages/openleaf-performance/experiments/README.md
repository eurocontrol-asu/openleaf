# Experiments

This directory documents the experiments from the OpenLeaf paper and carries
their committed results (`results/*.json`).

> **Note:** the reproduction scripts referenced below are not yet included in
> the public release; they will be published separately. Result files are
> generated locally into `results/` and are NOT committed: outputs derived
> from BADA3/BADA4 are subject to the BADA licence and must not be
> redistributed. The expected-output listings below were removed from the
> public repository for the same reason.

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

_Reference output withheld from the public repository (BADA-derived)._

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

_Reference output withheld from the public repository (BADA-derived)._

**Results saved to:** `experiments/results/mass_estimation_results.json`

---

### Cost Index Estimation — Table III

```bash
uv run python experiments/scripts/run_cost_index.py
```

_Reference output withheld from the public repository (BADA-derived)._

**Results saved to:** `experiments/results/cost_index_results.json`

---

### Optimal Cruise Altitude — Table IV

```bash
uv run python experiments/scripts/run_optimal_altitude.py
```

Search range: FL250–FL410 (B738 operational envelope). Values marked ≥FL410
indicate that the mathematical optimum lies above the aircraft's maximum
certified altitude.

_Reference output withheld from the public repository (BADA-derived)._

BADA3 shows ≥FL410 at all masses because its parabolic drag polar has no
wave-drag term, so ∂FF/∂h remains negative throughout the operational
envelope — the mathematical optimum is above FL450.

**Results saved to:** `experiments/results/optimal_altitude_results.json`

---

### Vectorization Benchmark

```bash
uv run python experiments/scripts/run_vectorization_benchmark.py
```

_Reference output withheld from the public repository (BADA-derived)._

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
