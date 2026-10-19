# Performance models

`openleaf-performance` implements four aircraft performance models. All share
the same `AircraftModel` interface (engine, aerodynamics, limits subsystems)
but differ in data source, fidelity, and licensing.

## Comparison

| | BADA3 | BADA4 | OpenAP | Poll-Schumann |
|---|---|---|---|---|
| **Source** | EUROCONTROL | EUROCONTROL | Open data (TU Delft) | pycontrails (Contrails.org) |
| **License** | Proprietary | Proprietary | Open | Open |
| **Data format** | `.OPF` text files | XML files | Python library | Python library |
| **Aircraft coverage** | ~400 types | ~150 types | ~100 types | ~80 types |
| **Fidelity** | Medium | High | Medium | Medium (cruise focus) |
| **Configurations** | 5 (CR, IC, TO, AP, LD) | Config-dependent (HLid + LG) | Flap angle + gear | Clean only |
| **Flight phases** | All | All | All | Cruise |

## BADA3

Base of Aircraft Data version 3, the most widely used model in ATM research.

- **Data**: OPF files parsed via regex, one file per aircraft type
- **Engine model**: Polynomial thrust coefficients (5 for max thrust, 4 for
  fuel flow). Thrust depends on altitude; fuel flow depends on TAS.
- **Aerodynamics**: Drag polars (CD0, CD2) per aerodynamic configuration.
  Stall speeds are mass-corrected using `sqrt(m/m_ref)`.
- **Envelope**: VMO/MMO/HMO limits. Maximum altitude is mass- and
  temperature-dependent (formula 3.5-1). Buffet onset via CLbo coefficient.
- **Synonyms**: `SYNONYM.NEW` file maps ICAO codes to OPF file prefixes.

## BADA4

Version 4 replaces simple coefficients with high-order polynomials evaluated
over (delta, Mach) domains.

- **Data**: XML files with structured element trees, one folder per aircraft
- **Engine model**: Bivariate polynomials for thrust as a function of pressure
  ratio (delta) and Mach number. Separate coefficient sets for different
  engine settings (MCMB, MCRZ, LIDL, etc.).
- **Aerodynamics**: Drag coefficients indexed by (high-lift device position,
  landing gear state). Supports ARPM (Airline Procedure Model) configuration.
- **Envelope**: Geometric, kinematic, and buffet limit models (GLM/KLM/BLM).
  Configuration-dependent VFE and CL_max.
- **Synonyms**: `SYNONYM.BD4` file maps ICAO codes to XML folder names.

## OpenAP

Open Aircraft Performance model based on publicly available data.

- **Data**: Loaded via the `openap` Python library (no file parsing)
- **Engine model**: Simplified thrust and fuel flow functions from the OpenAP
  database.
- **Aerodynamics**: Drag polar with configuration selection via flap angle
  and landing gear flags (rather than the altitude/speed thresholds used by
  BADA).
- **Envelope**: Constant VMO, MMO, HMO (service ceiling). No
  configuration-dependent adjustments.
- **Synonyms**: Uses OpenAP's built-in synonym dictionary.

OpenAP is the easiest model to get started with -- it requires no license or
data configuration.

## Poll-Schumann (PS)

A cruise-focused model designed for contrail and climate impact research.

- **Data**: Loaded via the `pycontrails` library
- **Engine model**: Thrust and efficiency coefficients with idle fuel flow
  parameters. Designed for cruise conditions.
- **Aerodynamics**: Reynolds-based skin friction drag plus wave drag
  (coefficients j1, j2, j3) and induced drag via Oswald efficiency factor.
  Wing aspect ratio is computed from span and area.
- **Envelope**: Altitude-dependent Mach limits derived from impact pressure
  constraints (p_i_max). VMO is effectively infinite (Mach limits dominate at
  cruise altitudes). The maximum altitude depends on mass, temperature
  deviation and Mach number: it is the lowest of hmo, the thrust-limited
  ceiling (300 ft/min residual climb at maximum thrust) and the lift-limited
  ceiling (maximum usable lift coefficient), and never below FL200.
- **Limitations**: Clean configuration only; not suitable for climb/descent
  with deployed high-lift devices.

## Choosing a model

- **Need open data with no setup?** Use OpenAP.
- **Need high fidelity for all flight phases?** Use BADA4 (requires license).
- **Need wide aircraft coverage?** Use BADA3 (requires license).
- **Working on contrail/climate research?** Use Poll-Schumann.
- **Not sure?** Use `ModelFactory.load("A320")` -- it picks the best available.
