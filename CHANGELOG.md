# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Release tags are per package: `performance/vX.Y.Z` for `openleaf-performance`
and `core/vX.Y.Z` for `openleaf-core`.

## [Unreleased]

### openleaf-performance

#### Added

- `PSEngineModel.overall_propulsion_efficiency(data)`: the overall propulsion
  efficiency that PS `fuel_flow` uses, with the same thrust-coefficient clip
  and floor (#45). The raw correlation `overall_propulsion_efficiency` is
  re-exported from `openleaf.performance.models.ps`.

#### Changed

- PS `max_altitude` now depends on the flight state, as BADA3's does: it is
  the lowest of `hmo`, the thrust-limited ceiling (300 ft/min residual climb at
  maximum thrust) and the lift-limited ceiling, at `mach_number` (design Mach
  if absent), never below FL200. It requires `aircraft_mass` and `delta_isa`,
  and the envelope `ceiling` flag follows it. `PSLimits` now takes the PS
  `engine` and `aerodynamics` models as keyword arguments (#44).

#### Fixed

- `tem.f_from_state`: the level-flight test is on specific excess power
  (1e-3 m/s) rather than a 1e-10 N thrust excess, so f no longer jumps between
  1 and ~1e14 with the last bits of thrust and drag (#46).
- Trajectory `thrust` keeps the m·a term for level accelerations, where the
  energy share factor is 0 (#47).
- BADA4 `max_altitude_for(hl_deployed=True)` returns MFA instead of HMO (#48).
- Poll & Schumann (2021) citation pages (257-295, with DOI) and the
  `PSModel("A320").mtom` docstring example (#49).

## [0.1.0] - Unreleased
<!-- Set the release date when tagging performance/v0.1.0 -->

Initial public release.

### openleaf-performance

- Unified aircraft-performance API: `ModelFactory` (model loading and
  availability queries across providers, with BADA4 > BADA3 > OpenAP > PS
  priority) and `TrajectoryData`, a lazy flight-state container that derives
  ISA atmosphere, Mach number, calibrated airspeed and energy-share factor
  on demand.
- Four model providers behind one interface: BADA 3 and BADA 4 (parsers for
  the licensed EUROCONTROL datasets — no BADA data is included or required
  for the open providers), Poll-Schumann (via pycontrails) and OpenAP (via
  the openap library), each providing thrust, fuel-flow, drag,
  configuration-selection and envelope-limit models.
- Multi-backend numerics: NumPy by default, with optional CasADi (symbolic
  computation and trajectory optimization), PyTorch (including GPU) and JAX
  backends behind a numpy-compatible dispatch layer (portions derived from
  AeroSandbox, MIT — see NOTICE).
- Physics modules: ISA atmosphere, total-energy model, energy-share factor,
  compressible airspeed conversions (CAS/TAS/Mach) and unit constants.
- Flight-envelope checking (speed, Mach, altitude and mass limits) with
  IGNORE / WARN / RAISE handling modes.
- Configuration of licensed BADA data paths via environment variables, a
  user-level settings file, or the Python API.
- Validation suite: reference checks against EUROCONTROL PTD tables for
  climb and descent (idle) performance for BADA3 and BADA4 across jet,
  turboprop and piston types; a fully synthetic OPF fixture pinning parser
  field order; cross-backend consistency tests. About 1,250 tests run with
  licensed BADA data configured, about 920 without (BADA-dependent tests
  skip with guidance).
- Bundled examples with an anonymized ADS-B sample trajectory and an
  ERA5-derived meteorological sample (see `data/README.md` for provenance
  and attribution).
- Packaging: EUPL-1.2 with PEP 639 license metadata; NOTICE and third-party
  license texts (AeroSandbox / MIT, pycontrails / Apache-2.0, OpenAP /
  LGPL-3.0) shipped in wheels and sdists; casadi, torch, jax, openap and
  pycontrails are optional extras — the base install runs with the NumPy
  backend alone.

### openleaf-core

- Reserved placeholder package for future shared types and interfaces of
  the OpenLeaf suite (no public API yet).
