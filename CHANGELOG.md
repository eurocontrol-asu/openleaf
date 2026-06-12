# Changelog

## [0.1.0] - 2026-06-02

### Bug Fixes

- adapt CI/CD workflows and config for openleaf packages
- centralize ruff config at workspace root and fix all lint errors
- use config API for BADA path resolution in test fixtures
- clean up dependency hygiene for openleaf-performance
- resolve lint issues (TC003 + import sorting)
- enable coverage XML report generation for Codecov
- remove redundant mypy overrides and fix factory typing
- resolve mypy duplicate module error for openleaf-performance
- lazy-resolve BADA data paths to fix test collection
- resolve mypy errors in KeyedData protocol and tests
- correct project metadata and broken developer tooling

### Build & CI

- switch docs deployment from Cloudflare to GitHub Pages
- rename docs workflow to GitHub Pages
- checkout BADA data repo for performance tests
- migrate CI jobs to a self-hosted GPU runner
- install all extras (torch) for GPU tests

### Documentation

- add initial documentation scaffolding
- document all BADA configuration methods in README
- add Material theme docs site
- rewrite documentation following Diataxis framework
- fix link paths and update docs dependencies
- add logo, update branding, coverage badges, and docs content
- remove placeholder and fix stale citation keys

### Features

- initial OpenLeaf workspace with openleaf-core and openleaf-performance
- initialize uv workspace with CI, docs, and project scaffolding

### Miscellaneous

- apply ruff format across all source and test files
- add examples dependency group for matplotlib
- add pre-commit config with ruff, mypy, and conventional-commits
- pin python version at workspace root
- add --import-mode=importlib to pytest config
- update repo URLs, authors, and copyright to eurocontrol-asu
- add standard open-source project files

### Refactor

- reduce complexity of is_casadi_type and is_torch_type
- reduce complexity of _promote_to_common_backend
- reduce complexity of parse_limits in bada4/parsing.py
- reduce complexity of parse_arpm in bada4/parsing.py
- reduce complexity of interpn in numpy/interpolate.py
- reduce complexity of _update_config_file in config.py
- rename aetherx references to openleaf across codebase
- reduce complexity of resolve_bada4_file and resolve_bada3_file
- reduce complexity of diag in array.py
- reduce complexity of resolve_bada3_file in synonyms.py
- reduce complexity of stack and reshape in array.py
- reduce complexity of _ensure_pair in logicals.py
- reduce complexity of ModelFactory.load in factory.py
- reduce complexity of _f_from_speed_schedule in tem.py
- reduce complexity of _interpn_casadi in interpolate.py
- reduce complexity of available_bada4_aircraft in synonyms.py
- extract EnvelopeChecker from AircraftTrajectoryData
- extract PerformanceMixin from AircraftTrajectoryData
- extract FlightStateMixin from AircraftTrajectoryData
