# Aircraft Performance Data

This directory should contain aircraft performance model data files.

## BADA (Base of Aircraft Data)

BADA is a proprietary aircraft performance model developed and maintained by EUROCONTROL.
The data files are **confidential** and cannot be distributed with this repository.

### How to Obtain BADA Data

1. Visit the EUROCONTROL BADA website: https://www.eurocontrol.int/model/bada
2. Request a license for BADA data
3. Once approved, download the aircraft data files

### Required Files for Tests

Place the following files in this directory to run the full test suite:

| File | Description | Format |
|------|-------------|--------|
| `A320__.OPF` | Airbus A320 BADA3 data | BADA3 OPF text format |
| `A320-212.xml` | Airbus A320-212 BADA4 data | BADA4 XML format |

### File Formats

- **BADA3**: Text-based format with `.OPF` (operations), `.APF` (airline procedures), and `.PTF` (performance table) files
- **BADA4**: XML-based format with comprehensive aircraft performance data

### Running Tests Without BADA Data

Tests that require BADA data will be automatically skipped if the data files are not present.
You can still run other tests (ISA, aerodynamics, type utilities) without the BADA data.

```bash
# Run all tests (BADA tests will be skipped if data unavailable)
pytest tests/

# Run only non-BADA tests
pytest tests/test_physics.py tests/test_numpy.py tests/test_types.py
```


## Bundled Sample Data

| File | Contents |
|------|----------|
| `flight_data.json` | One ADS-B-derived B738 trajectory (~1,100 km, callsign anonymized) used by the examples |
| `met.npz` | Pressure-level air temperature and zonal wind sample (8 levels, 150-400 hPa) matching the trajectory |

<!-- TODO(maintainers, before wider redistribution): record the exact source
     and license/attribution of both files here (ADS-B network terms for the
     trajectory; ERA5/Copernicus attribution if the met sample derives from
     CDS data), or replace them with synthetic equivalents. -->

## Other Performance Models

The OpenLeaf framework is designed to support multiple performance models:

- **OpenAP**: Open Aircraft Performance model (open source)
- **Poll-Schumann**: Fuel flow and emissions model
- **Custom models**: Implement the abstract interfaces in `openleaf.performance.core.perfos`

Contributions for additional performance models are welcome!
