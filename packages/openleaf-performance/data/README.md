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
| `flight_data.json` | One anonymized B738 climb/cruise trajectory (~1,100 km, 140 points) used by the examples |
| `met.npz` | Pressure-level air temperature and zonal wind sample (8 levels, 150-400 hPa) matching the trajectory |

### Provenance and licensing

- **`flight_data.json`** is derived from publicly broadcast ADS-B
  surveillance data. It has been anonymized for redistribution: it carries
  no callsign, ICAO 24-bit address, or registration (`flight_id` is the
  synthetic label `SAMPLE01`), and timestamps are rebased to elapsed
  seconds from the first sample. Only the aircraft type (B738) and the
  physical state arrays are retained, for demonstration purposes.
- **`met.npz`** contains modified Copernicus Climate Change Service
  information (ERA5 reanalysis) [2025]. Neither the European Commission
  nor ECMWF is responsible for any use that may be made of the Copernicus
  information or data it contains.

## Other Performance Models

The OpenLeaf framework is designed to support multiple performance models:

- **OpenAP**: Open Aircraft Performance model (open source)
- **Poll-Schumann**: Fuel flow and emissions model
- **Custom models**: Implement the abstract interfaces in `openleaf.performance.core.perfos`

Contributions for additional performance models are welcome!
