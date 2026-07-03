# Resolve an aircraft code

This guide shows how to find which performance models are available for a given
ICAO aircraft type code, and how the synonym system maps codes to data files.

## Check availability across all models

```python
from openleaf.performance.core import ModelFactory

# Check one aircraft across all model types
availability = ModelFactory.is_available("A320")
print(availability)
# {ModelType.BADA3: True, ModelType.BADA4: True, ModelType.OPENAP: True, ModelType.PS: True}

# Check for a specific model type
ModelFactory.is_available("A320", model_type="bada4")
# True
```

## List all available aircraft

```python
# All aircraft for a given model type
aircraft_list = ModelFactory.available_aircraft("openap")
print(aircraft_list)
# ['A319', 'A320', 'A321', 'B738', ...]

# All aircraft across all model types
all_aircraft = ModelFactory.available_aircraft()
# {ModelType.BADA3: [...], ModelType.BADA4: [...], ...}
```

## How synonym resolution works

BADA3 and BADA4 use synonym files to map ICAO codes to data files on disk.
This is necessary because data file names do not always match ICAO codes.

### BADA4 resolution order

1. Look up ICAO code in `SYNONYM.BD4` to get a folder name
2. Search for `folder/name.xml` in the data directory
3. Search for any XML file in the folder
4. Fall back to filesystem scan matching the ICAO prefix

### BADA3 resolution order

1. Look up ICAO code in `SYNONYM.NEW` to get an OPF prefix
2. Try `prefix.OPF` directly
3. Try underscore suffix variants (`prefix__.OPF`, `prefix___.OPF`, `prefix_.OPF`)
4. Fall back to filesystem scan

### OpenAP and Poll-Schumann

These models use their respective library's built-in synonym dictionaries.
No file-level resolution is needed.

## Handle missing aircraft

When an aircraft is not found, `ModelFactory.load()` raises a `ValueError`
listing all model types it tried:

```python
from openleaf.performance.core import ModelFactory

try:
    model = ModelFactory.load("ZZZZ")
except ValueError as e:
    print(e)
    # Aircraft 'ZZZZ' not found in any model. Tried:
    #   - bada4: ...
    #   - bada3: ...
    #   - openap: ...
    #   - ps: ...
```

To load from a specific file when the synonym system does not cover your
aircraft:

```python
model = ModelFactory.load_from_file("/path/to/custom/A320.xml", model_type="bada4")
```
