# openleaf-core

Shared types, protocols, and exceptions for the [OpenLeaf](https://github.com/eurocontrol-asu/openleaf) aviation sustainability suite.

This package is intentionally dependency-light — only numpy is required at runtime — so any OpenLeaf consumer can target the contract without inheriting heavy optional dependencies (CasADi, PyTorch, pandas).

## Public surface

```python
from openleaf.core import (
    # Types
    KeyedData,           # Protocol: dict-like flight state access
    ArrayScalarLike,     # Annotation: NumPy/pandas/CasADi numeric union
    is_keyed_data,       # Runtime check for KeyedData compliance

    # Model protocols (runtime-checkable except AircraftModelProtocol)
    EngineModelProtocol,
    AerodynamicsModelProtocol,
    LimitsModelProtocol,
    AircraftModelProtocol,

    # Exception hierarchy (catch-all root: OpenLeafError)
    OpenLeafError,
    ConfigurationError,
    ModelError, ModelNotFoundError, ModelDataError,
    ParsingError,
    ValidationError, MissingFieldError, ValueRangeError,
    EnvelopeError,
)
```

Concrete ABCs with physics-aware defaults (`EngineModel`, `AerodynamicsModel`, `LimitsModel`, `AircraftModel`) live in `openleaf-performance`. Provider-specific exception subclasses (`BADA3ParseError`, `BADA4ParseError`) also live with their providers.
