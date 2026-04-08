"""
Pytest configuration and shared fixtures for aetherx tests.

Provides fixtures for aircraft models, test data paths, and common test utilities.
Tests requiring proprietary BADA data are automatically skipped if data is unavailable.
"""

import os
import sys
from pathlib import Path


# Add tests directory to path for local test modules (e.g., ptd_parser)
sys.path.insert(0, str(Path(__file__).parent))

# Local data directory (for backwards compatibility)
_local_data_dir = Path(__file__).parent.parent / "data"

# Configure BADA paths from environment variables BEFORE importing aetherx
# Priority: env var > local data/ folder
if (
    "OPENLEAF_BADA3_PATH" not in os.environ
    and _local_data_dir.exists()
    and (_local_data_dir / "A320__.OPF").exists()
):
    os.environ["OPENLEAF_BADA3_PATH"] = str(_local_data_dir)

if "OPENLEAF_BADA4_PATH" not in os.environ and _local_data_dir.exists():
    # Check for flat structure (A320-212.xml) or folder structure (A320-212/A320-212.xml)
    flat_exists = (_local_data_dir / "A320-212.xml").exists()
    folder_exists = (_local_data_dir / "A320-212" / "A320-212.xml").exists()
    if flat_exists or folder_exists:
        os.environ["OPENLEAF_BADA4_PATH"] = str(_local_data_dir)

import numpy as np
import pytest

# Patch CasADi's broken SWIG __hash__ binding (GenSharedObject___hash__)
# so that CasADi types can be used in sets and as dict keys in tests.
try:
    import casadi

    for _cls in (casadi.DM, casadi.MX, casadi.SX):
        if not callable(getattr(_cls, "__hash__", None)):
            continue
        try:
            hash(_cls(0))
        except (NotImplementedError, TypeError):
            _cls.__hash__ = lambda self: id(self)
except ImportError:
    pass


# --- Helper Functions ---


def _find_bada3_file(data_dir: Path, filename: str) -> Path | None:
    """Find BADA3 file in data directory."""
    # Direct file (flat structure)
    path = data_dir / filename
    if path.exists():
        return path
    return None


def _find_bada4_file(data_dir: Path, aircraft_name: str) -> Path | None:
    """Find BADA4 XML file supporting both flat and folder structures.

    Supports:
    - Flat: data_dir/A320-212.xml
    - Folder: data_dir/A320-212/A320-212.xml
    """
    # Flat structure
    flat_path = data_dir / f"{aircraft_name}.xml"
    if flat_path.exists():
        return flat_path

    # Folder structure (BADA4 standard)
    folder_path = data_dir / aircraft_name / f"{aircraft_name}.xml"
    if folder_path.exists():
        return folder_path

    return None


def _find_bada4_ptd(data_dir: Path, aircraft_name: str, isa_suffix: str = "_ISA") -> Path | None:
    """Find BADA4 PTD file supporting both flat and folder structures.

    Supports:
    - Flat: data_dir/A320-212_ISA.PTD
    - Folder: data_dir/A320-212/A320-212_ISA.PTD
    """
    ptd_name = f"{aircraft_name}{isa_suffix}.PTD"

    # Flat structure
    flat_path = data_dir / ptd_name
    if flat_path.exists():
        return flat_path

    # Folder structure (BADA4 standard)
    folder_path = data_dir / aircraft_name / ptd_name
    if folder_path.exists():
        return folder_path

    return None


# --- Path Fixtures ---


@pytest.fixture
def data_dir() -> Path:
    """Return the path to the test data directory (from env var or local)."""
    # Prefer environment variable if set
    if "OPENLEAF_BADA3_PATH" in os.environ:
        return Path(os.environ["OPENLEAF_BADA3_PATH"])
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def bada3_data_dir() -> Path:
    """Return the path to BADA3 data directory."""
    if "OPENLEAF_BADA3_PATH" in os.environ:
        return Path(os.environ["OPENLEAF_BADA3_PATH"])
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def bada4_data_dir() -> Path:
    """Return the path to BADA4 data directory."""
    if "OPENLEAF_BADA4_PATH" in os.environ:
        return Path(os.environ["OPENLEAF_BADA4_PATH"])
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def bada3_a320_path(bada3_data_dir: Path) -> Path:
    """Return the path to the BADA3 A320 OPF file.

    Skips test if file is not available (proprietary data).
    """
    path = _find_bada3_file(bada3_data_dir, "A320__.OPF")
    if path is None:
        pytest.skip(
            f"BADA3 data file not found: {bada3_data_dir}/A320__.OPF. "
            "BADA data is proprietary and must be obtained separately from EUROCONTROL."
        )
    return path


@pytest.fixture
def bada4_a320_path(bada4_data_dir: Path) -> Path:
    """Return the path to the BADA4 A320 XML file.

    Skips test if file is not available (proprietary data).
    """
    path = _find_bada4_file(bada4_data_dir, "A320-212")
    if path is None:
        pytest.skip(
            f"BADA4 data file not found: {bada4_data_dir}/A320-212.xml or A320-212/A320-212.xml. "
            "BADA data is proprietary and must be obtained separately from EUROCONTROL."
        )
    return path


@pytest.fixture
def bada3_at72_path(bada3_data_dir: Path) -> Path:
    """Return the path to the BADA3 AT72 (ATR72 Turboprop) OPF file.

    Skips test if file is not available (proprietary data).
    """
    path = _find_bada3_file(bada3_data_dir, "AT72__.OPF")
    if path is None:
        pytest.skip(
            f"BADA3 data file not found: {bada3_data_dir}/AT72__.OPF. "
            "BADA data is proprietary and must be obtained separately from EUROCONTROL."
        )
    return path


@pytest.fixture
def bada3_sr20_path(bada3_data_dir: Path) -> Path:
    """Return the path to the BADA3 SR20 (Cirrus SR20 Piston) OPF file.

    Skips test if file is not available (proprietary data).
    """
    path = _find_bada3_file(bada3_data_dir, "SR20__.OPF")
    if path is None:
        pytest.skip(
            f"BADA3 data file not found: {bada3_data_dir}/SR20__.OPF. "
            "BADA data is proprietary and must be obtained separately from EUROCONTROL."
        )
    return path


@pytest.fixture
def bada4_at72_path(bada4_data_dir: Path) -> Path:
    """Return the path to the BADA4 ATR72-200 (Turboprop) XML file.

    Skips test if file is not available (proprietary data).
    """
    path = _find_bada4_file(bada4_data_dir, "ATR72-200")
    if path is None:
        pytest.skip(
            f"BADA4 data file not found: {bada4_data_dir}/ATR72-200.xml or ATR72-200/ATR72-200.xml. "
            "BADA data is proprietary and must be obtained separately from EUROCONTROL."
        )
    return path


@pytest.fixture
def bada4_sr20_path(bada4_data_dir: Path) -> Path:
    """Return the path to the BADA4 SR20 (Cirrus SR20 Piston) XML file.

    Skips test if file is not available (proprietary data).
    """
    path = _find_bada4_file(bada4_data_dir, "SR20")
    if path is None:
        pytest.skip(
            f"BADA4 data file not found: {bada4_data_dir}/SR20.xml or SR20/SR20.xml. "
            "BADA data is proprietary and must be obtained separately from EUROCONTROL."
        )
    return path


# --- Model Fixtures ---


@pytest.fixture
def bada3_model(bada3_a320_path: Path):
    """Create a BADA3 model from the A320 data file.

    Skips test if BADA3 data is not available.
    """
    from openleaf.performance.models.bada3 import BADA3Model

    return BADA3Model.from_file(str(bada3_a320_path))


@pytest.fixture
def bada4_model(bada4_a320_path: Path):
    """Create a BADA4 model from the A320 data file.

    Skips test if BADA4 data is not available.
    """
    from openleaf.performance.models.bada4 import BADA4Model

    return BADA4Model.from_file(str(bada4_a320_path))


@pytest.fixture
def bada3_turboprop_model(bada3_at72_path: Path):
    """Create a BADA3 model for ATR72-200 turboprop aircraft.

    Skips test if BADA3 data is not available.
    """
    from openleaf.performance.models.bada3 import BADA3Model

    return BADA3Model.from_file(str(bada3_at72_path))


@pytest.fixture
def bada3_piston_model(bada3_sr20_path: Path):
    """Create a BADA3 model for SR20 piston aircraft.

    Skips test if BADA3 data is not available.
    """
    from openleaf.performance.models.bada3 import BADA3Model

    return BADA3Model.from_file(str(bada3_sr20_path))


@pytest.fixture
def bada4_turboprop_model(bada4_at72_path: Path):
    """Create a BADA4 model for ATR72-200 turboprop aircraft.

    Skips test if BADA4 data is not available.
    """
    from openleaf.performance.models.bada4 import BADA4Model

    return BADA4Model.from_file(str(bada4_at72_path))


@pytest.fixture
def bada4_piston_model(bada4_sr20_path: Path):
    """Create a BADA4 model for SR20 piston aircraft.

    Skips test if BADA4 data is not available.
    """
    from openleaf.performance.models.bada4 import BADA4Model

    return BADA4Model.from_file(str(bada4_sr20_path))


# --- Flight State Fixtures ---


@pytest.fixture
def typical_cruise_state() -> dict:
    """Return a typical cruise flight state dictionary with all required fields."""
    from openleaf.performance.core.perfos import EngineSetting
    from openleaf.performance.models.bada_types import AerodynamicConfig
    from openleaf.performance.physics import isa

    altitude = 10668.0  # m (FL350)
    return {
        "true_airspeed": 230.0,  # m/s (~450 kt)
        "altitude": altitude,
        "aircraft_mass": 65000.0,  # kg
        "mach_number": 0.78,
        "air_pressure_ratio": 0.2234,
        "air_temperature_ratio": 0.7519,
        "air_density_ratio": 0.2972,
        # Required fields for engine methods
        "engine_setting": EngineSetting.MCMB,
        "delta_isa": 0.0,
        # Required fields for aero methods
        "aerodynamic_configuration": AerodynamicConfig.CR,
        "lift_coef": 0.5,
        # Required fields for config determination
        "flight_phase": "cruise",
        "vertical_speed": 0.0,
        "f": 1.0,
        # Required fields for PS model
        "air_temperature": isa.air_temperature(altitude),
        "air_pressure": isa.air_pressure(altitude),
        "thrust": 50000.0,  # N
    }


@pytest.fixture
def sea_level_state() -> dict:
    """Return a sea-level flight state dictionary with all required fields."""
    from openleaf.performance.core.perfos import EngineSetting
    from openleaf.performance.models.bada_types import AerodynamicConfig
    from openleaf.performance.physics import isa

    altitude = 0.0  # m (sea level)
    return {
        "true_airspeed": 150.0,  # m/s (~290 kt)
        "altitude": altitude,
        "aircraft_mass": 75000.0,  # kg (near MTOW)
        "mach_number": 0.44,
        "air_pressure_ratio": 1.0,
        "air_temperature_ratio": 1.0,
        "air_density_ratio": 1.0,
        # Required fields for engine methods
        "engine_setting": EngineSetting.MCMB,
        "delta_isa": 0.0,
        # Required fields for aero methods
        "aerodynamic_configuration": AerodynamicConfig.CR,
        "lift_coef": 0.6,
        # Required fields for config determination
        "flight_phase": "climb",
        "vertical_speed": 5.0,  # m/s climb
        "f": 1.0,
        # Required fields for PS model
        "air_temperature": isa.air_temperature(altitude),
        "air_pressure": isa.air_pressure(altitude),
        "thrust": 100000.0,  # N
    }


@pytest.fixture
def turboprop_cruise_state() -> dict:
    """Return a typical turboprop cruise flight state dictionary (ATR72-200)."""
    from openleaf.performance.core.perfos import EngineSetting
    from openleaf.performance.models.bada_types import AerodynamicConfig
    from openleaf.performance.physics import isa

    altitude = 6096.0  # m (FL200)
    return {
        "true_airspeed": 140.0,  # m/s (~270 kt)
        "altitude": altitude,
        "aircraft_mass": 19000.0,  # kg
        "mach_number": 0.45,
        "air_pressure_ratio": 0.4660,
        "air_temperature_ratio": 0.8617,
        "air_density_ratio": 0.5409,
        # Required fields for engine methods
        "engine_setting": EngineSetting.MCMB,
        "delta_isa": 0.0,
        # Required fields for aero methods
        "aerodynamic_configuration": AerodynamicConfig.CR,
        "lift_coef": 0.55,
        # Required fields for config determination
        "flight_phase": "cruise",
        "vertical_speed": 0.0,
        "f": 1.0,
        # Required fields for PS model
        "air_temperature": isa.air_temperature(altitude),
        "air_pressure": isa.air_pressure(altitude),
        "thrust": 15000.0,  # N
    }


@pytest.fixture
def piston_cruise_state() -> dict:
    """Return a typical piston cruise flight state dictionary (SR20)."""
    from openleaf.performance.core.perfos import EngineSetting
    from openleaf.performance.models.bada_types import AerodynamicConfig
    from openleaf.performance.physics import isa

    altitude = 2438.0  # m (8000 ft)
    return {
        "true_airspeed": 75.0,  # m/s (~145 kt)
        "altitude": altitude,
        "aircraft_mass": 1360.0,  # kg
        "mach_number": 0.23,
        "air_pressure_ratio": 0.7428,
        "air_temperature_ratio": 0.9449,
        "air_density_ratio": 0.7863,
        # Required fields for engine methods
        "engine_setting": EngineSetting.MCMB,
        "delta_isa": 0.0,
        # Required fields for aero methods
        "aerodynamic_configuration": AerodynamicConfig.CR,
        "lift_coef": 0.6,
        # Required fields for config determination
        "flight_phase": "cruise",
        "vertical_speed": 0.0,
        "f": 1.0,
        # Required fields for PS model
        "air_temperature": isa.air_temperature(altitude),
        "air_pressure": isa.air_pressure(altitude),
        "thrust": 2000.0,  # N
    }


@pytest.fixture
def altitude_array() -> np.ndarray:
    """Return an array of test altitudes from sea level to FL400."""
    return np.array([0, 3000, 6000, 9000, 10668, 12000])


@pytest.fixture
def mach_array() -> np.ndarray:
    """Return an array of test Mach numbers."""
    return np.array([0.3, 0.5, 0.7, 0.78, 0.82])


# --- Atmospheric Utilities ---


def get_atm_ratios(altitude: float) -> tuple[float, float, float]:
    """Get atmospheric ratios (theta, delta, sigma) at altitude.

    Args:
        altitude: Pressure altitude [m].

    Returns:
        Tuple of (theta, delta, sigma) dimensionless ratios.
    """
    from openleaf.performance.physics import isa

    T = isa.air_temperature(altitude)  # Absolute temperature [K]
    P = isa.air_pressure(altitude)  # Absolute pressure [Pa]
    theta = T / isa.TAU_SL  # Temperature ratio
    delta = P / isa.P_SL  # Pressure ratio
    sigma = delta / theta  # Density ratio
    return theta, delta, sigma


def get_atm_state(altitude: float) -> dict:
    """Get atmospheric state at altitude for BADA4 engine methods.

    BADA4 requires actual air_pressure and air_temperature, not ratios.
    This computes them from altitude using ISA for test convenience.

    Args:
        altitude: Pressure altitude [m].

    Returns:
        Dict with air_pressure [Pa] and air_temperature [K].
    """
    from openleaf.performance.physics import isa

    return {
        "air_pressure": isa.air_pressure(altitude),
        "air_temperature": isa.air_temperature(altitude),
    }


@pytest.fixture
def a320_cruise_trajectory() -> dict[str, np.ndarray]:
    """Return 10 cruise samples for A320 at FL350.

    This fixture provides realistic A320 cruise trajectory data with 10 samples
    varying aircraft mass during cruise (fuel burn). All required fields for
    TrajectoryData are included.

    Returns:
        Dictionary with:
            - altitude: Pressure altitude Hp [m] - constant FL350
            - true_airspeed: TAS [m/s] - constant Mach 0.78
            - vertical_speed: Pressure altitude rate dHp/dt [m/s] - zero (level)
            - aircraft_mass: Aircraft mass [kg] - decreasing from 72000 to 63000
            - f: Energy share factor [-] - constant 1.0 (level cruise)
    """
    from openleaf.performance.physics import isa

    n_samples = 10

    # Constant cruise altitude: FL350 = 35000 ft
    altitude = 10668.0  # m
    altitude_arr = np.full(n_samples, altitude)

    # Compute TAS from Mach 0.78 at FL350
    mach = 0.78
    T = isa.air_temperature(altitude)  # ~218.8 K at FL350
    a = np.sqrt(isa.GAMMA * isa.R * T)  # Speed of sound
    tas = mach * a  # ~230.6 m/s

    true_airspeed = np.full(n_samples, tas)

    # Level cruise: zero vertical speed
    vertical_speed = np.zeros(n_samples)

    # Decreasing mass during cruise (fuel burn) - typical A320 range
    # Start at 72000 kg, end at 63000 kg over 10 samples
    aircraft_mass = np.linspace(72000.0, 63000.0, n_samples)

    # Energy share factor f=1.0 for constant speed level flight
    f = np.ones(n_samples)

    return {
        "altitude": altitude_arr,
        "true_airspeed": true_airspeed,
        "vertical_speed": vertical_speed,
        "aircraft_mass": aircraft_mass,
        "f": f,
    }
