"""
Trajectory data container for aircraft performance calculations.

This module provides a simple, lazy data container that holds trajectory data
and automatically computes derived quantities on access.

All columns are per-row arrays. No scalars, no attrs resolution magic.

Altitude convention: PRESSURE altitude (Hp), as per BADA/aviation standard.
Vertical speed convention: PRESSURE altitude rate (dHp/dt).

Example:
    >>> import numpy as np
    >>> from openleaf.performance.core import TrajectoryData
    >>>
    >>> # From ADS-B data (has acceleration)
    >>> data = {
    ...     "altitude": np.array([10000.0, 10500.0]),
    ...     "true_airspeed": np.array([230.0, 232.0]),
    ...     "vertical_speed": np.array([5.0, 5.0]),
    ...     "aircraft_mass": np.array([65000.0, 64900.0]),
    ...     "acceleration": np.array([0.1, 0.1]),
    ... }
    >>> traj = TrajectoryData(data)
    >>> print(traj.mach_number)  # Computed automatically
    >>>
    >>> # Add aircraft model for performance
    >>> perf = traj.with_aircraft("BADA4", "A320")
    >>> print(perf.fuel_flow)  # Computes drag → thrust → fuel_flow
"""

from __future__ import annotations


__all__ = [
    "TrajectoryData",
    "AircraftTrajectoryData",
]

from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import openleaf.performance.numpy as np
from openleaf.performance.core.envelope import EnvelopeChecker
from openleaf.performance.core.mixins import FlightStateMixin
from openleaf.performance.core.mixins import PerformanceMixin
from openleaf.performance.core.types import EnvelopeMode
from openleaf.performance.logging import get_logger
from openleaf.performance.physics import isa
from openleaf.performance.physics.aero import AircraftAirspeed
from openleaf.performance.physics.units import EPSILON_TOLERANCE
from openleaf.performance.utils.types import ArrayScalarLike
from openleaf.performance.utils.types import KeyedData


if TYPE_CHECKING:
    from collections.abc import Mapping

    from openleaf.performance.core.perfos import AircraftModel

logger = get_logger(__name__)


class TrajectoryData:
    """Lazy trajectory container with automatic derivation of flight state.

    All columns are per-row arrays. Computed properties are cached.

    Required columns:
        - altitude: Pressure altitude Hp [m]
        - true_airspeed: TAS [m/s]
        - vertical_speed: Pressure altitude rate dHp/dt [m/s]
        - aircraft_mass: Aircraft mass [kg]
        - f OR acceleration: Energy share factor [-] or dV/dt [m/s²]

    Computed properties (lazy):
        - air_temperature, air_pressure, air_density (ISA from Hp)
        - mach_number, calibrated_airspeed
        - f ↔ acceleration (derive missing one)
        - vertical_speed_geometric
    """

    REQUIRED = frozenset(
        {
            "altitude",  # Pressure altitude Hp [m]
            "true_airspeed",  # TAS [m/s]
            "vertical_speed",  # Pressure altitude rate dHp/dt [m/s]
            "aircraft_mass",  # [kg]
        }
    )

    ENERGY_COLUMNS = frozenset({"f", "acceleration"})

    def __init__(self, data: KeyedData):
        """Initialize trajectory data container.

        Args:
            data: Dictionary, DataFrame, or pycontrails Flight/Fleet.
                Must contain REQUIRED columns plus either 'f' or 'acceleration'.

        Raises:
            ValueError: If required fields are missing.
        """
        self._data = dict(data)
        self._attrs: Mapping[str, Any] = getattr(data, "attrs", {})
        self._validate()

    def _validate(self) -> None:
        """Validate required columns are present."""
        missing = self.REQUIRED - self._data.keys()
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not (self.ENERGY_COLUMNS & self._data.keys()):
            raise ValueError("Must provide 'f' or 'acceleration' column")

    # ---- Direct access ----

    def __getitem__(self, key: str) -> ArrayScalarLike:
        """Get column or computed property."""
        if key in self._data:
            return self._data[key]

        # Check for cached_property
        attr = getattr(type(self), key, None)
        if isinstance(attr, cached_property):
            return getattr(self, key)

        raise KeyError(f"Key '{key}' not found in data or computed properties")

    def __contains__(self, key: str) -> bool:
        """Check if key is available in data or as computed property."""
        if key in self._data:
            return True
        attr = getattr(type(self), key, None)
        return isinstance(attr, cached_property)

    def __len__(self) -> int:
        """Return number of data points."""
        alt = self._data["altitude"]
        if hasattr(alt, "__len__"):
            return len(alt)
        return 1

    def __repr__(self) -> str:
        """Return string representation."""
        n = len(self)
        cols = sorted(self._data.keys())
        return f"TrajectoryData(n={n}, columns={cols})"

    def __iter__(self):
        """Iterate over data keys."""
        return iter(self._data)

    def keys(self):
        """Return data column keys."""
        return self._data.keys()

    def get(self, key: str, default: ArrayScalarLike | None = None) -> ArrayScalarLike | None:
        """Get a data value with optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def data(self) -> dict:
        """Expose the raw input data."""
        return self._data

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Metadata dictionary (from DataFrame.attrs or Flight.attrs)."""
        return self._attrs

    # ---- Atmospheric (ISA from pressure altitude) ----

    @cached_property
    def air_temperature(self) -> ArrayScalarLike:
        """Air temperature [K]. Uses provided value or ISA."""
        if "air_temperature" in self._data:
            return self._data["air_temperature"]
        return isa.air_temperature(self._data["altitude"])

    @cached_property
    def air_pressure(self) -> ArrayScalarLike:
        """Air pressure at pressure altitude [Pa]."""
        return isa.air_pressure(self._data["altitude"])

    @cached_property
    def air_density(self) -> ArrayScalarLike:
        """Air density [kg/m³]."""
        return self.air_pressure / (isa.R * self.air_temperature)

    @cached_property
    def delta_isa(self) -> ArrayScalarLike:
        """Temperature deviation from ISA [K]."""
        if "air_temperature" not in self._data:
            return np.zeros_like(self._data["altitude"])
        T_isa = isa.air_temperature(self._data["altitude"])
        return self._data["air_temperature"] - T_isa

    # ---- Airspeed conversions ----

    @cached_property
    def _airspeed(self) -> AircraftAirspeed:
        """Internal airspeed converter."""
        return AircraftAirspeed(
            T=self.air_temperature,
            p=self.air_pressure,
            v=self._data["true_airspeed"],
        )

    @cached_property
    def mach_number(self) -> ArrayScalarLike:
        """Mach number [-]."""
        return self._airspeed.mach_number

    @cached_property
    def calibrated_airspeed(self) -> ArrayScalarLike:
        """Calibrated airspeed [m/s]."""
        return self._airspeed.calibrated_airspeed

    # ---- Vertical speed conversions ----

    @cached_property
    def _temp_correction_factor(self) -> ArrayScalarLike:
        """Temperature correction factor T_actual / T_ISA."""
        T_isa = isa.air_temperature(self._data["altitude"])
        return self.air_temperature / T_isa

    @cached_property
    def vertical_speed_geometric(self) -> ArrayScalarLike:
        """Geometric vertical speed dh/dt [m/s].

        Converts pressure altitude rate to geometric:
            dh/dt = dHp/dt * (T_actual / T_ISA)
        """
        return self._data["vertical_speed"] * self._temp_correction_factor

    # ---- Energy share factor ----

    @cached_property
    def f(self) -> ArrayScalarLike:
        """Energy share factor [-] (per-row).

        If not provided, derived from acceleration:
            f = vs_geo / (vs_geo + (V/g) * a)
        """
        if "f" in self._data:
            return self._data["f"]

        # Derive from acceleration
        vs_geo = self.vertical_speed_geometric
        V = self._data["true_airspeed"]
        a = self._data["acceleration"]

        denominator = vs_geo + (V / isa.G) * a

        # Handle level flight (vs=0, a=0): f is irrelevant, use 1.0
        # Clamp denominator to avoid division by zero (np.where evaluates both branches)
        safe_denom = np.where(np.abs(denominator) < EPSILON_TOLERANCE, 1.0, denominator)
        return np.where(
            np.abs(denominator) < EPSILON_TOLERANCE,
            1.0,
            vs_geo / safe_denom,
        )

    @cached_property
    def acceleration(self) -> ArrayScalarLike:
        """Longitudinal acceleration dV/dt [m/s²] (per-row).

        If not provided, derived from f:
            a = (g/V) * vs_geo * (1 - f) / f
        """
        if "acceleration" in self._data:
            return self._data["acceleration"]

        # Derive from f
        vs_geo = self.vertical_speed_geometric
        V = self._data["true_airspeed"]
        f = self._data["f"]

        # Handle f=1 (constant speed): a=0
        # Handle vs=0 (level flight): a=0
        return np.where(
            (np.abs(f - 1.0) < EPSILON_TOLERANCE) | (np.abs(vs_geo) < EPSILON_TOLERANCE),
            0.0,
            (isa.G / V) * vs_geo * (1.0 - f) / f,
        )

    # ---- Aircraft model attachment ----

    def with_aircraft(
        self,
        model: AircraftModel | str,
        icao: str | None = None,
        envelope_mode: EnvelopeMode = EnvelopeMode.IGNORE,
    ) -> AircraftTrajectoryData:
        """Attach an aircraft model for performance calculations.

        Args:
            model: Either:
                - An instantiated model: BADA4Model("A320")
                - A model type string: "BADA3", "BADA4", "OpenAP", "PS"
            icao: ICAO type code, required if model is a string.
            envelope_mode: How to handle envelope violations (default: IGNORE).

        Returns:
            AircraftTrajectoryData with drag, thrust, fuel_flow, etc.

        Examples:
            # Option 1: Pre-built model
            perf = traj.with_aircraft(BADA4Model("A320"))

            # Option 2: Factory style
            perf = traj.with_aircraft("BADA4", "A320")
            perf = traj.with_aircraft("OpenAP", "A320")

            # Option 3: With envelope checking
            perf = traj.with_aircraft("BADA4", "A320", envelope_mode=EnvelopeMode.WARN)
        """
        if isinstance(model, str):
            if icao is None:
                raise ValueError("icao required when model is a string")
            model = _create_model(model, icao)

        return AircraftTrajectoryData(self, model, envelope_mode=envelope_mode)


class AircraftTrajectoryData(FlightStateMixin, PerformanceMixin):
    """Trajectory with aircraft model attached - adds performance properties.

    This class wraps a TrajectoryData and adds aircraft-specific computed
    properties like drag, thrust, and fuel_flow.

    All TrajectoryData properties are accessible via delegation.

    Args:
        envelope_mode: How to handle flight envelope violations:
            - IGNORE (default): Compute violations but take no action
            - WARN: Log warnings when violations are detected
            - RAISE: Raise EnvelopeViolationError when violations exist
    """

    def __init__(
        self,
        trajectory: TrajectoryData,
        model: AircraftModel,
        envelope_mode: EnvelopeMode = EnvelopeMode.IGNORE,
    ):
        """Initialize with trajectory and aircraft model.

        Args:
            trajectory: TrajectoryData instance.
            model: Aircraft performance model.
            envelope_mode: How to handle envelope violations (default: IGNORE).
        """
        self._traj = trajectory
        self._model = model
        self._envelope_mode = envelope_mode

    def __getitem__(self, key: str) -> ArrayScalarLike:
        """Get column or computed property from trajectory or aircraft.

        Lookup order:
        1. Input data (trajectory._data)
        2. TrajectoryData computed properties (air_temperature, mach_number, etc.)
        3. AircraftTrajectoryData computed properties (drag_coef, thrust, etc.)
        """
        # Check input data first
        if key in self._traj._data:
            return self._traj._data[key]

        # Check TrajectoryData computed properties
        if key in self._traj:
            return self._traj[key]

        # Check our cached_property
        attr = getattr(type(self), key, None)
        if isinstance(attr, cached_property):
            return getattr(self, key)

        raise KeyError(f"Key '{key}' not found in data or computed properties")

    def __contains__(self, key: str) -> bool:
        """Check if key is available in input data or TrajectoryData.

        NOTE: This does NOT include AircraftTrajectoryData computed properties
        (drag_coef, thrust, etc.) to allow models to distinguish "user provided
        this value" from "system will compute this value on access".
        """
        # Only report keys that are in the input data or TrajectoryData
        # Do NOT include our computed properties - those are outputs, not inputs
        return key in self._traj

    def __len__(self) -> int:
        """Return number of data points in the trajectory."""
        return len(self._traj)

    def __repr__(self) -> str:
        """Return string representation."""
        n = len(self)
        model_name = self._model.model
        model_type = type(self._model).__name__
        return f"AircraftTrajectoryData(n={n}, model={model_name!r}, type={model_type})"

    def __iter__(self):
        """Iterate over input data keys."""
        return iter(self._traj)

    def keys(self):
        """Return input data column keys."""
        return self._traj.keys()

    def get(self, key: str, default: ArrayScalarLike | None = None) -> ArrayScalarLike | None:
        """Get a value by key with optional default.

        Args:
            key: Key to retrieve.
            default: Default value if key not found.

        Returns:
            Value for key or default.
        """
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def data(self) -> dict:
        """Raw input data dictionary from trajectory."""
        return self._traj.data

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Metadata dictionary from trajectory."""
        return self._traj.attrs

    @property
    def trajectory(self) -> TrajectoryData:
        """Access the underlying TrajectoryData."""
        return self._traj

    @property
    def model(self) -> AircraftModel:
        """Access the aircraft model."""
        return self._model

    # ---- Delegated properties from TrajectoryData ----

    @property
    def air_temperature(self) -> ArrayScalarLike:
        """Air temperature [K] from trajectory."""
        return self._traj.air_temperature

    @property
    def air_pressure(self) -> ArrayScalarLike:
        """Air pressure [Pa] from trajectory."""
        return self._traj.air_pressure

    @property
    def air_density(self) -> ArrayScalarLike:
        """Air density [kg/m³] from trajectory."""
        return self._traj.air_density

    @property
    def mach_number(self) -> ArrayScalarLike:
        """Mach number [-] from trajectory."""
        return self._traj.mach_number

    @property
    def calibrated_airspeed(self) -> ArrayScalarLike:
        """Calibrated airspeed [m/s] from trajectory."""
        return self._traj.calibrated_airspeed

    @property
    def vertical_speed_geometric(self) -> ArrayScalarLike:
        """Geometric vertical speed [m/s] from trajectory."""
        return self._traj.vertical_speed_geometric

    @property
    def f(self) -> ArrayScalarLike:
        """Energy share factor [-] from trajectory."""
        return self._traj.f

    @property
    def acceleration(self) -> ArrayScalarLike:
        """Longitudinal acceleration [m/s²] from trajectory."""
        return self._traj.acceleration

    # ---- Flight envelope (delegated to EnvelopeChecker) ----

    @cached_property
    def _envelope_checker(self) -> EnvelopeChecker:
        """Lazily create the EnvelopeChecker with converted flight state."""
        return EnvelopeChecker(
            model=self._model,
            flight_state=self,
            envelope_mode=self._envelope_mode,
        )

    @property
    def cas_kt(self) -> ArrayScalarLike:
        """Calibrated airspeed in knots (for EnvelopeChecker protocol)."""
        from openleaf.performance.physics.units import MS_TO_KT

        return self.calibrated_airspeed * MS_TO_KT

    @property
    def altitude_ft(self) -> ArrayScalarLike:
        """Pressure altitude in feet (for EnvelopeChecker protocol)."""
        from openleaf.performance.physics.units import M_TO_FT

        return self._traj["altitude"] * M_TO_FT

    @property
    def mass(self) -> ArrayScalarLike:
        """Aircraft mass in kg (for EnvelopeChecker protocol)."""
        return self.aircraft_mass

    @property
    def mach(self) -> ArrayScalarLike:
        """Mach number (for EnvelopeChecker protocol)."""
        return self.mach_number

    @cached_property
    def max_cas(self) -> ArrayScalarLike:
        """Maximum calibrated airspeed [kt]."""
        return self._envelope_checker.max_cas

    @cached_property
    def min_cas(self) -> ArrayScalarLike:
        """Minimum calibrated airspeed (stall speed) [kt]."""
        return self._envelope_checker.min_cas

    @cached_property
    def max_mach(self) -> ArrayScalarLike:
        """Maximum Mach number [-]."""
        return self._envelope_checker.max_mach

    @cached_property
    def max_altitude(self) -> ArrayScalarLike:
        """Maximum operating altitude [ft]."""
        return self._envelope_checker.max_altitude

    @property
    def envelope_mode(self) -> EnvelopeMode:
        """Current envelope violation handling mode."""
        return self._envelope_checker.envelope_mode

    @cached_property
    def envelope_violations(self) -> dict[str, ArrayScalarLike]:
        """Detect flight envelope violations (pure computation, no side effects).

        Returns:
            Dictionary with violation flags.
        """
        return self._envelope_checker.envelope_violations

    def check_envelope(self) -> dict[str, ArrayScalarLike]:
        """Check envelope violations and handle according to envelope_mode.

        Returns:
            Dictionary with violation flags.

        Raises:
            EnvelopeViolationError: If envelope_mode is RAISE and violations exist.
        """
        return self._envelope_checker.check_envelope()


def _create_model(model_type: str, icao: str) -> AircraftModel:
    """Factory function to create aircraft models.

    Args:
        model_type: Model type string ("BADA3", "BADA4", "OpenAP", "PS").
        icao: ICAO aircraft type code.

    Returns:
        Instantiated aircraft model.

    Raises:
        ValueError: If model_type is unknown.
    """
    from openleaf.performance.core.factory import ModelFactory
    from openleaf.performance.core.factory import ModelType

    # Normalize model type string
    type_map = {
        "BADA3": ModelType.BADA3,
        "BADA4": ModelType.BADA4,
        "OPENAP": ModelType.OPENAP,
        "PS": ModelType.PS,
    }

    key = model_type.upper()
    if key not in type_map:
        valid = list(type_map.keys())
        raise ValueError(f"Unknown model type: {model_type}. Valid types: {valid}")

    return ModelFactory.load(icao, model_type=type_map[key])
