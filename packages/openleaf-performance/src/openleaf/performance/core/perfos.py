"""
Core performance model interfaces and data structures for aircraft simulation.

This module defines the abstract base classes and enums for aircraft performance models,
including engine and aerodynamic models. It provides the interface that concrete
implementations (such as BADA3 and BADA4) must follow.
"""

from __future__ import annotations


__all__ = [
    "AerodynamicsData",
    "AerodynamicsModel",
    "AerodynamicsModelProtocol",
    "AeroDataT",
    "AircraftData",
    "AircraftModel",
    "AircraftModelProtocol",
    "EngineData",
    "EngineDataT",
    "EngineModel",
    "EngineModelProtocol",
    "EngineSetting",
    "EngineType",
    "FlightPhase",
    "LimitsData",
    "LimitsDataT",
    "LimitsModel",
    "LimitsModelProtocol",
]

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ConfigDict

# Protocols and shared types now live in openleaf-core; re-export here for
# backward compatibility with any code that imported them from this module.
from openleaf.core.protocols import AerodynamicsModelProtocol
from openleaf.core.protocols import AircraftModelProtocol
from openleaf.core.protocols import EngineModelProtocol
from openleaf.core.protocols import LimitsModelProtocol
from openleaf.core.types import ArrayScalarLike
from openleaf.core.types import KeyedData
from openleaf.performance.core.types import EngineSetting
from openleaf.performance.core.types import EngineType
from openleaf.performance.core.types import FlightPhase
from openleaf.performance.physics import aero
from openleaf.performance.physics import isa


class EngineData(BaseModel):
    """Base class for engine data. To be subclassed by specific engine models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AerodynamicsData(BaseModel):
    """Aerodynamic data for an aircraft."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    surface_area: float


class LimitsData(BaseModel):
    """Base aircraft limits data container."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Mass limits
    oem: float  # Operating empty mass [kg]
    mtom: float  # Maximum takeoff mass [kg]
    mpl: float  # Maximum payload [kg]

    # Speed and altitude limits
    vmo: float
    mmo: float | None = None  # Optional for some aircraft (e.g., piston)
    hmo: float


# TypeVars for generic model classes - allows subclasses to specify their data types
EngineDataT = TypeVar("EngineDataT")
AeroDataT = TypeVar("AeroDataT")
LimitsDataT = TypeVar("LimitsDataT")


class AircraftData(BaseModel):
    """Static aircraft data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    wtc: str
    engine_type: EngineType


# --- Protocols for Structural Typing ---
# The canonical Protocol definitions now live in openleaf-core
# (openleaf.core.protocols). They are re-imported above so downstream code
# that does ``from openleaf.performance.core.perfos import EngineModelProtocol``
# continues to work unchanged.


class LimitsModel(Generic[LimitsDataT]):
    """Aircraft limitations with computed methods.

    Generic over the limits data type to allow subclasses to specify their
    own data container while maintaining type safety.

    Example:
        >>> class BADA4Limits(LimitsModel[BADA4LimitsData]):
        ...     @property
        ...     def data(self) -> BADA4LimitsData:
        ...         return self._bada4_data
    """

    def __init__(self, data: LimitsDataT):
        """Initialize aircraft limits.

        Args:
            data: Limits data container with vmo, mmo, hmo.
        """
        # Use object.__setattr__ to allow subclasses to override 'data' as property
        object.__setattr__(self, "_limits_data", data)

    @property
    def data(self) -> LimitsDataT:
        """The underlying limits data container."""
        return self._limits_data

    # Convenience properties for direct access to common limits
    @property
    def mtom(self) -> float | None:
        """Maximum takeoff mass [kg]."""
        return getattr(self.data, "mtom", None)

    @property
    def oem(self) -> float | None:
        """Operating empty mass [kg]."""
        return getattr(self.data, "oem", None)

    @property
    def mpl(self) -> float | None:
        """Maximum payload [kg]."""
        return getattr(self.data, "mpl", None)

    @property
    def vmo(self) -> float:
        """Maximum operating calibrated airspeed [kt]."""
        return self.data.vmo

    @property
    def mmo(self) -> float | None:
        """Maximum operating Mach number, None if not applicable."""
        return self.data.mmo

    @property
    def hmo(self) -> float:
        """Maximum operating altitude [ft]."""
        return self.data.hmo

    def min_cas(self, data: KeyedData) -> ArrayScalarLike:  # pylint: disable=unused-argument
        """Get minimum calibrated airspeed."""
        return 0.0

    def min_mach(self, data: KeyedData) -> ArrayScalarLike:  # pylint: disable=unused-argument
        """Get minimum mach number."""
        return 0.0

    def max_cas(self, data: KeyedData) -> ArrayScalarLike:  # pylint: disable=unused-argument
        """Get maximum calibrated airspeed."""
        return self.data.vmo

    def max_mach(self, data: KeyedData) -> ArrayScalarLike:  # pylint: disable=unused-argument
        """Get maximum Mach number."""
        return self.data.mmo

    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:  # pylint: disable=unused-argument
        """Get maximum operating altitude."""
        return self.data.hmo


# --- Subsystem Interfaces ---


class EngineModel(ABC, Generic[EngineDataT]):
    """Abstract base class for aircraft engine models.

    Generic over the engine data type to allow subclasses to specify their
    own data container while maintaining type safety.

    Example:
        >>> class BADA4JetEngineModel(EngineModel[BADA4JetEngineData]):
        ...     pass
    """

    def __init__(self, data: EngineDataT):
        """Initialize engine model with engine data.

        Args:
            data: Engine data containing parameters specific to the engine model.
        """
        self.data = data

    # Abstract methods
    @abstractmethod
    def thrust_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum available thrust.

        Args:
            data: Dictionary containing flight state data.

        Returns:
            Minimum available thrust in Newtons.
        """

    @abstractmethod
    def thrust_max(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate maximum available thrust.

        Args:
            data: Dictionary containing flight state data.

        Returns:
            Maximum available thrust in Newtons.
        """

    @abstractmethod
    def fuel_flow(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate fuel flow rate at a given thrust.

        Args:
            data: Dictionary containing flight state data including thrust.

        Returns:
            Fuel flow rate in kg/s.
        """

    @abstractmethod
    def fuel_flow_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum fuel flow rate.

        Args:
            data: Dictionary containing flight state data.

        Returns:
            Minimum fuel flow rate in kg/s.
        """


class AerodynamicsModel(ABC, Generic[AeroDataT]):
    """Abstract base class for aerodynamic models.

    Generic over the aerodynamic data type to allow subclasses to specify their
    own data container while maintaining type safety.

    Example:
        >>> class BADA4AeroModel(AerodynamicsModel[BADA4AeroData]):
        ...     pass
    """

    def __init__(self, data: AeroDataT):
        """Initialize aerodynamic model with aerodynamic data.

        Args:
            data: Aerodynamic data containing wing reference area.
        """
        self.data = data

    # Quick access to basic, generic, data
    @property
    def surface_area(self) -> float:
        """Wing reference area in square meters."""
        return self.data.surface_area

    @abstractmethod
    def drag_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate drag coefficient from flight state data.

        Args:
            data: Dictionary containing flight state data.

        Returns:
            Drag coefficient (dimensionless).
        """


# --- Base AircraftModel ---


class AircraftModel(ABC):
    """Abstract base class for aircraft performance models.

    Combines engine and aerodynamic models to provide complete aircraft
    performance characteristics.
    """

    def __init__(
        self,
        data: AircraftData,
        engine: EngineModel,
        aerodynamics: AerodynamicsModel,
        limits: LimitsModel,
    ):
        """Initialize aircraft model with data and subsystems.

        Args:
            data: Aircraft data containing specifications.
            engine: Engine model for thrust and fuel flow calculations.
            aerodynamics: Aerodynamic model for drag calculations.
            limits: Aircraft mass, speed and altitude limits (vmo, mmo, hmo).
        """
        self.data = data
        self.engine = engine
        self.aerodynamics = aerodynamics
        self.limits = limits

    # Abstract init from file
    @classmethod
    @abstractmethod
    def from_file(cls, file: str, **kwargs: Any) -> AircraftModel:
        """Load aircraft model from file.

        Args:
            file: Path to aircraft data file.
            **kwargs: Additional arguments passed to the parser.

        Returns:
            Initialized aircraft model.
        """

    # Quick access to basic data
    @property
    def model(self) -> str:
        """Aircraft model name."""
        return self.data.model

    @property
    def wtc(self) -> str:
        """Wake turbulence category."""
        return self.data.wtc

    @property
    def engine_type(self) -> EngineType:
        """Type of engine (jet, turboprop, or piston)."""
        return self.data.engine_type

    @property
    def vmo(self) -> float:
        """Maximum operating speed in knots."""
        return self.limits.vmo

    @property
    def hmo(self) -> float:
        """Maximum operating altitude in feet."""
        return self.limits.hmo

    @property
    def mmo(self) -> float:
        """Maximum operating Mach number."""
        return self.limits.mmo

    @property
    def oem(self) -> float:
        """Operating empty mass in kilograms."""
        return self.limits.oem

    @property
    def mtom(self) -> float:
        """Maximum takeoff mass in kilograms."""
        return self.limits.mtom

    @property
    def mpl(self) -> float:
        """Maximum payload mass in kilograms."""
        return self.limits.mpl

    @property
    def surface_area(self) -> float:
        """Wing reference area in square meters."""
        return self.aerodynamics.surface_area

    # --- Abstract Methods ---

    def high_lift(self, data: KeyedData) -> ArrayScalarLike | None:
        """Determine high-lift device setting from flight state.

        Returns the current flap/slat configuration. The return type and
        semantics are model-dependent, reflecting each model's native
        representation:

        - BADA4: HLid string from parsed XML AeroConfSchedule (e.g., "0", "1").
                 Raises ValueError if AeroConfSchedule not available in XML.
        - BADA3: Returns None. BADA3 OPF/APF files do not define HLid.
                 Use aerodynamic_configuration() for BADA3-specific data.
        - OpenAP: Flap angle in degrees (0, 5, or 20).
        - PS: Returns 0 (cruise-only model, no high-lift modeling).

        Args:
            data: Flight state data dictionary.

        Returns:
            High-lift device setting (scalar or array), or None if not modeled.
        """
        return None

    def landing_gear(self, data: KeyedData) -> ArrayScalarLike:
        """Determine landing gear state from flight state.

        Default implementation returns False (gear retracted).
        Model-specific behavior:

        - BADA4: Boolean from parsed XML AeroConfSchedule LG mapping.
        - BADA3: Boolean derived from aerodynamic configuration (True = LD).
        - OpenAP: Boolean based on altitude and descent conditions.
        - PS: Returns False (cruise-only, gear always retracted).

        Args:
            data: Flight state data dictionary.

        Returns:
            Boolean (True = gear extended, False = retracted).
        """
        return False

    def aerodynamic_configuration(self, data: KeyedData) -> ArrayScalarLike | None:
        """Determine aerodynamic configuration from flight state.

        Default implementation returns None (cruise-only or not modeled).
        This enables trajectory simulation with the flow:
            altitude/speed → configuration → hlid/gear → drag_coef

        Model-specific behavior:
        - BADA3: AerodynamicConfig enum based on altitude and speed thresholds
        - BADA4: AerodynamicConfig enum from ARPM configuration schedule
        - OpenAP: String config name based on flight phase detection
        - PS: None (cruise-only model, no configuration rules)

        Args:
            data: Flight state data dictionary containing:
                - altitude: Altitude [m] or altitude_agl for ground-relative
                - calibrated_airspeed or true_airspeed: Speed [m/s or kt]
                - aircraft_mass: Mass for stall speed calculation [kg]
                - flight_phase: Optional flight phase hint

        Returns:
            AerodynamicConfig enum, string, or None if not modeled.
        """
        return None

    # --- Core Performance Methods (KeyedData interface) ---

    def lift_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate lift coefficient from flight state.

        Uses aero.lift_coefficient with aircraft wing area.

        Args:
            data: Flight state dictionary with keys:
                - aircraft_mass: Aircraft mass [kg]
                - true_airspeed: True airspeed [m/s]
                - air_density: Air density [kg/m³]
                - load_factor: Load factor [-] (optional, default 1.0)

        Returns:
            Lift coefficient (dimensionless).
        """
        mass = data["aircraft_mass"]
        v = data["true_airspeed"]
        rho = data["air_density"]
        load_factor = data.get("load_factor", 1.0)

        q = aero.dynamic_pressure(v, rho)
        lift = mass * isa.G * load_factor

        return aero.aerodynamic_coefficient(lift, q, self.aerodynamics.surface_area)

    def lift_coef_from_inputs(
        self,
        v: ArrayScalarLike,
        m: ArrayScalarLike,
        sigma: ArrayScalarLike,
        load_factor: ArrayScalarLike = 1.0,
    ) -> ArrayScalarLike:
        """Calculate lift coefficient from explicit inputs.

        Convenience method using air density ratio (sigma) instead of density.

        Args:
            v: True airspeed [m/s]
            m: Aircraft mass [kg]
            sigma: Air density ratio (rho/rho_SL) [-]
            load_factor: Load factor [-] (default 1.0)

        Returns:
            Lift coefficient (dimensionless).
        """
        rho = sigma * isa.RHO_SL  # Convert to actual density
        q = aero.dynamic_pressure(v, rho)
        lift = m * isa.G * load_factor

        return aero.aerodynamic_coefficient(lift, q, self.aerodynamics.surface_area)

    def drag_from_inputs(
        self, v: ArrayScalarLike, sigma: ArrayScalarLike, cd: ArrayScalarLike
    ) -> ArrayScalarLike:
        """Calculate drag force from explicit inputs.

        Convenience method using air density ratio (sigma) instead of density.

        Args:
            v: True airspeed [m/s]
            sigma: Air density ratio (rho/rho_SL) [-]
            cd: Drag coefficient [-]

        Returns:
            Drag force [N].
        """
        rho = sigma * isa.RHO_SL  # Convert to actual density
        q = aero.dynamic_pressure(v, rho)

        return aero.aerodynamic_force(cd, q, self.aerodynamics.surface_area)

    def drag(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate aerodynamic drag force from flight state.

        Uses aero.aerodynamic_drag with aircraft wing area.

        Args:
            data: Flight state dictionary with keys:
                - drag_coef: Drag coefficient [-]
                - true_airspeed: True airspeed [m/s]
                - air_density: Air density [kg/m³]

        Returns:
            Drag force [N].
        """
        cd = data["drag_coef"]
        v = data["true_airspeed"]
        rho = data["air_density"]

        q = aero.dynamic_pressure(v, rho)

        return aero.aerodynamic_force(cd, q, self.aerodynamics.surface_area)

    def thrust_required(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate required thrust for flight condition.

        Uses aero.thrust_required with energy share factor formulation.

        Args:
            data: Flight state dictionary with keys:
                - drag: Drag force [N]
                - aircraft_mass: Aircraft mass [kg]
                - true_airspeed: True airspeed [m/s]
                - vertical_speed: Rate of climb [m/s]
                - f: Energy share factor [-]

        Returns:
            Required thrust [N].
        """
        drag = data["drag"]
        mass = data["aircraft_mass"]
        v = data["true_airspeed"]
        vs = data["vertical_speed"]
        f = data["f"]

        return aero.thrust_required(drag, mass, v, vs, f)

    # --- Limits Delegation Methods ---
    def min_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum mach number.

        Args:
            data: Flight state data dictionary.

        Returns:
            Minimum calibrated airspeed in knots.
        """
        return self.limits.min_mach(data)

    def min_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum calibrated airspeed.

        Args:
            data: Flight state data dictionary.

        Returns:
            Minimum calibrated airspeed in knots.
        """
        return self.limits.min_cas(data)

    def max_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum calibrated airspeed.

        Args:
            data: Flight state data dictionary.

        Returns:
            Maximum calibrated airspeed in knots.
        """
        return self.limits.max_cas(data)

    def max_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum Mach number.

        Args:
            data: Flight state data dictionary.

        Returns:
            Maximum Mach number.
        """
        return self.limits.max_mach(data)

    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum operating altitude.

        Args:
            data: Flight state data dictionary.

        Returns:
            Maximum operating altitude in feet.
        """
        return self.limits.max_altitude(data)

    # --- Aerodynamics Delegation Methods ---

    def drag_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate drag coefficient from flight state.

        Delegates to self.aerodynamics.drag_coef(data).

        Args:
            data: Flight state data dictionary.

        Returns:
            Drag coefficient (dimensionless).
        """
        return self.aerodynamics.drag_coef(data)

    # --- Engine Delegation Methods ---

    def thrust_min(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum available thrust.

        Args:
            data: Flight state data dictionary.

        Returns:
            Minimum thrust in N.
        """
        return self.engine.thrust_min(data)

    def thrust_max(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum available thrust.

        Args:
            data: Flight state data dictionary.

        Returns:
            Maximum thrust in N.
        """
        return self.engine.thrust_max(data)

    def fuel_flow(self, data: KeyedData) -> ArrayScalarLike:
        """Get fuel flow at current thrust.

        Args:
            data: Flight state data dictionary including thrust.

        Returns:
            Fuel flow rate in kg/s.
        """
        return self.engine.fuel_flow(data)

    def fuel_flow_min(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum fuel flow.

        Args:
            data: Flight state data dictionary.

        Returns:
            Minimum fuel flow rate in kg/s.
        """
        return self.engine.fuel_flow_min(data)
