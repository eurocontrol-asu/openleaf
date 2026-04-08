"""Poll-Schumann (PS) aircraft performance models.

Provides engine, aerodynamic, and aircraft models for PS aircraft
performance calculations including thrust, fuel flow, and drag.

Usage:
    # Initialize from ICAO code (requires pycontrails)
    model = PSModel("A320")

    # Initialize from pre-loaded data
    model = PSModel.from_data(ps_data)

    # List available aircraft
    aircraft = PSModel.available_aircraft()

Reference:
    Poll, D.I.A., and Schumann, U. (2021). "An estimation method for the fuel burn
    and other performance characteristics of civil transport aircraft in the cruise."
    The Aeronautical Journal, 125(1284), 550-577.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.core.perfos import AerodynamicsModel
from openleaf.performance.core.perfos import AircraftModel
from openleaf.performance.core.perfos import EngineModel
from openleaf.performance.core.validation import requires
from openleaf.performance.logging import get_logger
from openleaf.performance.models.ps.limits import PSLimits
from openleaf.performance.models.ps.parsing import load_from_pycontrails
from openleaf.performance.models.ps.ps_physics import airframe_drag_coefficient
from openleaf.performance.models.ps.ps_physics import max_available_thrust_coefficient
from openleaf.performance.models.ps.ps_physics import oswald_efficiency_factor
from openleaf.performance.models.ps.ps_physics import overall_propulsion_efficiency
from openleaf.performance.models.ps.ps_physics import reynolds_number
from openleaf.performance.models.ps.ps_physics import skin_friction_coefficient
from openleaf.performance.models.ps.ps_physics import thrust_coefficient_at_max_efficiency
from openleaf.performance.models.ps.ps_physics import wave_drag_coefficient
from openleaf.performance.models.ps.ps_physics import zero_lift_drag_coefficient
from openleaf.performance.models.ps.schemas import PSAerodynamicsData
from openleaf.performance.models.ps.schemas import PSAircraftData
from openleaf.performance.models.ps.schemas import PSData
from openleaf.performance.models.ps.schemas import PSEngineData
from openleaf.performance.models.ps.schemas import PSLimitsData
from openleaf.performance.physics import isa
from openleaf.performance.physics import units


logger = get_logger(__name__)

if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = [
    "PSAeroModel",
    "PSEngineModel",
    "PSLimits",
    "PSModel",
]

# Physical constants (from units module)
# Using units.GAMMA for adiabatic index, units.R_AIR for gas constant


# --- Aerodynamics Model ---


class PSAeroModel(AerodynamicsModel[PSAerodynamicsData]):
    """Poll-Schuman aerodynamic model.

    Calculates drag using Reynolds number-based skin friction,
    induced drag with Oswald efficiency, and compressibility wave drag.

    Unlike simpler drag models, PS model requires atmospheric state
    (temperature, pressure) for Reynolds number calculations. When
    these are provided via drag_coef(data), the model produces more
    accurate drag coefficients.
    """

    @requires("lift_coef", "mach_number", "air_temperature", "air_pressure")
    def drag_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate drag coefficient from flight state data.

        Extracts lift coefficient, Mach number, and atmospheric state
        from the data dict. Unlike simpler models, PS requires
        air_temperature and air_pressure for Reynolds number effects.

        IMPORTANT: Poll-Schumann model is designed for cruise flight only.
        Non-clean configurations (takeoff, approach, landing) are not
        supported by the reference implementation.

        Args:
            data: Dictionary containing:
                - lift_coef: Lift coefficient (dimensionless)
                - mach_number: Mach number (dimensionless)
                - air_temperature: Air temperature [K]
                - air_pressure: Air pressure [Pa]

        Returns:
            Drag coefficient (dimensionless).

        Raises:
            MissingFieldError: If required fields are missing.

        Reference:
            Poll & Schumann (2021), "An estimation method for the fuel burn
            and other performance characteristics of civil transport aircraft
            in the cruise." The Aeronautical Journal, 125(1284), 550-577.
        """
        # PS model is cruise-only - compute clean configuration drag
        return self._drag_coef_clean(
            cl=data["lift_coef"],
            mach=data["mach_number"],
            air_temperature=data["air_temperature"],
            air_pressure=data["air_pressure"],
        )

    # NOTE: PS model is cruise-only. The reference implementation (Poll & Schumann 2021,
    # pycontrails) does not define drag coefficients for non-clean configurations
    # (takeoff, approach, landing). No invented drag increments are provided here.

    def _drag_coef_clean(
        self,
        *,
        cl: ArrayScalarLike,
        mach: ArrayScalarLike,
        air_temperature: ArrayScalarLike = None,
        air_pressure: ArrayScalarLike = None,
    ) -> ArrayScalarLike:
        """Calculate drag coefficient for clean configuration.

        Note: If air_temperature and air_pressure are not provided,
        uses ISA conditions at FL350 as reference.
        """
        # Reynolds number and skin friction
        rn = reynolds_number(
            self.data.surface_area,
            mach,
            air_temperature,
            air_pressure,
        )
        c_f = skin_friction_coefficient(rn)

        # Zero-lift drag
        c_drag_0 = zero_lift_drag_coefficient(c_f, self.data.psi_0)

        # Oswald efficiency factor
        e_ls = oswald_efficiency_factor(
            c_drag_0,
            self.data.wing_aspect_ratio,
            self.data.cos_sweep,
            self.data.delta_2,
            self.data.winglets,
        )

        # Wave drag
        c_drag_w = wave_drag_coefficient(
            mach,
            cl,
            self.data.cos_sweep,
            self.data.wing_constant,
            self.data.j_1,
            self.data.j_2,
            self.data.j_3,
            self.data.x_ref,
        )

        # Total drag
        return airframe_drag_coefficient(
            c_drag_0,
            c_drag_w,
            cl,
            e_ls,
            self.data.wing_aspect_ratio,
        )

    def drag_coef_with_state(
        self,
        *,
        cl: ArrayScalarLike,
        mach: ArrayScalarLike,
        air_temperature: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate drag coefficient using full atmospheric state.

        This is the preferred method when atmospheric conditions are known.

        Args:
            cl: Lift coefficient.
            mach: Mach number.
            air_temperature: Air temperature [K].
            air_pressure: Air pressure [Pa].

        Returns:
            Drag coefficient.
        """
        return self._drag_coef_clean(
            cl=cl,
            mach=mach,
            air_temperature=air_temperature,
            air_pressure=air_pressure,
        )


# --- Engine Model ---


class PSEngineModel(EngineModel[PSEngineData]):
    """Poll-Schuman engine model.

    Calculates thrust and fuel flow using propulsion efficiency
    and thrust coefficient correlations.
    """

    def __init__(self, data: PSEngineData, aero_data: PSAerodynamicsData):
        """Initialize engine model.

        Args:
            data: Engine data.
            aero_data: Aerodynamic data (needed for wing area).
        """
        super().__init__(data)
        self.aero_data = aero_data

    def thrust_coefficient(
        self,
        thrust: ArrayScalarLike,
        mach_num: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate engine thrust coefficient.

        Args:
            thrust: Total thrust [N].
            mach_num: Mach number.
            air_pressure: Air pressure [Pa].

        Returns:
            Thrust coefficient.
        """
        return thrust / (
            0.5 * units.GAMMA * air_pressure * mach_num**2 * self.aero_data.surface_area
        )

    def thrust_from_coefficient(
        self,
        c_t: ArrayScalarLike,
        mach_num: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate thrust from thrust coefficient.

        Args:
            c_t: Thrust coefficient.
            mach_num: Mach number.
            air_pressure: Air pressure [Pa].

        Returns:
            Thrust [N].
        """
        return c_t * 0.5 * units.GAMMA * air_pressure * mach_num**2 * self.aero_data.surface_area

    @requires("mach_number", "air_temperature", "air_pressure")
    def thrust_max(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate maximum available thrust.

        Args:
            data: Flight state containing mach_number, air_temperature, air_pressure.

        Returns:
            Maximum thrust [N].

        Raises:
            MissingFieldError: If required fields are missing.
        """
        return self._thrust_max_impl(
            mach=data["mach_number"],
            air_temperature=data["air_temperature"],
            air_pressure=data["air_pressure"],
        )

    def _thrust_max_impl(
        self,
        mach: ArrayScalarLike,
        air_temperature: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate maximum available thrust (typed implementation).

        Args:
            mach: Mach number (dimensionless).
            air_temperature: Air temperature [K].
            air_pressure: Air pressure [Pa].

        Returns:
            Maximum thrust [N].
        """
        # Thrust coefficient at max efficiency
        c_t_eta_b = thrust_coefficient_at_max_efficiency(mach, self.data.m_des, self.data.c_t_des)

        # Max available thrust coefficient
        c_t_max = max_available_thrust_coefficient(
            air_temperature,
            mach,
            c_t_eta_b,
            self.data.tet_mcc,
            self.data.tr_ec,
            self.data.m_ec,
            buffer=self.data.thrust_buffer,
        )

        return self.thrust_from_coefficient(c_t_max, mach, air_pressure)

    @requires("mach_number")
    def thrust_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum thrust (idle).

        Args:
            data: Flight state containing mach_number.

        Returns:
            Minimum thrust [N] (approximately zero for idle).

        Raises:
            MissingFieldError: If required fields are missing.
        """
        return self._thrust_min_impl(mach=data["mach_number"])

    def _thrust_min_impl(self, mach: ArrayScalarLike) -> ArrayScalarLike:
        """Calculate minimum thrust (typed implementation).

        Args:
            mach: Mach number (used to maintain array shape).

        Returns:
            Minimum thrust [N] (approximately zero for idle).
        """
        # Idle thrust is approximately zero
        return 0.0 * mach  # Maintain array shape

    @requires("thrust", "mach_number", "air_temperature", "air_pressure", "altitude")
    def fuel_flow(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate fuel flow rate.

        Args:
            data: Flight state containing thrust, mach_number, air_temperature,
                air_pressure, and altitude.

        Returns:
            Fuel flow rate [kg/s].

        Raises:
            MissingFieldError: If required fields are missing.
        """
        return self._fuel_flow_impl(
            thrust=data["thrust"],
            mach=data["mach_number"],
            air_temperature=data["air_temperature"],
            air_pressure=data["air_pressure"],
            altitude=data["altitude"],
        )

    def _fuel_flow_impl(
        self,
        thrust: ArrayScalarLike,
        mach: ArrayScalarLike,
        air_temperature: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
        altitude: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate fuel flow rate (typed implementation).

        Args:
            thrust: Thrust [N].
            mach: Mach number (dimensionless).
            air_temperature: Air temperature [K].
            air_pressure: Air pressure [Pa].
            altitude: Altitude [m].

        Returns:
            Fuel flow rate [kg/s].
        """
        # Thrust coefficient
        c_t = self.thrust_coefficient(thrust, mach, air_pressure)

        # Thrust coefficient at max efficiency
        c_t_eta_b = thrust_coefficient_at_max_efficiency(mach, self.data.m_des, self.data.c_t_des)

        # Clamp c_t to available range
        c_t_max = max_available_thrust_coefficient(
            air_temperature,
            mach,
            c_t_eta_b,
            self.data.tet_mcc,
            self.data.tr_ec,
            self.data.m_ec,
            buffer=self.data.thrust_buffer,
        )
        c_t = np.clip(c_t, 0.0, c_t_max)

        # Overall propulsion efficiency
        eta = overall_propulsion_efficiency(
            mach,
            c_t,
            c_t_eta_b,
            self.data.m_des,
            self.data.eta_1,
            self.data.eta_2,
        )

        # Avoid division by zero
        eta = np.maximum(eta, 0.01)

        # Fuel flow
        fuel_flow = (
            (units.GAMMA / 2)
            * (c_t * mach**3 / eta)
            * np.sqrt(units.GAMMA * units.R_AIR * air_temperature)
            * air_pressure
            * self.aero_data.surface_area
            / units.Q_FUEL
        )

        # Apply min/max corrections
        return self._correct_fuel_flow_impl(
            fuel_flow=fuel_flow,
            altitude=altitude,
            air_temperature=air_temperature,
            air_pressure=air_pressure,
            mach=mach,
        )

    def _correct_fuel_flow_impl(
        self,
        fuel_flow: ArrayScalarLike,
        altitude: ArrayScalarLike,
        air_temperature: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Apply fuel flow corrections (typed implementation).

        Args:
            fuel_flow: Uncorrected fuel flow [kg/s].
            altitude: Altitude [m].
            air_temperature: Air temperature [K].
            air_pressure: Air pressure [Pa].
            mach: Mach number (dimensionless).

        Returns:
            Corrected fuel flow [kg/s].
        """
        altitude_ft = altitude * units.M_TO_FT

        # Minimum fuel flow (idle)
        ff_min = self._fuel_flow_idle(altitude_ft)

        # Maximum fuel flow
        theta = air_temperature / isa.TAU_SL
        delta = air_pressure / isa.P_SL
        ff_max = self._equivalent_fuel_flow_at_cruise(theta, delta, mach)

        return np.clip(fuel_flow, ff_min, ff_max)

    def _fuel_flow_idle(self, altitude_ft: ArrayScalarLike) -> ArrayScalarLike:
        """Calculate minimum fuel flow at idle.

        Args:
            altitude_ft: Altitude in feet.

        Returns:
            Idle fuel flow [kg/s].
        """
        x = altitude_ft / 10000.0
        return self.data.ff_idle_sls * (1.0 - 0.178 * x + 0.0085 * x**2)

    def _equivalent_fuel_flow_at_cruise(
        self,
        theta: ArrayScalarLike,
        delta: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate equivalent maximum fuel flow at cruise conditions.

        Args:
            theta: Temperature ratio.
            delta: Pressure ratio.
            mach: Mach number.

        Returns:
            Maximum fuel flow [kg/s].
        """
        denom = (np.power(theta, 3.8) / delta) * np.exp(0.2 * mach**2)
        denom = np.maximum(denom, 1.0)
        return self.data.ff_max_sls / denom

    @requires("altitude")
    def fuel_flow_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum fuel flow rate.

        Args:
            data: Flight state containing altitude.

        Returns:
            Minimum fuel flow [kg/s].

        Raises:
            MissingFieldError: If altitude is missing.
        """
        return self._fuel_flow_min_impl(altitude=data["altitude"])

    def _fuel_flow_min_impl(self, altitude: ArrayScalarLike) -> ArrayScalarLike:
        """Calculate minimum fuel flow (typed implementation).

        Args:
            altitude: Altitude [m].

        Returns:
            Minimum fuel flow [kg/s].
        """
        altitude_ft = altitude * units.M_TO_FT
        logger.debug("Computing fuel_flow_min")
        return self._fuel_flow_idle(altitude_ft)


# --- Full Aircraft Model ---


class PSModel(AircraftModel):
    """Poll-Schuman aircraft performance model.

    Combines PS aerodynamic and engine models for complete
    aircraft performance calculations.

    Attributes:
        engine: PS engine model.
        aero: PS aerodynamic model.
        data: PS aircraft data.
        limits: PS operational limits (Mach constraints by altitude).
    """

    engine: PSEngineModel
    data: PSAircraftData

    def __init__(self, icao: str, *, thrust_buffer: float = 0.0):
        """Initialize PS model from ICAO aircraft type code.

        Requires pycontrails to be installed for aircraft parameters.

        Args:
            icao: ICAO aircraft type code (e.g., 'A320', 'B738').
            thrust_buffer: Safety buffer for max thrust coefficient (default 0.0).
                Use 0.0 for thrust consistent with BADA MCC rating.
                Use 0.20 for pycontrails' conservative upper bound.

        Raises:
            ImportError: If pycontrails is not installed.
            KeyError: If aircraft not found in pycontrails database.

        Example:
            >>> model = PSModel("A320")
            >>> model.mtom
            78000.0
            >>> model_conservative = PSModel("A320", thrust_buffer=0.20)
        """
        logger.info("Loading PS model for %s", icao)
        # Use load_from_pycontrails to get PS data, then initialize
        ps_data = load_from_pycontrails(icao)
        self._init_from_data(ps_data, thrust_buffer=thrust_buffer)

    def _init_from_data(self, ps_data: PSData, *, thrust_buffer: float = 0.0) -> None:
        """Initialize model from PS data object.

        Args:
            ps_data: Complete PS model data.
            thrust_buffer: Safety buffer for max thrust coefficient.
        """
        logger.debug("Initializing PS aero and engine models")
        aero = PSAeroModel(ps_data.aero)

        # Create engine data with specified thrust_buffer
        engine_data = PSEngineData(
            n_engines=ps_data.engine.n_engines,
            m_des=ps_data.engine.m_des,
            c_t_des=ps_data.engine.c_t_des,
            eta_1=ps_data.engine.eta_1,
            eta_2=ps_data.engine.eta_2,
            tet_mcc=ps_data.engine.tet_mcc,
            tr_ec=ps_data.engine.tr_ec,
            m_ec=ps_data.engine.m_ec,
            ff_idle_sls=ps_data.engine.ff_idle_sls,
            ff_max_sls=ps_data.engine.ff_max_sls,
            p_i_max=ps_data.engine.p_i_max,
            p_inf_co=ps_data.engine.p_inf_co,
            thrust_buffer=thrust_buffer,
        )
        engine = PSEngineModel(engine_data, ps_data.aero)

        # Initialize operational limits
        limits_data = PSLimitsData(
            oem=ps_data.aircraft.oem,
            mtom=ps_data.aircraft.mtom,
            mpl=ps_data.aircraft.mpl,
            vmo=ps_data.aircraft.vmo,
            mmo=ps_data.aircraft.mmo,
            hmo=ps_data.aircraft.hmo,
            p_i_max=ps_data.engine.p_i_max,
            p_inf_co=ps_data.engine.p_inf_co,
        )
        limits = PSLimits(limits_data)

        super().__init__(ps_data.aircraft, engine, aero, limits)
        self._ps_data = ps_data
        logger.info(
            "PS model initialized: %s (mtom=%.0f kg)",
            ps_data.aircraft.model,
            ps_data.aircraft.mtom,
        )

    # limits inherited from AircraftModel base class

    @classmethod
    def from_data(cls, ps_data: PSData, *, thrust_buffer: float = 0.0) -> PSModel:
        """Create PS model from pre-loaded data.

        Args:
            ps_data: Complete PS model data object.
            thrust_buffer: Safety buffer for max thrust coefficient (default 0.0).

        Returns:
            Initialized PS model.
        """
        instance = object.__new__(cls)
        instance._init_from_data(ps_data, thrust_buffer=thrust_buffer)
        return instance

    @classmethod
    def from_file(cls, file: str, *, thrust_buffer: float = 0.0, **kwargs) -> PSModel:
        """Load PS model from file.

        For PS model, 'file' is treated as an ICAO code since PS uses
        pycontrails database rather than files.

        Args:
            file: ICAO aircraft type code (e.g., 'A320').
            thrust_buffer: Safety buffer for max thrust coefficient (default 0.0).
            **kwargs: Additional arguments (unused).

        Returns:
            Initialized PS model.
        """
        return cls(file, thrust_buffer=thrust_buffer)

    @classmethod
    def from_pycontrails(cls, aircraft_type: str, *, thrust_buffer: float = 0.0) -> PSModel:
        """Create PS model using pycontrails aircraft parameters.

        Requires pycontrails to be installed.

        Note: This is now equivalent to PSModel(aircraft_type).
        Kept for backward compatibility.

        Args:
            aircraft_type: ICAO aircraft type code (e.g., 'A320', 'B738').
            thrust_buffer: Safety buffer for max thrust coefficient (default 0.0).

        Returns:
            Initialized PS model.

        Raises:
            ImportError: If pycontrails is not installed.
        """
        return cls(aircraft_type, thrust_buffer=thrust_buffer)

    @classmethod
    def available_aircraft(cls) -> list[str]:
        """List available PS aircraft types from pycontrails.

        Returns:
            Sorted list of ICAO codes for available aircraft.

        Raises:
            ImportError: If pycontrails is not installed.
        """
        try:
            from pycontrails.models.ps_model.ps_aircraft_params import load_aircraft_engine_params
        except ImportError as e:
            raise ImportError(
                "pycontrails is required for PSModel.available_aircraft(). "
                "Install with: pip install pycontrails"
            ) from e

        aircraft_engine_params = load_aircraft_engine_params()
        return sorted(aircraft_engine_params.keys())

    def high_lift(self, data: KeyedData) -> ArrayScalarLike:
        """Determine high-lift device setting from flight state.

        PS model is primarily designed for cruise analysis.
        Always returns 0 (clean configuration, no flaps).

        Args:
            data: Flight state dictionary.

        Returns:
            0 (clean configuration).
        """
        return 0

    def landing_gear(self, data: KeyedData) -> ArrayScalarLike:
        """Determine landing gear state from flight state.

        PS model is cruise-focused and doesn't model landing gear.
        Always returns False (gear retracted).

        Args:
            data: Flight state dictionary.

        Returns:
            False (gear always retracted in cruise).
        """
        return False

    # --- PS-specific convenience methods ---
    #
    # These methods use the PS-specific formulation with pressure/Mach
    # and climb angle. They are kept for backward compatibility and for
    # cases where these specific inputs are more convenient.
    # For model-agnostic code, use lift_coef(data) and thrust_required(data).

    def lift_coefficient(
        self,
        aircraft_mass: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
        mach_num: ArrayScalarLike,
        climb_angle: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Calculate lift coefficient from pressure and Mach (PS formulation).

        This uses the PS-specific formulation with air pressure and Mach.
        For model-agnostic code, use lift_coef(data) from the base class.

        Args:
            aircraft_mass: Aircraft mass [kg].
            air_pressure: Air pressure [Pa].
            mach_num: Mach number.
            climb_angle: Climb angle [degrees].

        Returns:
            Lift coefficient.
        """
        climb_angle_rad = climb_angle * np.pi / 180.0
        lift_force = aircraft_mass * units.G * np.cos(climb_angle_rad)
        denom = (
            (units.GAMMA / 2.0) * air_pressure * mach_num**2 * self.aerodynamics.data.surface_area
        )
        return lift_force / denom

    def compute_thrust_required(
        self,
        aircraft_mass: ArrayScalarLike,
        c_lift: ArrayScalarLike,
        c_drag: ArrayScalarLike,
        dv_dt: ArrayScalarLike = 0.0,
        climb_angle: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Calculate thrust required using climb angle formulation.

        This is a PS-specific convenience method using climb angle and
        acceleration directly. For model-agnostic code, use thrust_required(data)
        from the base class which uses vertical_speed and energy_share_factor.

        Args:
            aircraft_mass: Aircraft mass [kg].
            c_lift: Lift coefficient.
            c_drag: Drag coefficient.
            dv_dt: Acceleration [m/s**2].
            climb_angle: Climb angle [degrees].

        Returns:
            Required thrust [N].
        """
        theta_rad = climb_angle * np.pi / 180.0
        thrust = (
            (aircraft_mass * units.G * np.cos(theta_rad) * (c_drag / c_lift))
            + (aircraft_mass * units.G * np.sin(theta_rad))
            + aircraft_mass * dv_dt
        )
        return np.maximum(thrust, 0.0)
