"""
Total Energy Model (TEM) for aircraft performance.

The Total Energy Model equates the rate of work done by forces acting on the
aircraft to the rate of increase in potential and kinetic energy.

Fundamental equation (BADA User Manual 3.1-1):
    (T - D) · V_TAS = m · g · (dh/dt) + m · V_TAS · (dV_TAS/dt)

Rearranged with Energy Share Factor f (BADA 3.1-4):
    dh/dt = [(T - D) · V_TAS / (m · g)] · f

IMPORTANT: Pressure vs Geometric Altitude
-------------------------------------------
BADA defines ROCD as the rate of change of PRESSURE altitude (dHp/dt),
not geometric altitude (dh_geo/dt). The temperature correction is:

    dHp/dt = dh_geo/dt · (T_ISA / T_actual)
    dh_geo/dt = dHp/dt · (T_actual / T_ISA)

All functions in this module work with PRESSURE altitude rate (dHp/dt)
unless explicitly noted otherwise.

Reference: EUROCONTROL BADA User Manual, Section 3.1
"""

from __future__ import annotations


__all__ = [
    "DEFAULT_ENERGY_SHARE_FACTOR",
    "TotalEnergyModel",
    "rocd_pressure",
    "thrust_from_rocd",
    "f_from_state",
    "residual",
    "rocd_geometric",
    "temp_correction_factor",
]

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.physics import isa
from openleaf.performance.physics.units import EPSILON_CLAMP
from openleaf.performance.utils.types import ArrayScalarLike


if TYPE_CHECKING:
    from openleaf.performance.physics.esf import SpeedSchedule

# Default energy share factor for level constant-speed flight
DEFAULT_ENERGY_SHARE_FACTOR = 1.0


def temp_correction_factor(
    air_temperature: ArrayScalarLike,
    altitude: ArrayScalarLike,
) -> ArrayScalarLike:
    """Compute temperature correction factor for pressure/geometric altitude conversion.

    The correction factor relates pressure altitude rate to geometric altitude rate
    under non-ISA conditions.

    Args:
        air_temperature: Actual air temperature [K].
        altitude: Geometric altitude [m] (for computing ISA temperature).

    Returns:
        Correction factor T_ISA / T_actual (dimensionless).
        - Factor > 1: Cold conditions (pressure altitude rate > geometric rate)
        - Factor < 1: Hot conditions (pressure altitude rate < geometric rate)
        - Factor = 1: ISA conditions
    """
    T_isa = isa.air_temperature(altitude)
    return T_isa / air_temperature


def rocd_pressure(
    thrust: ArrayScalarLike,
    drag: ArrayScalarLike,
    mass: ArrayScalarLike,
    true_airspeed: ArrayScalarLike,
    f: ArrayScalarLike,
    air_temperature: ArrayScalarLike,
    altitude: ArrayScalarLike,
) -> ArrayScalarLike:
    """Compute rate of climb/descent in PRESSURE altitude (dHp/dt).

    This is the primary TEM equation for computing vertical speed when
    thrust and energy share factor are known.

    From BADA User Manual, the temperature-corrected ROCD formula:
        ROCD = [(T - D) · V_TAS · f / (m · g)] · (T_ISA / T_actual)

    Args:
        thrust: Engine thrust [N].
        drag: Aerodynamic drag [N].
        mass: Aircraft mass [kg].
        true_airspeed: True airspeed [m/s].
        f: Energy share factor (dimensionless), typically 0.3 to 1.7.
        air_temperature: Actual air temperature [K].
        altitude: Geometric altitude [m] (for ISA reference).

    Returns:
        Rate of climb/descent in pressure altitude [m/s].
        Positive = climb, negative = descent.

    Reference:
        BADA User Manual Section 3.1, pyBADA implementation
    """
    # Uncorrected (geometric) ROCD
    rocd_geo = (thrust - drag) * true_airspeed * f / (mass * isa.G)

    # Apply temperature correction to get pressure altitude rate
    correction = temp_correction_factor(air_temperature, altitude)
    return rocd_geo * correction


def thrust_from_rocd(
    rocd: ArrayScalarLike,
    drag: ArrayScalarLike,
    mass: ArrayScalarLike,
    true_airspeed: ArrayScalarLike,
    f: ArrayScalarLike,
    air_temperature: ArrayScalarLike,
    altitude: ArrayScalarLike,
) -> ArrayScalarLike:
    """Compute required thrust from PRESSURE altitude rate (dHp/dt).

    Inverse of rocd_pressure(): given the desired vertical speed and f,
    compute the required thrust.

    Args:
        rocd: Rate of climb/descent in pressure altitude [m/s].
        drag: Aerodynamic drag [N].
        mass: Aircraft mass [kg].
        true_airspeed: True airspeed [m/s].
        f: Energy share factor (dimensionless).
        air_temperature: Actual air temperature [K].
        altitude: Geometric altitude [m] (for ISA reference).

    Returns:
        Required thrust [N].

    Reference:
        Derived from BADA TEM equation (3.1-4)
    """
    # Convert pressure ROCD to geometric ROCD
    correction = temp_correction_factor(air_temperature, altitude)
    rocd_geo = rocd / correction

    # Solve TEM for thrust: T = D + (m · g · rocd_geo) / (V · f)
    # Clamp denominator to avoid division by zero (level flight with f→0)
    denom = true_airspeed * f
    safe_denom = np.maximum(np.abs(denom), EPSILON_CLAMP) * np.sign(denom + EPSILON_CLAMP)
    return drag + (mass * isa.G * rocd_geo) / safe_denom


def f_from_state(
    thrust: ArrayScalarLike,
    drag: ArrayScalarLike,
    mass: ArrayScalarLike,
    true_airspeed: ArrayScalarLike,
    rocd: ArrayScalarLike,
    air_temperature: ArrayScalarLike,
    altitude: ArrayScalarLike,
) -> ArrayScalarLike:
    """Compute energy share factor from thrust and PRESSURE altitude rate.

    Given thrust and vertical speed, compute the implied f. Useful for
    analyzing flight data or when f is the free variable.

    Args:
        thrust: Engine thrust [N].
        drag: Aerodynamic drag [N].
        mass: Aircraft mass [kg].
        true_airspeed: True airspeed [m/s].
        rocd: Rate of climb/descent in pressure altitude [m/s].
        air_temperature: Actual air temperature [K].
        altitude: Geometric altitude [m] (for ISA reference).

    Returns:
        Energy share factor f (dimensionless).

    Reference:
        Derived from BADA TEM equation (3.1-4)
    """
    # Convert pressure ROCD to geometric ROCD
    correction = temp_correction_factor(air_temperature, altitude)
    rocd_geo = rocd / correction

    # Solve TEM for f: f = (m · g · rocd_geo) / [(T - D) · V]
    # When T=D (level flight), f is undefined; return default (1.0)
    excess_thrust = thrust - drag
    denom = excess_thrust * true_airspeed
    safe_denom = np.maximum(np.abs(denom), EPSILON_CLAMP) * np.sign(denom + EPSILON_CLAMP)

    # Compute f, but return default when excess_thrust ≈ 0
    f_computed = (mass * isa.G * rocd_geo) / safe_denom
    return np.where(
        np.abs(excess_thrust) < EPSILON_CLAMP,
        DEFAULT_ENERGY_SHARE_FACTOR,
        f_computed,
    )


def residual(
    thrust: ArrayScalarLike,
    drag: ArrayScalarLike,
    mass: ArrayScalarLike,
    true_airspeed: ArrayScalarLike,
    f: ArrayScalarLike,
    rocd: ArrayScalarLike,
    air_temperature: ArrayScalarLike,
    altitude: ArrayScalarLike,
) -> ArrayScalarLike:
    """Compute TEM residual for use as optimization constraint.

    Returns zero when the TEM equation is satisfied. Use as:
        opti.subject_to(tem.residual(...) == 0)

    Args:
        thrust: Engine thrust [N].
        drag: Aerodynamic drag [N].
        mass: Aircraft mass [kg].
        true_airspeed: True airspeed [m/s].
        f: Energy share factor (dimensionless).
        rocd: Rate of climb/descent in pressure altitude [m/s].
        air_temperature: Actual air temperature [K].
        altitude: Geometric altitude [m] (for ISA reference).

    Returns:
        TEM residual [m/s]. Zero when equation is satisfied.
    """
    rocd_computed = rocd_pressure(thrust, drag, mass, true_airspeed, f, air_temperature, altitude)
    return rocd_computed - rocd


def rocd_geometric(
    rocd_p: ArrayScalarLike,
    air_temperature: ArrayScalarLike,
    altitude: ArrayScalarLike,
) -> ArrayScalarLike:
    """Convert pressure altitude rate to geometric altitude rate.

    Use this when you need the actual geometric vertical speed (e.g., for
    trajectory integration in geometric coordinates).

    Args:
        rocd_p: Rate of climb/descent in pressure altitude [m/s].
        air_temperature: Actual air temperature [K].
        altitude: Altitude [m] (for ISA reference temperature lookup).

    Returns:
        Rate of climb/descent in geometric altitude [m/s].
    """
    # dh_geo/dt = dHp/dt · (T_actual / T_ISA)
    T_isa = isa.air_temperature(altitude)
    return rocd_p * (air_temperature / T_isa)


class TotalEnergyModel:
    """Total Energy Model state resolver.

    Resolves the three TEM variables (thrust, vertical_speed, f) given any two.
    Uses per-row NaN-based branching to handle mixed data.

    Decision tree (per row):
        - thrust + f provided → compute vertical_speed
        - vertical_speed + f provided → compute thrust
        - thrust + vertical_speed provided → compute f
        - only thrust provided → use f from speed_schedule or 1.0, compute vertical_speed
        - only vertical_speed provided → use f from speed_schedule or 1.0, compute thrust
        - neither provided → defaults (thrust=drag, vertical_speed=0, f=1.0)

    f resolution priority:
        1. User-provided f → use directly
        2. speed_schedule provided → compute from schedule (needs mach)
        3. Default: 1.0 (constant speed)

    Example:
        >>> from openleaf.performance.physics.esf import SpeedSchedule
        >>> tem_state = TotalEnergyModel(
        ...     drag=50000.0,
        ...     mass=70000.0,
        ...     true_airspeed=230.0,
        ...     altitude=10000.0,
        ...     mach=0.78,
        ...     thrust=80000.0,
        ...     speed_schedule=SpeedSchedule.CONSTANT_MACH,
        ... )
        >>> vs = tem_state.vertical_speed  # Computed with f from speed schedule
    """

    def __init__(
        self,
        drag: ArrayScalarLike,
        mass: ArrayScalarLike,
        true_airspeed: ArrayScalarLike,
        altitude: ArrayScalarLike,
        air_temperature: ArrayScalarLike | None = None,
        thrust: ArrayScalarLike | None = None,
        vertical_speed: ArrayScalarLike | None = None,
        f: ArrayScalarLike | None = None,
        mach: ArrayScalarLike | None = None,
        phase: ArrayScalarLike | None = None,
        speed_schedule: SpeedSchedule | None = None,
    ):
        """Initialize TEM state resolver.

        Args:
            drag: Aerodynamic drag [N] (required).
            mass: Aircraft mass [kg] (required).
            true_airspeed: True airspeed [m/s] (required).
            altitude: Geometric altitude [m] (required).
            air_temperature: Actual air temperature [K]. If None, uses ISA.
            thrust: Engine thrust [N] (optional).
            vertical_speed: Pressure altitude rate [m/s] (optional).
            f: Energy share factor [-] (optional).
            mach: Mach number [-] (optional, needed for speed_schedule).
            phase: FlightPhase enum (CLIMB, CRUISE, DESCENT) (optional).
            speed_schedule: SpeedSchedule enum for f computation (optional).
        """
        self._drag = drag
        self._mass = mass
        self._true_airspeed = true_airspeed
        self._altitude = altitude
        self._air_temperature = (
            air_temperature if air_temperature is not None else isa.air_temperature(altitude)
        )

        # Store user-provided values (may contain NaN for per-row branching)
        self._thrust_input = thrust
        self._vertical_speed_input = vertical_speed
        self._f_input = f
        self._mach = mach
        self._phase = phase
        self._speed_schedule = speed_schedule

        # Cached computed values
        self._thrust_computed = None
        self._vertical_speed_computed = None
        self._f_computed = None

    def _has_valid(self, arr: ArrayScalarLike | None) -> ArrayScalarLike:
        """Check if array has valid (non-NaN) values per element.

        Returns boolean array matching shape of self._altitude.
        """
        if arr is None:
            return np.zeros_like(self._altitude, dtype=bool)
        return ~np.isnan(arr)

    @property
    def thrust(self) -> ArrayScalarLike:
        """Thrust force [N].

        Resolution (per row):
            1. User-provided thrust → use directly
            2. vertical_speed provided → compute via TEM
            3. Default: thrust = drag (level flight)
        """
        if self._thrust_computed is not None:
            return self._thrust_computed

        # Start with default: T = D
        result = np.asarray(self._drag)

        # Where vertical_speed is provided, compute thrust
        if self._vertical_speed_input is not None:
            f_val = self._resolve_f_input()
            thrust_from_vs = thrust_from_rocd(
                rocd=self._vertical_speed_input,
                drag=self._drag,
                mass=self._mass,
                true_airspeed=self._true_airspeed,
                f=f_val,
                air_temperature=self._air_temperature,
                altitude=self._altitude,
            )
            has_vs = self._has_valid(self._vertical_speed_input)
            result = np.where(has_vs, thrust_from_vs, result)

        # Where thrust is provided, use it (highest priority)
        if self._thrust_input is not None:
            has_thrust = self._has_valid(self._thrust_input)
            result = np.where(has_thrust, self._thrust_input, result)

        self._thrust_computed = result
        return result

    def _resolve_f_input(self) -> ArrayScalarLike:
        """Resolve f value without circular dependency on thrust/vertical_speed.

        Priority:
            1. User-provided f → use directly
            2. From speed schedule → compute
            3. Default: 1.0 (constant speed)
        """
        if self._f_input is not None:
            has_f = self._has_valid(self._f_input)
            f_from_schedule = self._f_from_speed_schedule()
            return np.where(has_f, self._f_input, f_from_schedule)
        return self._f_from_speed_schedule()

    @property
    def vertical_speed(self) -> ArrayScalarLike:
        """Vertical speed as pressure altitude rate [m/s].

        Resolution (per row):
            1. User-provided vertical_speed → use directly
            2. thrust provided → compute via TEM
            3. Default: 0.0 (level flight)
        """
        if self._vertical_speed_computed is not None:
            return self._vertical_speed_computed

        # Start with default: level flight
        result = np.zeros_like(self._altitude)

        # Where thrust is provided, compute vertical_speed
        if self._thrust_input is not None:
            f_val = self._resolve_f_input()
            vs_from_thrust = rocd_pressure(
                thrust=self._thrust_input,
                drag=self._drag,
                mass=self._mass,
                true_airspeed=self._true_airspeed,
                f=f_val,
                air_temperature=self._air_temperature,
                altitude=self._altitude,
            )
            has_thrust = self._has_valid(self._thrust_input)
            result = np.where(has_thrust, vs_from_thrust, result)

        # Where vertical_speed is provided, use it (highest priority)
        if self._vertical_speed_input is not None:
            has_vs = self._has_valid(self._vertical_speed_input)
            result = np.where(has_vs, self._vertical_speed_input, result)

        self._vertical_speed_computed = result
        return result

    def _f_from_speed_schedule(self) -> ArrayScalarLike:
        """Compute f from speed_schedule if available, else return default."""
        from openleaf.performance.physics import esf

        if self._speed_schedule is None:
            return DEFAULT_ENERGY_SHARE_FACTOR

        schedule = self._speed_schedule

        # Constant schedules — pure dict lookup
        schedule_constants = {
            esf.SpeedSchedule.ACCEL_CLIMB: esf.F_ACCEL_CLIMB,
            esf.SpeedSchedule.DECEL_CLIMB: esf.F_DECEL_CLIMB,
            esf.SpeedSchedule.ACCEL_DESCENT: esf.F_ACCEL_DESCENT,
            esf.SpeedSchedule.DECEL_DESCENT: esf.F_DECEL_DESCENT,
        }
        if schedule in schedule_constants:
            return schedule_constants[schedule]

        # Computed schedules — require mach
        computed_handlers = {
            esf.SpeedSchedule.CONSTANT_MACH: esf.f_constant_mach,
            esf.SpeedSchedule.CONSTANT_CAS: esf.f_constant_cas,
        }
        handler = computed_handlers.get(schedule)
        if handler is not None:
            if self._mach is None:
                return DEFAULT_ENERGY_SHARE_FACTOR
            return handler(self._mach, self._altitude)

        return DEFAULT_ENERGY_SHARE_FACTOR

    @property
    def f(self) -> ArrayScalarLike:
        """Energy share factor [-].

        Resolution (per row):
            1. User-provided f → use directly
            2. Both thrust AND vertical_speed provided → compute via TEM
            3. speed_schedule provided → compute from schedule
            4. Default: 1.0 (constant speed)
        """
        if self._f_computed is not None:
            return self._f_computed

        # Start with default from speed schedule (or 1.0)
        f_default = self._f_from_speed_schedule()
        if np.isscalar(f_default):
            result = np.full_like(self._altitude, f_default)
        else:
            result = np.asarray(f_default)

        # Where both thrust and vertical_speed are provided, compute f from TEM
        if self._thrust_input is not None and self._vertical_speed_input is not None:
            f_computed = f_from_state(
                thrust=self._thrust_input,
                drag=self._drag,
                mass=self._mass,
                true_airspeed=self._true_airspeed,
                rocd=self._vertical_speed_input,
                air_temperature=self._air_temperature,
                altitude=self._altitude,
            )
            has_both = self._has_valid(self._thrust_input) & self._has_valid(
                self._vertical_speed_input
            )
            result = np.where(has_both, f_computed, result)

        # Where f is provided, use it (highest priority)
        if self._f_input is not None:
            has_f = self._has_valid(self._f_input)
            result = np.where(has_f, self._f_input, result)

        self._f_computed = result
        return result

    @property
    def vertical_speed_geometric(self) -> ArrayScalarLike:
        """Vertical speed as geometric altitude rate [m/s]."""
        return rocd_geometric(
            rocd_p=self.vertical_speed,
            air_temperature=self._air_temperature,
            altitude=self._altitude,
        )
