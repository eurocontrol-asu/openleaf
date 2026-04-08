"""
BADA3 aircraft performance models.

Provides engine, aerodynamic, and aircraft models for BADA3 aircraft
performance calculations including thrust, fuel flow, and drag.

Usage:
    # Initialize from ICAO code (requires configuration)
    model = BADA3Model("A320")

    # Initialize from specific file
    model = BADA3Model.from_file("/path/to/A320__.OPF")

    # Initialize from pre-loaded data
    model = BADA3Model.from_data(bada3_data)

    # List available aircraft
    aircraft = BADA3Model.available_aircraft()
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.config import get_bada3_path
from openleaf.performance.config import get_bada3_synonyms_path
from openleaf.performance.core.perfos import AerodynamicsModel
from openleaf.performance.core.perfos import AircraftModel
from openleaf.performance.core.perfos import EngineModel
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import EngineSetting
from openleaf.performance.core.types import EngineType
from openleaf.performance.core.types import FlightPhase
from openleaf.performance.core.validation import requires
from openleaf.performance.logging import get_logger
from openleaf.performance.models.bada3.limits import BADA3LimitsModel
from openleaf.performance.models.bada3.parsing import parse_bada3_file
from openleaf.performance.models.bada3.schemas import BADA3AerodynamicsData
from openleaf.performance.models.bada3.schemas import BADA3Data
from openleaf.performance.models.bada3.schemas import BADA3EngineData
from openleaf.performance.models.bada3.synonyms import available_bada3_aircraft
from openleaf.performance.models.bada3.synonyms import resolve_bada3_file
from openleaf.performance.models.common import AERO_CONFIG_THRESHOLDS
from openleaf.performance.physics.units import FT
from openleaf.performance.physics.units import KT


logger = get_logger(__name__)

data_path = get_bada3_path()
synonym_path = get_bada3_synonyms_path()

if TYPE_CHECKING:
    from pathlib import Path

    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = [
    "BADA3AeroModel",
    "BADA3EngineModel",
    "BADA3JetEngineModel",
    "BADA3LimitsModel",
    "BADA3Model",
    "BADA3PistonEngineModel",
    "BADA3PropellerModel",
    "BADA3TurbineModel",
    "BADA3TurbopropEngineModel",
]

# BADA3 unit conversion constants
# Fuel flow coefficients are given in kg/min per unit thrust
# These constants convert to kg/s per Newton

_MINUTES_TO_SECONDS = 60.0
"""Conversion factor: minutes to seconds."""

_JET_FF_COEF_CONVERSION = 60_000.0
"""Jet engine fuel flow coefficient conversion: kg/(min·kN) to kg/(s·N)."""

_TURBOPROP_FF_COEF_CONVERSION = 60_000_000.0
"""Turboprop fuel flow coefficient conversion: kg/(min·kN·kt) to kg/(s·N·(m/s))."""


# --- Engine Type Models ---


class BADA3EngineModel(EngineModel[BADA3EngineData]):
    """BADA3 engine model with thrust and fuel flow calculations.

    Implements BADA3 engine performance models with thrust and fuel flow
    calculations based on speed, altitude, and throttle settings.
    """

    # Common methods
    def _thrust_max(
        self,
        *,
        v: ArrayScalarLike,
        h: ArrayScalarLike,
        delta_isa: ArrayScalarLike = 0.0,
        setting: ArrayScalarLike = EngineSetting.MCMB,
    ) -> ArrayScalarLike:
        """Calculate maximum available thrust from flight conditions.

        Implements BADA3 formula 3.7-1 with ISA correction:
            T_max = T_max_isa * (1 - CT_c5 * (ΔTISA - CT_c4))

        With constraint from manual:
            - if (ΔTISA - CT_c4) < 0, then (ΔTISA - CT_c4) = 0

        Args:
            v: True airspeed in m/s.
            h: Geometric altitude in m.
            delta_isa: Temperature deviation from ISA (K).
            setting: Engine throttle setting (default: max climb).

        Returns:
            Maximum available thrust in Newtons.
        """
        thrust_max_isa = self._thrust_max_isa(v=v, h=h)

        coef = np.where(
            np.equal(setting, EngineSetting.MCRZ),
            self.data.ct_crz,
            1.0,
        )

        # Non-ISA correction per formula 3.7-1
        # Per manual: if (ΔTISA - CT_c4) < 0, then set to 0
        delta_isa_eff = np.maximum(delta_isa - self.data.ct_4, 0.0)
        isa_correction = 1.0 - self.data.ct_5 * delta_isa_eff

        # thrust_max_isa on the left ensures torch/JAX type propagation
        return thrust_max_isa * (isa_correction * coef)

    def _thrust_min(
        self,
        *,
        v: ArrayScalarLike,
        h: ArrayScalarLike,
        delta_isa: ArrayScalarLike = 0.0,
        setting: ArrayScalarLike = EngineSetting.MCMB,
        conf: AerodynamicConfig = AerodynamicConfig.CR,
    ) -> ArrayScalarLike:
        """Calculate minimum available thrust from flight conditions.

        Args:
            v: True airspeed in m/s.
            h: Geometric altitude in m.
            delta_isa: Temperature deviation from ISA (K).
            setting: Engine throttle setting.
            conf: Aerodynamic configuration for idle thrust selection.

        Returns:
            Minimum available thrust in Newtons.
        """
        h_ft = h / FT

        coef = np.where(
            np.greater(h_ft, self.data.h_des),
            self.data.ct_des_high,
            np.where(
                np.equal(conf, AerodynamicConfig.AP),
                self.data.ct_des_app,
                np.where(
                    np.equal(conf, AerodynamicConfig.LD),
                    self.data.ct_des_ld,
                    self.data.ct_des_low,
                ),
            ),
        )

        thrust_max = self._thrust_max(
            v=v,
            h=h,
            setting=setting,
            delta_isa=delta_isa,
        )

        # thrust_max on left ensures torch/JAX type propagation
        return thrust_max * coef

    def _fuel_flow(
        self,
        *,
        v: ArrayScalarLike,
        thrust: ArrayScalarLike,
        setting: ArrayScalarLike = EngineSetting.MCMB,
    ) -> ArrayScalarLike:
        """Calculate fuel flow from flight conditions and thrust.

        Args:
            v: True airspeed in m/s.
            thrust: Actual thrust in Newtons.
            setting: Engine throttle setting.

        Returns:
            Fuel flow rate in kg/s.
        """
        fuel_flow_nom = self._fuel_flow_nom(
            v=v,
            thrust=thrust,
        )
        coef = np.where(
            np.equal(setting, EngineSetting.MCRZ),
            self.data.cf_crz,
            1.0,
        )
        # fuel_flow_nom on the left ensures torch/JAX type propagation
        return fuel_flow_nom * coef

    @requires("true_airspeed", "altitude", "engine_setting", "delta_isa")
    def thrust_max(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum thrust from flight state dictionary.

        Args:
            data: Flight state data with:
                - true_airspeed: True airspeed [m/s] (required)
                - altitude: Altitude [m] (required)
                - engine_setting: Engine setting enum (required)
                - delta_isa: Temperature deviation from ISA [K] (required)

        Returns:
            Maximum available thrust in Newtons.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        logger.debug("Computing thrust_max with setting=%s", data["engine_setting"])

        return self._thrust_max(
            v=data["true_airspeed"],
            h=data["altitude"],
            setting=data["engine_setting"],
            delta_isa=data["delta_isa"],
        )

    @requires(
        "true_airspeed",
        "altitude",
        "engine_setting",
        "delta_isa",
        "aerodynamic_configuration",
    )
    def thrust_min(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum thrust from flight state dictionary.

        Args:
            data: Flight state data with:
                - true_airspeed: True airspeed [m/s] (required)
                - altitude: Altitude [m] (required)
                - engine_setting: Engine setting enum (required)
                - delta_isa: Temperature deviation from ISA [K] (required)
                - aerodynamic_configuration: Config enum (required)

        Returns:
            Minimum available thrust in Newtons.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        logger.debug(
            "Computing thrust_min with setting=%s, config=%s",
            data["engine_setting"],
            data["aerodynamic_configuration"],
        )

        return self._thrust_min(
            v=data["true_airspeed"],
            h=data["altitude"],
            setting=data["engine_setting"],
            delta_isa=data["delta_isa"],
            conf=data["aerodynamic_configuration"],
        )

    @requires("true_airspeed", "thrust", "engine_setting")
    def fuel_flow(self, data: KeyedData) -> ArrayScalarLike:
        """Get fuel flow from flight state dictionary.

        Args:
            data: Flight state data with:
                - true_airspeed: True airspeed [m/s] (required)
                - thrust: Actual thrust [N] (required)
                - engine_setting: Engine setting enum (required)

        Returns:
            Fuel flow rate in kg/s.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        logger.debug("Computing fuel_flow")

        return self._fuel_flow(
            v=data["true_airspeed"],
            thrust=data["thrust"],
            setting=data["engine_setting"],
        )

    @requires("altitude")
    def fuel_flow_min(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum fuel flow from flight state dictionary.

        Args:
            data: Flight state data with:
                - altitude: Altitude [m] (required)

        Returns:
            Minimum fuel flow rate in kg/s.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        logger.debug("Computing fuel_flow_min")

        return self._fuel_flow_min(h=data["altitude"])

    # Abstract methods
    @abstractmethod
    def _thrust_max_isa(
        self,
        *,
        v: ArrayScalarLike,
        h: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate maximum thrust under ISA conditions (abstract method).

        Args:
            v: True airspeed in m/s.
            h: Geometric altitude in m.

        Returns:
            Maximum thrust under ISA conditions in Newtons.
        """

    @abstractmethod
    def _fuel_flow_nom(
        self,
        *,
        v: ArrayScalarLike,
        thrust: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate nominal fuel flow (abstract method).

        Args:
            v: True airspeed in m/s.
            thrust: Actual thrust in Newtons.

        Returns:
            Nominal fuel flow rate in kg/s.
        """

    @abstractmethod
    def _fuel_flow_min(
        self,
        *,
        h: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate minimum fuel flow (abstract method).

        Args:
            h: Geometric altitude in m.

        Returns:
            Minimum fuel flow rate in kg/s.
        """


class BADA3TurbineModel(BADA3EngineModel):
    """Base class for BADA3 turbine engine models.

    Provides common implementations for jet and turboprop engines based on
    engine efficiency and nominal fuel flow calculations.
    """

    @abstractmethod
    def _engine_efficiency(
        self,
        *,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate engine efficiency (abstract method).

        Args:
            v: True airspeed in m/s.

        Returns:
            Engine efficiency value.
        """

    def _fuel_flow_nom(
        self,
        *,
        v: ArrayScalarLike,
        thrust: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate nominal fuel flow based on engine efficiency.

        Args:
            v: True airspeed in m/s.
            thrust: Actual thrust in Newtons.

        Returns:
            Nominal fuel flow rate in kg/s.
        """
        eta = self._engine_efficiency(v=v)
        return eta * thrust

    def _fuel_flow_min(
        self,
        *,
        h: ArrayScalarLike,
    ) -> ArrayScalarLike:
        h_ft = h / FT
        return self.data.cf_3 * (1.0 - h_ft / self.data.cf_4) / _MINUTES_TO_SECONDS  # in kg/s


class BADA3PropellerModel(BADA3EngineModel):
    """Base class for BADA3 propeller engine models.

    Base implementation for turboprop and piston engine models that use propellers.
    """


class BADA3JetEngineModel(BADA3TurbineModel):
    """BADA3 jet engine model.

    Implements thrust and fuel flow calculations for jet engines using
    BADA3 turbine engine coefficients.
    """

    def _thrust_max_isa(
        self,
        *,
        v: ArrayScalarLike,
        h: ArrayScalarLike,
    ) -> ArrayScalarLike:
        h_ft = h / FT
        return self.data.ct_1 * (1.0 - h_ft / self.data.ct_2 + self.data.ct_3 * h_ft * h_ft)

    def _engine_efficiency(
        self,
        *,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        v_kt = v / KT
        return (
            self.data.cf_1 * (1.0 + v_kt / self.data.cf_2) / _JET_FF_COEF_CONVERSION
        )  # in kg/(s·N)


class BADA3TurbopropEngineModel(BADA3TurbineModel, BADA3PropellerModel):
    """BADA3 turboprop engine model.

    Implements thrust and fuel flow calculations for turboprop engines using
    BADA3 turbine and propeller engine coefficients.
    """

    def _thrust_max_isa(
        self,
        *,
        v: ArrayScalarLike,
        h: ArrayScalarLike,
    ) -> ArrayScalarLike:
        v_kt = v / KT
        h_ft = h / FT
        return self.data.ct_1 * (1.0 - h_ft / self.data.ct_2) / v_kt + self.data.ct_3

    def _engine_efficiency(
        self,
        *,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        v_kt = v / KT
        return (
            self.data.cf_1 * (1.0 - v_kt / self.data.cf_2) * (v_kt / _TURBOPROP_FF_COEF_CONVERSION)
        )  # in kg/(s·N)


class BADA3PistonEngineModel(BADA3PropellerModel):
    """BADA3 piston engine model.

    Implements thrust and fuel flow calculations for piston engines using
    BADA3 piston engine coefficients.
    """

    def _thrust_max_isa(
        self,
        *,
        v: ArrayScalarLike,
        h: ArrayScalarLike,
    ) -> ArrayScalarLike:
        v_kt = v / KT
        h_ft = h / FT
        return self.data.ct_1 * (1.0 - h_ft / self.data.ct_2) + self.data.ct_3 / v_kt

    def _fuel_flow_nom(
        self,
        *,
        v: ArrayScalarLike,
        thrust: ArrayScalarLike,
    ) -> ArrayScalarLike:
        # Multiply by (thrust * 0 + 1) to preserve backend type (torch/CasADi/JAX)
        return (thrust * 0 + 1) * self.data.cf_1 / _MINUTES_TO_SECONDS

    def _fuel_flow_min(
        self,
        *,
        h: ArrayScalarLike,
    ) -> ArrayScalarLike:
        # Multiply by (h * 0 + 1) to preserve backend type (torch/CasADi/JAX)
        return (h * 0 + 1) * self.data.cf_3 / _MINUTES_TO_SECONDS


# --- BADA3 Aerodynamics Model ---


class BADA3AeroModel(AerodynamicsModel[BADA3AerodynamicsData]):
    """BADA3 aerodynamic model.

    Implements drag coefficient calculations for different aircraft
    configurations using BADA3 drag polar coefficients.
    """

    @requires("lift_coef", "aerodynamic_configuration")
    def drag_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate drag coefficient from flight state data.

        Extracts lift coefficient and configuration from the data dict
        and computes drag using BADA3 drag polars.

        Args:
            data: Dictionary containing:
                - lift_coef: Lift coefficient (required)
                - aerodynamic_configuration: Config enum (required)

        Returns:
            Drag coefficient (dimensionless).

        Raises:
            MissingFieldError: If any required field is missing.
        """
        logger.debug("Computing drag_coef with config=%s", data["aerodynamic_configuration"])

        return self._drag_coef(
            cl=data["lift_coef"],
            conf=data["aerodynamic_configuration"],
        )

    def drag_coef_from_inputs(
        self,
        *,
        cl: ArrayScalarLike,
        mach: ArrayScalarLike,
        conf: ArrayScalarLike = AerodynamicConfig.CR,
    ) -> ArrayScalarLike:
        """Calculate drag coefficient from explicit inputs.

        Note:
            BADA3 drag polar is independent of Mach number (no wave drag).
            The ``mach`` parameter is included for API consistency with BADA4
            but is not used in the calculation.

        Args:
            cl: Lift coefficient (dimensionless).
            mach: Mach number (dimensionless) - unused, for API consistency.
            conf: Aerodynamic configuration (default: cruise).

        Returns:
            Drag coefficient (dimensionless).
        """
        return self._drag_coef(cl=cl, conf=conf)

    def _drag_coef(
        self,
        *,
        cl: ArrayScalarLike,
        conf: ArrayScalarLike = AerodynamicConfig.CR,
    ) -> ArrayScalarLike:
        """Method to calculate drag coefficient from standard inputs.

        Args:
            cl: Lift coefficient (dimensionless).
            conf: Aerodynamic configuration (default: cruise).

        Returns:
            Drag coefficient (dimensionless).
        """
        # Fast path for scalar configuration (avoids np.where namespace
        # conflicts when cl is a JAX/PyTorch array and np.equal returns
        # a numpy.bool_ that triggers mixed-namespace detection).
        if isinstance(conf, AerodynamicConfig):
            return self._drag_polar(cl=cl, conf=conf)

        # Vectorized case: conf is an array of configurations
        cd = self._drag_polar(cl=cl, conf=AerodynamicConfig.CR)

        for config in (
            AerodynamicConfig.TO,
            AerodynamicConfig.IC,
            AerodynamicConfig.AP,
            AerodynamicConfig.LD,
        ):
            cd = np.where(
                np.equal(conf, config),
                self._drag_polar(cl=cl, conf=config),
                cd,
            )

        return cd

    def _drag_polar(
        self,
        *,
        cl: ArrayScalarLike,
        conf: AerodynamicConfig,
    ) -> ArrayScalarLike:
        """Calculate drag polar for a single configuration.

        Args:
            cl: Lift coefficient
            conf: Single AerodynamicConfig enum value

        Returns:
            Drag coefficient
        """
        cd_0 = self.data.cd_0[conf]
        cd_2 = self.data.cd_2[conf]
        return cd_0 + cd_2 * cl**2


# --- Full BADA3 Aircraft Model ---


class BADA3Model(AircraftModel):
    """Complete BADA3 aircraft performance model.

    Integrates aircraft data, aerodynamics, and engine models to provide
    full aircraft performance calculations according to BADA3 specification.

    Attributes:
        limits: BADA3-specific aircraft limits including mass/temperature
            dependent max altitude, stall speeds, and buffet data.
    """

    def __init__(self, icao: str):
        """Initialize BADA3 model from ICAO aircraft type code.

        Args:
            icao: ICAO aircraft type code (e.g., 'A320', 'B738').

        Raises:
            ConfigurationError: If BADA3 data path is not configured.
            FileNotFoundError: If aircraft not found in BADA3 data.

        Example:
            >>> model = BADA3Model("A320")
            >>> model.mtom
            77000.0
        """
        logger.info("Loading BADA3 model for %s", icao)
        opf_file = resolve_bada3_file(icao, data_path, synonym_path)
        logger.debug("Resolved OPF file: %s", opf_file)
        data = parse_bada3_file(str(opf_file))
        self._init_from_data(data)

    def _init_from_data(self, data: BADA3Data) -> None:
        """Initialize model from BADA3 data object.

        Args:
            data: Parsed BADA3 data.
        """
        engine: EngineModel

        if data.aircraft.engine_type == EngineType.JET:
            logger.debug("Initializing jet engine model")
            engine = BADA3JetEngineModel(data.engine)
        elif data.aircraft.engine_type == EngineType.TURBOPROP:
            logger.debug("Initializing turboprop engine model")
            engine = BADA3TurbopropEngineModel(data.engine)
        elif data.aircraft.engine_type == EngineType.PISTON:
            logger.debug("Initializing piston engine model")
            engine = BADA3PistonEngineModel(data.engine)
        else:
            logger.error("Unsupported engine type: %s", data.aircraft.engine_type)
            raise ValueError(f"Unsupported engine type: {data.aircraft.engine_type}")

        aerodynamics = BADA3AeroModel(data.aerodynamics)
        limits = BADA3LimitsModel(data.limits)
        super().__init__(data.aircraft, engine, aerodynamics, limits)
        logger.info(
            "BADA3 model initialized: %s (engine=%s, mtom=%.0f kg)",
            data.aircraft.model,
            data.aircraft.engine_type.name,
            data.limits.mtom,
        )

    @classmethod
    def from_data(cls, data: BADA3Data) -> BADA3Model:
        """Create BADA3 model from pre-loaded data.

        Args:
            data: Parsed BADA3 data object.

        Returns:
            Initialized BADA3 aircraft model.
        """
        instance = object.__new__(cls)
        instance._init_from_data(data)
        return instance

    @classmethod
    def from_file(
        cls,
        file: str | Path,
        **kwargs,
    ) -> BADA3Model:
        """Load BADA3 model from OPF file.

        Args:
            file: Path to BADA3 OPF file.
            **kwargs: Additional arguments (unused, for interface consistency).

        Returns:
            Initialized BADA3 aircraft model.
        """
        data = parse_bada3_file(str(file))
        return cls.from_data(data)

    @classmethod
    def available_aircraft(cls) -> list[str]:
        """List available BADA3 aircraft types.

        Returns:
            Sorted list of ICAO codes for available aircraft.

        Raises:
            ConfigurationError: If BADA3 data path is not configured.
        """
        return available_bada3_aircraft(data_path, synonym_path)

    def aerodynamic_configuration(self, data: KeyedData) -> ArrayScalarLike:
        """Determine BADA3 aerodynamic configuration from flight state.

        Implements BADA3 configuration selection rules from Section 3.5 and 5.6.
        Fully vectorized using np.where for array inputs.

        For CLIMB (flight_phase = "climb" or not specified):
          - TO (Take-off): altitude <= H_max_TO (400 ft)
          - IC (Initial Climb): H_max_TO < altitude <= H_max_IC (2000 ft)
          - CR (Cruise): altitude > H_max_IC

        For DESCENT (flight_phase = "descent"):
          - CR: altitude > H_max_AP (8000 ft) or V > Vmin_CR + 10 kt
          - AP: altitude <= H_max_AP and V < Vmin_CR + 10 kt
          - LD: altitude <= H_max_LD (3000 ft) and V < Vmin_AP + 10 kt

        Stall speeds are mass-corrected per Formula 3.4-1:
          V_stall = V_stall_ref * sqrt(m / m_ref)

        Minimum speeds per Section 5.7:
          V_min = C_Vmin * V_stall where C_Vmin = 1.3

        Args:
            data: Flight state dictionary with:
                - altitude: Geometric altitude in meters (required)
                - flight_phase: "climb", "cruise", or "descent" (required)
                - true_airspeed: True airspeed [m/s] (required)
                - aircraft_mass: Aircraft mass [kg] (required)
                - aerodynamic_configuration: Explicit override (optional)

        Returns:
            AerodynamicConfig enum value (BADA-specific, vectorized).

        Raises:
            MissingFieldError: If any required field is missing.
        """
        # Allow explicit override
        if "aerodynamic_configuration" in data:
            logger.debug("Using explicit aerodynamic_configuration from data")
            return data["aerodynamic_configuration"]

        # Use common thresholds (Section 5.6 altitude, Section 5.7 c_vmin)
        thresholds = AERO_CONFIG_THRESHOLDS

        # Extract inputs (all required when not overridden)
        altitude = data["altitude"]
        h_ft = altitude / FT
        flight_phase = data["flight_phase"]
        v = data["true_airspeed"]
        v_kt = v / KT
        mass = data["aircraft_mass"]

        # Get reference stall speeds (in knots CAS)
        vstall_cr_ref = self.limits.stall_speed(AerodynamicConfig.CR)
        vstall_ap_ref = self.limits.stall_speed(AerodynamicConfig.AP)

        # Apply mass correction: V_stall = V_stall_ref * sqrt(m / m_ref)
        # Formula 3.4-1 from Section 3.4
        mref = self.limits.data.mref
        mass_factor = np.sqrt(mass / mref) if mref > 0 else 1.0
        vstall_cr = vstall_cr_ref * mass_factor
        vstall_ap = vstall_ap_ref * mass_factor

        # Minimum speeds: Vmin = c_vmin * Vstall (Section 5.7)
        vmin_cr = thresholds.c_vmin * vstall_cr
        vmin_ap = thresholds.c_vmin * vstall_ap

        # Boolean conditions for flight phase
        is_descent = np.equal(flight_phase, FlightPhase.DESCENT)
        is_cruise = np.equal(flight_phase, FlightPhase.CRUISE)

        # Altitude conditions
        in_to_zone = np.less_equal(h_ft, thresholds.h_max_to)
        in_ic_zone = np.logical_and(
            np.greater(h_ft, thresholds.h_max_to),
            np.less_equal(h_ft, thresholds.h_max_ic),
        )
        above_ap_zone = np.greater(h_ft, thresholds.h_max_ap)
        in_ld_zone = np.less_equal(h_ft, thresholds.h_max_ld)

        # Speed conditions for descent
        fast_for_cr = np.greater_equal(v_kt, vmin_cr + thresholds.speed_margin)
        slow_for_ld = np.less(v_kt, vmin_ap + thresholds.speed_margin)

        # Climb configuration selection (based on altitude only)
        climb_config = np.where(
            in_to_zone,
            AerodynamicConfig.TO,
            np.where(in_ic_zone, AerodynamicConfig.IC, AerodynamicConfig.CR),
        )

        # Descent configuration selection (based on altitude and speed)
        # CR: above 8000 ft OR fast enough (V >= Vmin_CR + 10)
        # AP: below 8000 ft AND slow (V < Vmin_CR + 10) AND above 3000 ft
        # LD: below 3000 ft AND slow (V < Vmin_AP + 10)
        descent_config = np.where(
            np.logical_or(above_ap_zone, fast_for_cr),
            AerodynamicConfig.CR,
            np.where(
                np.logical_and(in_ld_zone, slow_for_ld),
                AerodynamicConfig.LD,
                AerodynamicConfig.AP,
            ),
        )

        # Select based on flight phase
        config = np.where(
            is_descent,
            descent_config,
            np.where(is_cruise, AerodynamicConfig.CR, climb_config),
        )

        return config

    # Note: high_lift() is intentionally NOT overridden.
    # BADA3 OPF/APF files do not define HLid values like BADA4 XML files.
    # Use aerodynamic_configuration() for BADA3-specific configuration data.
    # The base class returns None for high_lift().

    def landing_gear(self, data: KeyedData) -> ArrayScalarLike:
        """Determine landing gear state from flight state.

        BADA3 OPF files do not explicitly define landing gear position per
        configuration. By BADA3 convention (Section 3.5), landing gear is
        extended in the LD (landing) configuration and retracted otherwise.

        Args:
            data: Flight state dictionary (same as aerodynamic_configuration).

        Returns:
            Boolean indicating gear extended (True = LD config) or
            retracted (False = other configs).
        """
        config = self.aerodynamic_configuration(data)
        return np.equal(config, AerodynamicConfig.LD)
