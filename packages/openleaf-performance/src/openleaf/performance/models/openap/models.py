# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""OpenAP aircraft performance models.

This module wraps the OpenAP library (https://github.com/junzis/openap) to provide
aircraft performance calculations following the openleaf interface patterns.

OpenAP is an open-source library for aircraft performance and emission calculations
developed by Junzi Sun at TU Delft.

All inputs are expected in SI units (m/s, meters, kg, N).
All outputs are in SI units.

Usage:
    model = OpenAPModel("A320")
    aircraft = OpenAPModel.available_aircraft()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
from openleaf.performance.core.perfos import AerodynamicsModel
from openleaf.performance.core.perfos import AircraftModel
from openleaf.performance.core.perfos import EngineModel
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import EngineType
from openleaf.performance.core.validation import requires
from openleaf.performance.exceptions import ModelDataError
from openleaf.performance.logging import get_logger
from openleaf.performance.models.openap import openap_aero
from openleaf.performance.models.openap.backend import OpenleafBackend
from openleaf.performance.models.openap.limits import OpenAPLimits
from openleaf.performance.models.openap.schemas import OpenAPAerodynamicsData
from openleaf.performance.models.openap.schemas import OpenAPAircraftData
from openleaf.performance.models.openap.schemas import OpenAPData
from openleaf.performance.models.openap.schemas import OpenAPEngineData
from openleaf.performance.models.openap.schemas import OpenAPLimitsData
from openleaf.performance.physics import isa
from openleaf.performance.physics import units


logger = get_logger(__name__)

if TYPE_CHECKING:
    from openap.base import DragBase
    from openap.base import FuelFlowBase
    from openap.base import ThrustBase

    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData


__all__ = [
    "OpenAPAeroModel",
    "OpenAPEngineModel",
    "OpenAPLimits",
    "OpenAPModel",
]


# --- Unit conversion constants ---
_KT = units.MS_TO_KT  # m/s -> knots
_FT = units.M_TO_FT  # m -> feet
_FPM = units.MS_TO_FPM  # m/s -> feet/min

# --- ICAO Wake Turbulence Category thresholds ---
# Reference: ICAO Doc 4444 PANS-ATM
_WTC_HEAVY_THRESHOLD_KG = 136_000.0
"""MTOW threshold for Heavy wake turbulence category [kg]."""

_WTC_MEDIUM_THRESHOLD_KG = 7_000.0
"""MTOW threshold for Medium wake turbulence category [kg]."""


# ==============================================================================
# Helper: Build OpenAP components on the openleaf backend
# ==============================================================================
_BACKEND = OpenleafBackend()
"""OpenAP math backend shared by every wrapped component (it holds no state)."""


def _unwrap_ndarrayconvert(obj: object) -> None:
    """Unwrap @ndarrayconvert decorators from an OpenAP object's methods.

    OpenAP's @ndarrayconvert decorator passes CasADi and JAX values through but
    converts every other input, PyTorch tensors included, to a NumPy array,
    which would detach tensors from the autograd graph. This function replaces
    decorated methods with their original unwrapped versions.

    The decorator stores the original function in `.orig_func` attribute.

    Args:
        obj: OpenAP object (Thrust, FuelFlow, Drag) to unwrap.
    """
    import types

    for name in dir(obj):
        if name.startswith("__"):
            continue
        attr = getattr(obj, name, None)
        if callable(attr) and hasattr(attr, "orig_func"):
            orig_func = attr.orig_func
            setattr(obj, name, types.MethodType(orig_func, obj))


def _adapt(obj: DragBase | ThrustBase | FuelFlowBase) -> None:
    """Make an OpenAP component accept every openleaf backend.

    Unwraps its NumPy input conversion and swaps its atmosphere for
    :mod:`~openleaf.performance.models.openap.openap_aero`, so OpenAP uses
    openleaf's ISA model. The math backend itself is passed to the
    component's constructor, because OpenAP binds it at construction time.

    Args:
        obj: OpenAP object (Thrust, FuelFlow, Drag) built with ``_BACKEND``.
    """
    _unwrap_ndarrayconvert(obj)
    obj.aero = openap_aero


# ==============================================================================
# Aerodynamic Model
# ==============================================================================
class OpenAPAeroModel(AerodynamicsModel[OpenAPAerodynamicsData]):
    """Aerodynamic model using OpenAP.

    Wraps OpenAP's Drag class on the openleaf math backend, so it accepts
    NumPy, CasADi, PyTorch and JAX inputs.
    """

    def __init__(self, model_name: str, data: OpenAPAerodynamicsData) -> None:
        """Initialize OpenAP aerodynamic model.

        Args:
            model_name: ICAO aircraft type code (e.g., "A320").
            data: Aerodynamics data container.
        """
        super().__init__(data)
        self.model_name = model_name

        from openap import Drag

        self._drag = Drag(model_name, backend=_BACKEND)
        _adapt(self._drag)

        self._wing_span = data.wing_span

    @property
    def wing_span(self) -> float:
        """Wing span [m]."""
        return self._wing_span

    @requires(
        "aircraft_mass",
        "altitude",
        "true_airspeed",
        "vertical_speed",
        "aerodynamic_configuration",
    )
    def drag_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate drag coefficient.

        CasADi-compatible: Uses np.where for configuration selection.

        Args:
            data: Flight state with:
                - aircraft_mass: Aircraft mass [kg]
                - altitude: Altitude [m]
                - true_airspeed: True airspeed [m/s]
                - vertical_speed: Rate of climb [m/s]
                - aerodynamic_configuration: AerodynamicConfig enum or string
                - flap_angle: Flap angle [deg] (for non-clean)
                - landing_gear: Gear extended flag (for non-clean)

        Returns:
            Drag coefficient [-].
        """
        mass = data["aircraft_mass"]
        altitude_m = data["altitude"]
        tas_ms = data["true_airspeed"]
        vs_ms = data["vertical_speed"]
        config = data["aerodynamic_configuration"]

        # Convert to OpenAP units
        altitude_ft = altitude_m * _FT
        tas_kt = tas_ms * _KT
        vs_fpm = vs_ms * _FPM

        # For non-clean: use provided values or defaults
        flap_angle = data.get("flap_angle", 0.0)
        gear_extended = data.get("landing_gear", False)

        # Compute both configurations (CasADi-compatible)
        drag_clean = self._drag.clean(mass=mass, tas=tas_kt, alt=altitude_ft, vs=vs_fpm)
        drag_nonclean = self._drag.nonclean(
            mass=mass,
            tas=tas_kt,
            alt=altitude_ft,
            flap_angle=flap_angle,
            vs=vs_fpm,
            landing_gear=gear_extended,
        )

        # Select based on configuration (IntEnum comparison)
        is_clean = np.equal(config, AerodynamicConfig.CR)
        drag_n = np.where(is_clean, drag_clean, drag_nonclean)

        # Convert drag force to coefficient: CD = D / (q * S)
        delta = isa.delta(altitude_m)
        theta = isa.theta(altitude_m)
        rho = (delta / theta) * isa.RHO_SL
        q = 0.5 * rho * tas_ms**2

        return drag_n / (q * self.surface_area)


# ==============================================================================
# Engine Model
# ==============================================================================
class OpenAPEngineModel(EngineModel[OpenAPEngineData]):
    """Engine model using OpenAP.

    Wraps OpenAP's Thrust and FuelFlow classes on the openleaf math backend, so
    they accept NumPy, CasADi, PyTorch and JAX inputs.
    """

    def __init__(self, model_name: str, data: OpenAPEngineData) -> None:
        """Initialize OpenAP engine model.

        Args:
            model_name: ICAO aircraft type code.
            data: Engine data container.
        """
        super().__init__(data)
        self.model_name = model_name

        from openap import FuelFlow
        from openap import Thrust

        self._thrust = Thrust(model_name, backend=_BACKEND)
        # FuelFlow builds its own Thrust and Drag on the same backend.
        self._fuel_flow = FuelFlow(model_name, backend=_BACKEND)
        for component in (
            self._thrust,
            self._fuel_flow,
            self._fuel_flow.thrust,
            self._fuel_flow.drag,
        ):
            _adapt(component)

    @property
    def n_engines(self) -> int:
        """Number of engines."""
        return self.data.n_engines

    @property
    def engine_name(self) -> str:
        """Engine name."""
        return self.data.engine_name

    @requires("altitude", "true_airspeed")
    def thrust_max(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate maximum available thrust.

        Thrust depends on engine_setting:
        - MTKF (Max Takeoff): Use takeoff thrust (rated max)
        - Otherwise: Use climb thrust at given ROC (original OpenAP behavior)

        Args:
            data: Flight state with:
                - altitude: Altitude [m]
                - true_airspeed: True airspeed [m/s]
                - vertical_speed: Rate of climb [m/s] (for climb)
                - engine_setting: EngineSetting enum (optional)

        Returns:
            Maximum thrust [N].
        """
        from openleaf.performance.core.perfos import EngineSetting

        altitude_ft = data["altitude"] * _FT
        tas_kt = data["true_airspeed"] * _KT

        # Get engine setting
        engine_setting = data.get("engine_setting")

        # Get vertical speed for climb thrust calculation
        vs = data.get("vertical_speed", 0.0)
        roc_fpm = vs * _FPM

        # MTKF: takeoff thrust (rated max)
        # Otherwise: climb thrust at given ROC (original OpenAP behavior)
        takeoff_thrust = self._thrust.takeoff(tas=tas_kt, alt=altitude_ft)
        climb_thrust = self._thrust.climb(tas=tas_kt, alt=altitude_ft, roc=roc_fpm)

        return np.where(
            np.equal(engine_setting, EngineSetting.MTKF),
            takeoff_thrust,
            climb_thrust,  # Default: climb thrust (original OpenAP)
        )

    @requires("altitude", "true_airspeed")
    def thrust_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum (idle) thrust.

        Args:
            data: Flight state with:
                - altitude: Altitude [m]
                - true_airspeed: True airspeed [m/s]

        Returns:
            Minimum thrust [N].
        """
        altitude_ft = data["altitude"] * _FT
        tas_kt = data["true_airspeed"] * _KT
        return self._thrust.descent_idle(tas=tas_kt, alt=altitude_ft)

    @requires("thrust")
    def fuel_flow(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate fuel flow rate from thrust.

        Args:
            data: Flight state with:
                - thrust: Thrust [N]

        Returns:
            Fuel flow rate [kg/s].
        """
        return self._fuel_flow.at_thrust(data["thrust"])

    @requires("altitude", "true_airspeed")
    def fuel_flow_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum (idle) fuel flow rate.

        Args:
            data: Flight state with:
                - altitude: Altitude [m]
                - true_airspeed: True airspeed [m/s]

        Returns:
            Minimum fuel flow rate [kg/s].
        """
        idle_thrust = self.thrust_min(data)
        return self._fuel_flow.at_thrust(idle_thrust)

    def fuel_flow_enroute(
        self,
        mass: ArrayScalarLike,
        true_airspeed: ArrayScalarLike,
        altitude: ArrayScalarLike,
        vertical_speed: ArrayScalarLike,
        acceleration: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Calculate fuel flow using OpenAP's enroute model.

        This is a convenience method for cases where thrust is not known.
        OpenAP internally computes drag and thrust for this calculation.

        Args:
            mass: Aircraft mass [kg].
            true_airspeed: True airspeed [m/s].
            altitude: Altitude [m].
            vertical_speed: Rate of climb [m/s].
            acceleration: Longitudinal acceleration [m/s²].

        Returns:
            Fuel flow rate [kg/s].
        """
        return self._fuel_flow.enroute(
            mass=mass,
            tas=true_airspeed * _KT,
            alt=altitude * _FT,
            vs=vertical_speed * _FPM,
            acc=acceleration,
        )


# ==============================================================================
# Aircraft Model
# ==============================================================================
class OpenAPModel(AircraftModel):
    """Complete OpenAP aircraft model.

    Wraps OpenAP library to provide aircraft performance calculations
    following the openleaf interface patterns.
    """

    def __init__(self, icao: str) -> None:
        """Initialize OpenAP model from ICAO aircraft type code.

        Args:
            icao: ICAO aircraft type code (e.g., "A320", "B738").

        Raises:
            ValueError: If aircraft not found in OpenAP database.
            ModelDataError: If the OpenAP database entry is missing a value
                required by openleaf (some types ship with null fields).
        """
        logger.info("Loading OpenAP model for %s", icao)
        self._model_name = icao

        from openap import prop

        ac_data = prop.aircraft(icao)

        def require_number(key: str) -> float:
            """Fetch a numeric root-level field, with a clear error for nulls."""
            value = ac_data.get(key)
            if value is None:
                raise ModelDataError(
                    f"{icao}: OpenAP database entry has no value for '{key}'; "
                    "this aircraft cannot be used with the OpenAP backend"
                )
            return float(value)

        # Determine engine type
        engine_type_map = {
            "turbofan": EngineType.JET,
            "turbojet": EngineType.JET,
            "turboprop": EngineType.TURBOPROP,
            "piston": EngineType.PISTON,
        }
        engine_type = engine_type_map.get(ac_data["engine"]["type"], EngineType.JET)

        mtow = require_number("mtow")
        oew = require_number("oew")
        mlw = require_number("mlw")

        # Build data structures from OpenAP - NO DEFAULTS
        # Note: OpenAP has values at root level (lowercase) and in 'limits' (uppercase)
        aircraft_data = OpenAPAircraftData(
            model=icao,
            wtc=self._get_wtc(mtow),
            engine_type=engine_type,
            n_engines=int(ac_data["engine"]["number"]),
            engine_name=ac_data["engine"]["default"],
            vmo=require_number("vmo"),
            mmo=require_number("mmo"),
            hmo=require_number("ceiling") * _FT,
            oem=oew,
            mtom=mtow,
            # OpenAP provides no MZFW/max-payload; mlw - oew is the standard
            # upper bound for payload. (Never mtow - oew - mfc: mfc is a fuel
            # VOLUME in litres, not a mass.)
            mpl=max(mlw - oew, 0.0),
            mlm=mlw,
            mfc=require_number("mfc"),
        )

        engine_data = OpenAPEngineData(
            n_engines=int(ac_data["engine"]["number"]),
            engine_name=ac_data["engine"]["default"],
        )

        aero_data = OpenAPAerodynamicsData(
            surface_area=float(ac_data["wing"]["area"]),
            wing_span=float(ac_data["wing"]["span"]),
        )

        limits_data = OpenAPLimitsData(
            vmo=aircraft_data.vmo,
            mmo=aircraft_data.mmo,
            hmo=aircraft_data.hmo,
            mtom=aircraft_data.mtom,
            mlm=aircraft_data.mlm,
            oem=aircraft_data.oem,
            mfc=aircraft_data.mfc,
            mpl=aircraft_data.mpl,
        )

        self._data = OpenAPData(
            aircraft=aircraft_data,
            engine=engine_data,
            aerodynamics=aero_data,
            limits=limits_data,
        )

        aero = OpenAPAeroModel(icao, aero_data)
        engine = OpenAPEngineModel(icao, engine_data)
        limits = OpenAPLimits(limits_data)

        super().__init__(aircraft_data, engine, aero, limits)
        logger.info(
            "OpenAP model initialized: %s (engine=%s, mtom=%.0f kg)",
            icao,
            engine_type.name,
            aircraft_data.mtom,
        )

    @staticmethod
    def _get_wtc(mtow: float) -> str:
        """Get wake turbulence category from MTOW.

        Uses ICAO Doc 4444 PANS-ATM thresholds:
        - Heavy (H): MTOW > 136,000 kg
        - Medium (M): 7,000 kg < MTOW <= 136,000 kg
        - Light (L): MTOW <= 7,000 kg
        """
        if mtow > _WTC_HEAVY_THRESHOLD_KG:
            return "H"
        elif mtow > _WTC_MEDIUM_THRESHOLD_KG:
            return "M"
        return "L"

    @classmethod
    def available_aircraft(cls) -> list[str]:
        """Return list of available aircraft types in OpenAP.

        Raises:
            ImportError: If the optional openap library is not installed.
        """
        try:
            from openap import prop
        except ImportError as e:
            raise ImportError(
                "openap is required for the OpenAP backend. "
                "Install with: pip install openleaf-performance[openap]"
            ) from e

        return prop.available_aircraft()

    @classmethod
    def from_file(cls, file: str, **kwargs: object) -> OpenAPModel:
        """Load aircraft model from OpenAP database.

        For OpenAP, 'file' is the aircraft type code (e.g., "A320").

        Args:
            file: ICAO aircraft type code.
            **kwargs: Unused (for API compatibility).

        Returns:
            Initialized OpenAP aircraft model.
        """
        return cls(file)

    @property
    def model(self) -> str:
        """Aircraft model name."""
        return self._model_name

    @property
    def openap_data(self) -> OpenAPData:
        """Complete OpenAP data container."""
        return self._data

    def lift_coefficient(
        self,
        mass: ArrayScalarLike,
        air_pressure: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate lift coefficient for level flight.

        Args:
            mass: Aircraft mass [kg].
            air_pressure: Air pressure [Pa].
            mach: Mach number.

        Returns:
            Lift coefficient [-].
        """
        weight = mass * units.G
        q = 0.5 * units.GAMMA * air_pressure * mach**2
        return weight / (q * self.aerodynamics.surface_area)

    def compute_thrust_required(
        self,
        mass: ArrayScalarLike,
        cl: ArrayScalarLike,
        cd: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate thrust required for level flight (T = W * CD/CL).

        Args:
            mass: Aircraft mass [kg].
            cl: Lift coefficient [-].
            cd: Drag coefficient [-].

        Returns:
            Required thrust [N].
        """
        weight = mass * units.G
        return weight * cd / cl
