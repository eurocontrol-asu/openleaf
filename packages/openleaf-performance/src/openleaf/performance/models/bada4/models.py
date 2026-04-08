"""
BADA4 (Base of Aircraft Data 4) model implementations.

Provides engine, aerodynamic, and aircraft performance models for BADA4.

Reference: EUROCONTROL BADA Family 4 User Manual

Usage:
    # Initialize from ICAO code (requires configuration)
    model = BADA4Model("A320")

    # Initialize from specific file
    model = BADA4Model.from_file("/path/to/A320-214.xml")

    # Initialize from pre-loaded data
    model = BADA4Model.from_data(bada4_data)

    # List available aircraft
    aircraft = BADA4Model.available_aircraft()
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import openleaf.performance.numpy as np
import openleaf.performance.physics.isa as isa
from openleaf.performance.config import get_bada4_path
from openleaf.performance.config import get_bada4_synonyms_path
from openleaf.performance.core.perfos import AerodynamicsModel
from openleaf.performance.core.perfos import AircraftModel
from openleaf.performance.core.perfos import EngineModel
from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.perfos import EngineType
from openleaf.performance.core.perfos import FlightPhase
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import LandingGearPosition
from openleaf.performance.core.validation import requires
from openleaf.performance.exceptions import MissingFieldError
from openleaf.performance.logging import get_logger
from openleaf.performance.models.bada4.parsing import parse_bada4_file
from openleaf.performance.models.bada4.schemas import BADA4AerodynamicsData
from openleaf.performance.models.bada4.schemas import BADA4ARPMConfig
from openleaf.performance.models.bada4.schemas import BADA4Data
from openleaf.performance.models.bada4.schemas import BADA4EngineData
from openleaf.performance.models.bada4.synonyms import available_bada4_aircraft
from openleaf.performance.models.bada4.synonyms import resolve_bada4_file
from openleaf.performance.models.common import AERO_CONFIG_THRESHOLDS
from openleaf.performance.models.common.config import select_by_config
from openleaf.performance.physics.units import EPSILON_CLAMP
from openleaf.performance.physics.units import HP
from openleaf.performance.physics.units import KT
from openleaf.performance.physics.units import M_TO_FT
from openleaf.performance.physics.units import MS_TO_KT


logger = get_logger(__name__)

data_path = get_bada4_path()
synonym_path = get_bada4_synonyms_path()

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = [
    "BADA4AeroModel",
    "BADA4EngineModel",
    "BADA4JetEngineModel",
    "BADA4Model",
    "BADA4PistonEngineModel",
    "BADA4PropellerModel",
    "BADA4TurbineModel",
    "BADA4TurbopropEngineModel",
    "poly_eval",
]

# Threshold for detecting infinity (CasADi-compatible alternative to np.isinf)
# Used because np.isinf doesn't work with CasADi symbolic types
_INF_THRESHOLD = 1e10

# Default CL_max fallback when configuration CL_max is undefined (infinite)
_CL_MAX_FALLBACK = 2.0


def poly_eval(x: list, y: list, c: npt.NDArray):
    """Evaluate a bivariate polynomial using pre-computed basis function values.

    Computes: sum_i sum_j c[i,j] * x_i * y_j
    where c is stored row-major and x_i, y_j are caller-supplied basis values
    (e.g. powers of a variable: [z**0, z**1, z**2, ...]).

    Args:
        x: List of n_x basis evaluations, e.g. [1, delta, delta**2, ...].
        y: List of n_y basis evaluations, e.g. [1, mach, mach**2, ...].
        c: Coefficient array of length n_x * n_y, stored row-major so that
           c[i * n_y + j] is the coefficient for (x[i] * y[j]).

    Returns:
        Scalar or array of polynomial values, same type as x/y.
    """
    coef = c.reshape((len(x), len(y)))

    # If ANY element in x or y is a CasADi type, route the entire computation
    # through CasADi to avoid mixed numpy/CasADi shape mismatches.  This can
    # happen when some inputs (e.g. delta from ISA pressure) are plain floats
    # while others (e.g. mach derived from a CasADi interpolant) are ca.DM or
    # ca.MX.
    if np.is_casadi_type(x + y, recursive=True):
        import casadi as ca

        x_mat = ca.horzcat(*x)  # (1, n_x)
        y_mat = ca.horzcat(*y)  # (1, n_y)
        return ca.sum2(ca.mtimes(x_mat, coef) * y_mat)

    x = np.stack(x, axis=-1)
    y = np.stack(y, axis=-1)

    # coef is always a plain numpy array (parsed from BADA4 data). When x/y are
    # torch tensors or JAX arrays the matmul would trigger a NumPy 2.0
    # __array_wrap__ DeprecationWarning (and eventually an error). Convert coef
    # to the same framework as x before the matmul.
    xns = np.get_array_namespace(x)
    coef = xns.asarray(coef)

    return np.sum((x @ coef) * y, axis=-1)


class BADA4EngineModel(EngineModel[BADA4EngineData]):
    """BADA4 engine model with thrust and fuel flow calculations.

    Supports jet, turboprop, and piston engines with detailed performance
    coefficients for different throttle settings and flight conditions.
    """

    # Common methods
    def _ct_to_thrust(
        self,
        *,
        ct: ArrayScalarLike,
        delta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Convert thrust coefficient to actual thrust.

        Args:
            ct: Thrust coefficient (dimensionless).
            delta: Air pressure ratio (dimensionless).

        Returns:
            Thrust in Newtons.
        """
        return ct * delta * self.data.wref

    def _cf_to_fuel_flow(
        self,
        *,
        cf: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Convert fuel flow coefficient to actual fuel flow.

        Args:
            cf: Fuel flow coefficient (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).

        Returns:
            Fuel flow in kg/s.
        """
        return delta * np.sqrt(theta) * self.data.wref * isa.A_SL * cf / self.data.lhv

    def thrust_max_from_inputs(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        setting: ArrayScalarLike = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Calculate maximum available thrust from flight conditions.

        Args:
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).
            setting: Engine throttle setting (default: max climb).
            delta_isa: Temperature deviation from ISA (K).

        Returns:
            Maximum available thrust in Newtons.
        """
        ct = np.where(
            np.equal(setting, EngineSetting.MCRZ),
            self._thrust_max_coef(
                mach=mach,
                delta=delta,
                theta=theta,
                setting=EngineSetting.MCRZ,
                delta_isa=delta_isa,
            ),
            self._thrust_max_coef(
                mach=mach,
                delta=delta,
                theta=theta,
                setting=EngineSetting.MCMB,
                delta_isa=delta_isa,
            ),
        )

        return self._ct_to_thrust(
            ct=ct,
            delta=delta,
        )

    def thrust_min_from_inputs(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate minimum available thrust from flight conditions.

        Args:
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).

        Returns:
            Minimum available thrust in Newtons.
        """
        ct = self._thrust_min_coef(
            mach=mach,
            delta=delta,
            theta=theta,
        )

        return self._ct_to_thrust(
            ct=ct,
            delta=delta,
        )

    def fuel_flow_from_inputs(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        thrust: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate fuel flow from flight conditions and thrust.

        Args:
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).
            thrust: Actual thrust in Newtons.

        Returns:
            Fuel flow rate in kg/s.
        """
        ct = thrust / (delta * self.data.wref)

        cf = self._fuel_flow_coef(
            ct=ct,
            mach=mach,
            delta=delta,
            theta=theta,
        )

        return self._cf_to_fuel_flow(
            cf=cf,
            delta=delta,
            theta=theta,
        )

    def fuel_flow_min_from_inputs(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate minimum fuel flow from flight conditions.

        Args:
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).

        Returns:
            Minimum fuel flow rate in kg/s.
        """
        cf = self._fuel_flow_min_coef(
            mach=mach,
            theta=theta,
            delta=delta,
        )
        return self._cf_to_fuel_flow(
            cf=cf,
            delta=delta,
            theta=theta,
        )

    def _get_atm_ratios(self, data: KeyedData) -> tuple[ArrayScalarLike, ArrayScalarLike]:
        """Compute atmospheric ratios from actual air pressure and temperature.

        BADA4 requires actual atmospheric conditions - no ISA assumptions.

        Args:
            data: Flight state dictionary with:
                - air_pressure: Actual air pressure [Pa] (required)
                - air_temperature: Actual air temperature [K] (required)

        Returns:
            Tuple of (delta, theta) - pressure and temperature ratios.

        Note:
            This is a private method; callers must validate inputs exist.
        """
        delta = data["air_pressure"] / isa.P_SL
        theta = data["air_temperature"] / isa.TAU_SL
        return delta, theta

    @requires("mach_number", "air_pressure", "air_temperature", "engine_setting", "delta_isa")
    def thrust_max(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum thrust from flight state dictionary.

        Args:
            data: Flight state data dictionary with:
                - mach_number: Mach number (required)
                - air_pressure: Actual air pressure [Pa] (required)
                - air_temperature: Actual air temperature [K] (required)
                - engine_setting: Engine setting enum (required)
                - delta_isa: Temperature deviation from ISA [K] (required)

        Returns:
            Maximum available thrust in Newtons.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        delta, theta = self._get_atm_ratios(data)

        logger.debug("Computing thrust_max with setting=%s", data["engine_setting"])

        return self.thrust_max_from_inputs(
            mach=data["mach_number"],
            delta=delta,
            theta=theta,
            setting=data["engine_setting"],
            delta_isa=data["delta_isa"],
        )

    @requires("mach_number", "air_pressure", "air_temperature")
    def thrust_min(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum thrust from flight state dictionary.

        Args:
            data: Flight state data dictionary with:
                - mach_number: Mach number (required)
                - air_pressure: Actual air pressure [Pa] (required)
                - air_temperature: Actual air temperature [K] (required)

        Returns:
            Minimum available thrust in Newtons.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        delta, theta = self._get_atm_ratios(data)

        logger.debug("Computing thrust_min")

        return self.thrust_min_from_inputs(
            mach=data["mach_number"],
            delta=delta,
            theta=theta,
        )

    @requires("mach_number", "air_pressure", "air_temperature", "thrust")
    def fuel_flow(self, data: KeyedData) -> ArrayScalarLike:
        """Get fuel flow from flight state dictionary.

        Args:
            data: Flight state data dictionary with:
                - mach_number: Mach number (required)
                - air_pressure: Actual air pressure [Pa] (required)
                - air_temperature: Actual air temperature [K] (required)
                - thrust: Actual thrust in Newtons (required)

        Returns:
            Fuel flow rate in kg/s.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        delta, theta = self._get_atm_ratios(data)

        logger.debug("Computing fuel_flow")

        return self.fuel_flow_from_inputs(
            mach=data["mach_number"],
            delta=delta,
            theta=theta,
            thrust=data["thrust"],
        )

    @requires("mach_number", "air_pressure", "air_temperature")
    def fuel_flow_min(self, data: KeyedData) -> ArrayScalarLike:
        """Get minimum fuel flow from flight state dictionary.

        Args:
            data: Flight state data dictionary with:
                - mach_number: Mach number (required)
                - air_pressure: Actual air pressure [Pa] (required)
                - air_temperature: Actual air temperature [K] (required)

        Returns:
            Minimum fuel flow rate in kg/s.

        Raises:
            MissingFieldError: If any required field is missing.
        """
        delta, theta = self._get_atm_ratios(data)

        logger.debug("Computing fuel_flow_min")

        return self.fuel_flow_min_from_inputs(
            mach=data["mach_number"],
            delta=delta,
            theta=theta,
        )

    @abstractmethod
    def _throttle(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Calculate throttle setting (abstract method).

        Args:
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).
            mach: Mach number (dimensionless).
            setting: Engine throttle setting.
            delta_isa: Temperature deviation from ISA (K).

        Returns:
            Throttle setting value.
        """

    @abstractmethod
    def _thrust_min_coef(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate minimum thrust coefficient (abstract method).

        Args:
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).

        Returns:
            Minimum thrust coefficient (dimensionless).
        """

    @abstractmethod
    def _thrust_max_coef(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Calculate maximum thrust coefficient (abstract method).

        Args:
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).
            setting: Engine throttle setting.
            delta_isa: Temperature deviation from ISA (K).

        Returns:
            Maximum thrust coefficient (dimensionless).
        """

    @abstractmethod
    def _fuel_flow_min_coef(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate minimum fuel flow coefficient (abstract method).

        Args:
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).

        Returns:
            Minimum fuel flow coefficient (dimensionless).
        """

    @abstractmethod
    def _fuel_flow_coef(
        self,
        *,
        ct: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Calculate fuel flow coefficient (abstract method).

        Args:
            ct: Thrust coefficient (dimensionless).
            mach: Mach number (dimensionless).
            delta: Air pressure ratio (dimensionless).
            theta: Temperature ratio (dimensionless).

        Returns:
            Fuel flow coefficient (dimensionless).
        """


class BADA4TurbineModel(BADA4EngineModel):
    """BADA4 turbine engine model base class.

    Implements polynomial coefficient calculations for turbine engines
    (jet and turboprop). Provides methods for thrust and fuel flow
    coefficient computation.
    """

    def _poly_fuel_flow_min_coef(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return poly_eval(
            [np.power(mach, i) for i in range(3)],
            [np.power(delta, j) for j in range(3)],
            self.data.fi[:9],
        )

    def _poly_fuel_flow_gen_coef(
        self,
        *,
        x: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return poly_eval(
            [np.power(mach, i) for i in range(5)],
            [np.power(x, j) for j in range(5)],
            self.data.f[:25],
        )

    def _poly_thrust_min_coef(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return poly_eval(
            [np.power(mach, i) for i in range(3)],
            [np.power(delta, j - 1) for j in range(4)],
            self.data.ti[:12],
        )

    def _fuel_flow_gen_coef(
        self,
        *,
        x: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return self._poly_fuel_flow_gen_coef(x=x, mach=mach)

    def _clipped_fuel_flow_coef(
        self,
        *,
        x: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        cf = self._fuel_flow_gen_coef(
            x=x,
            mach=mach,
        )
        cf_min = self._fuel_flow_min_coef(
            mach=mach,
            delta=delta,
            theta=theta,
        )
        return np.clip(cf, min=cf_min, max=+np.inf)


class BADA4PropellerModel(BADA4EngineModel):
    """BADA4 propeller engine model base class.

    Implements propeller-specific calculations including advance ratio
    and power conversions. Base class for turboprop and piston engines.
    """

    def _mu(
        self,
        *,
        wp: ArrayScalarLike,
        sigma: ArrayScalarLike,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Solve cubic equation for propeller efficiency using Cardano's formula.

        Solves: eta^3 + p*eta + q = 0
        where p = (n_eng*sigma*rho0*D^2*pi*V^3*eta_max) / (2*WP)
        and q = -p * eta_max

        Uses Cardano's formula: eta = cbrt(-q/2 + sqrt(D)) + cbrt(-q/2 - sqrt(D))
        where D = (q/2)^2 + (p/3)^3
        """
        # Guard against zero/near-zero shaft power (idle condition):
        # When wp≈0, propeller produces no thrust → efficiency is undefined but
        # we clamp to avoid NaN from inf-inf in Cardano's formula.
        wp_safe = np.maximum(wp, 1e-12)

        p = (
            (self.data.engines / wp_safe)
            * sigma
            * isa.RHO_SL
            * self.data.prop_dia
            * self.data.prop_dia
            * np.pi
            * v
            * v
            * v
            * self.data.max_eff
            / 2.0
        )

        q = -p * self.data.max_eff

        d = np.power(q / 2.0, 2) + np.power(p / 3.0, 3)

        # Cardano's formula: x = cbrt(-q/2 + sqrt(d)) + cbrt(-q/2 - sqrt(d))
        neg_q_half = -q / 2.0
        eta = np.cbrt(neg_q_half + np.sqrt(d)) + np.cbrt(neg_q_half - np.sqrt(d))

        # When wp was effectively zero, return 0 efficiency
        return np.where(wp < 1e-12, 0.0, eta)

    @abstractmethod
    def _wp_to_cp(
        self,
        *,
        wp: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        pass

    @abstractmethod
    def _ct_to_cp(
        self,
        *,
        ct: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        pass

    @abstractmethod
    def _throttle_to_cp(
        self,
        *,
        throttle: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        pass


class BADA4JetEngineModel(BADA4TurbineModel):
    """BADA4 jet engine model.

    Implements thrust and fuel flow calculations for jet engines using
    BADA4 turbine performance coefficients.
    """

    def _fuel_flow_min_coef(
        self,
        *,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return self._poly_fuel_flow_min_coef(
            mach=mach,
            delta=delta,
        ) / (delta * np.sqrt(theta))

    def _fuel_flow_coef(
        self,
        *,
        ct: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return self._clipped_fuel_flow_coef(
            x=ct,
            mach=mach,
            delta=delta,
            theta=theta,
        )

    def _thrust_min_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return self._poly_thrust_min_coef(mach=mach, delta=delta)

    def _throttle(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        b = self.data.b[setting]
        c = self.data.c[setting]
        kink = self.data.kink[setting]

        # Flat-rated area
        throttle_flat = poly_eval(
            [np.power(delta, i) for i in range(6)],
            [np.power(mach, j) for j in range(6)],
            b[:36],
        )

        # Temperature-rated area
        theta_t = theta * (1.0 + (mach * mach) * (isa.GAMMA - 1.0) / 2.0)

        throttle_temp = poly_eval(
            [np.power(theta_t, i) for i in range(5)],
            [np.power(mach, j) for j in range(5)],
            c[:25],
        ) + poly_eval(
            [np.power(delta, i) for i in range(1, 5)],
            [np.power(mach, j) for j in range(5)],
            c[25:45],
        )

        # Select throttle based on kink point per formula 3.3-4:
        # When ΔT ≤ ΔTkink → flat-rated area
        # When ΔT > ΔTkink → temperature-rated area
        return np.where(
            np.less_equal(delta_isa, kink),
            throttle_flat,
            throttle_temp,
        )

    def _thrust_max_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        throttle = self._throttle(
            delta=delta,
            theta=theta,
            mach=mach,
            setting=setting,
            delta_isa=delta_isa,
        )
        return poly_eval(
            [np.power(throttle, i) for i in range(6)],
            [np.power(mach, j) for j in range(6)],
            self.data.a[:36],
        )


class BADA4TurbopropEngineModel(BADA4TurbineModel, BADA4PropellerModel):
    """BADA4 turboprop engine model.

    Implements thrust and fuel flow calculations for turboprop engines using
    both turbine and propeller performance coefficients.
    """

    def _fuel_flow_min_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
    ):
        return (
            self._poly_fuel_flow_min_coef(
                mach=mach,
                delta=delta,
            )
            + (
                self.data.fi[9] * theta
                + self.data.fi[10] * theta * theta
                + self.data.fi[11] * mach * theta
                + self.data.fi[12] * mach * delta * np.sqrt(theta)
                + self.data.fi[13] * mach * delta * theta
            )
        ) / (delta * np.sqrt(theta))

    def _fuel_flow_coef(
        self,
        *,
        ct: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ):
        cp = self._ct_to_cp(ct=ct, mach=mach)
        return self._clipped_fuel_flow_coef(
            x=cp,
            mach=mach,
            delta=delta,
            theta=theta,
        )

    def _thrust_min_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
    ):
        return (
            self._poly_thrust_min_coef(
                mach=mach,
                delta=delta,
            )
            + (
                self.data.ti[12] * np.sqrt(theta)
                + self.data.ti[13] * theta
                + self.data.ti[14] / np.sqrt(theta)
                + self.data.ti[15] * theta * theta
            )
            + (
                self.data.ti[16] / delta
                + self.data.ti[17] * delta
                + self.data.ti[18] * delta * delta
                + self.data.ti[19] * mach
                + self.data.ti[20] * mach * mach
            )
            / np.sqrt(theta)
            + (
                self.data.ti[21] / mach
                + self.data.ti[22] * delta / mach
                + self.data.ti[23] * mach * mach * mach
            )
            + (
                self.data.ti[24] * mach
                + self.data.ti[25] * mach * mach
                + self.data.ti[26]
                + self.data.ti[27] * mach / delta
            )
            / theta
            + (
                self.data.ti[28] * mach / (delta * theta * theta)
                + self.data.ti[29] * mach * mach / (delta * theta * theta)
                + self.data.ti[30] * mach * mach / (delta * np.sqrt(theta))
                + self.data.ti[31] * delta / theta
            )
        )

    def _thrust_max_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ):
        throttle = self._throttle(
            delta=delta,
            theta=theta,
            mach=mach,
            setting=setting,
            delta_isa=delta_isa,
        )

        cp = self._throttle_to_cp(
            throttle=throttle,
            mach=mach,
            delta=delta,
            theta=theta,
        )

        v = mach * isa.A_SL * np.sqrt(theta)  # TAS = Mach * a = Mach * a_SL * sqrt(theta)
        cp_max = self._wp_to_cp(
            wp=self.data.max_power[setting],
            delta=delta,
            theta=theta,
            v=v,
        )

        cp = np.clip(cp, -np.inf, cp_max)

        return cp / mach

    def _throttle(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        p = self.data.p[setting]
        return poly_eval(
            [np.power(theta, i) for i in range(6)],
            [np.power(mach, j) for j in range(6)],
            p[:36],
        )

    def _throttle_to_cp(
        self,
        *,
        throttle: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return poly_eval(
            [np.power(throttle, i) for i in range(6)],
            [np.power(mach, j) for j in range(6)],
            self.data.a[:36],
        )

    def _ct_to_cp(
        self,
        *,
        ct: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        return ct * mach

    def _wp_to_cp(
        self,
        *,
        wp: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        a = isa.A_SL * np.sqrt(theta)  # speed of sound = a_SL * sqrt(theta)
        sigma = delta / theta
        prop_eff = self._mu(
            wp=wp,
            sigma=sigma,
            v=v,
        )
        cp = wp * prop_eff / (delta * self.data.wref * a)
        return cp


class BADA4PistonEngineModel(BADA4PropellerModel):
    """BADA4 piston engine model (PEM) - Section 3.3.3.

    Implements thrust and fuel flow calculations for piston engines
    according to BADA4 specification.
    """

    # -------------------------------------------------------------------------
    # Power Coefficient Calculations (Section 3.3.3.2)
    # -------------------------------------------------------------------------

    def _cp_max_std_msl(self) -> ArrayScalarLike:
        """Maximum power coefficient at MSL in standard atmosphere (Eq. 3.3-16).

        CP_max,std,MSL = W_mref^-1 · a0^-1 · W_P1,max,std,MSL · n_eng
        """
        return self.data.p * HP * self.data.engines / (self.data.wref * isa.A_SL)

    def _cp_std_msl(
        self,
        throttle: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Power coefficient at MSL in standard atmosphere (Eq. 3.3-17).

        CP_std,MSL = CP_max,std,MSL · δT
        """
        return self._cp_max_std_msl() * throttle

    def _power_coef(
        self,
        *,
        throttle: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Power coefficient with turbocharger derating (Eq. 3.3-18).

        For turbocharged engines, power is maintained up to sigma_turbo,
        then derated at higher altitudes.
        """
        sigma = delta / theta
        cp_std_msl = self._cp_std_msl(throttle)
        cp_max_std_msl = self._cp_max_std_msl()

        # Turbocharger limits
        sigma_turbo = self.data.sigma_turbo
        delta_turbo = self.data.delta_turbo
        theta_turbo = self.data.theta_turbo

        # Derated power coefficient at high altitude (Eq. 3.3-18)
        cp_derated = np.minimum(
            cp_std_msl,
            cp_max_std_msl * delta / delta_turbo * np.sqrt(theta_turbo) / np.sqrt(theta),
        )

        # Select based on density ratio
        return np.where(
            np.greater_equal(sigma, sigma_turbo),
            cp_std_msl,
            cp_derated,
        )

    # -------------------------------------------------------------------------
    # Rating Model (Section 3.3.3.3)
    # -------------------------------------------------------------------------

    def _throttle(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Piston rating model (Eq. 3.3-19).

        δT = 1 for MCMB/MCRZ, δT = 0 for idle.
        """
        return np.where(
            np.equal(setting, EngineSetting.LIDL),
            0.0,
            1.0,
        )

    # -------------------------------------------------------------------------
    # Thrust Coefficient (Section 3.3.3.1)
    # -------------------------------------------------------------------------

    def _thrust_coef_from_cp(
        self,
        *,
        cp: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Thrust coefficient from power coefficient (Eq. 3.3-14).

        CT = η · a0 · CP · δ^-1 · V_TAS^-1
        """
        v = mach * isa.A_SL * np.sqrt(theta)  # TAS = Mach * a = Mach * a_SL * sqrt(theta)
        sigma = delta / theta

        # All-engine power: WP = Wmref · a0 · CP (Eq. 3.3-15)
        wp = self.data.wref * isa.A_SL * cp

        # Propeller efficiency (Eq. 3.3-20)
        eta = self._propeller_efficiency(wp=wp, sigma=sigma, v=v)

        # CT = η · a0 · CP · δ^-1 · V_TAS^-1
        return eta * isa.A_SL * cp / (delta * v)

    def _thrust_max_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
        setting: EngineSetting = EngineSetting.MCMB,
        delta_isa: ArrayScalarLike = 0.0,
    ) -> ArrayScalarLike:
        """Maximum thrust coefficient for given setting."""
        throttle = self._throttle(
            delta=delta,
            theta=theta,
            mach=mach,
            setting=setting,
            delta_isa=delta_isa,
        )
        cp = self._power_coef(throttle=throttle, delta=delta, theta=theta)
        return self._thrust_coef_from_cp(cp=cp, delta=delta, theta=theta, mach=mach)

    def _thrust_min_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Minimum (idle) thrust coefficient."""
        return self._thrust_max_coef(
            delta=delta,
            theta=theta,
            mach=mach,
            setting=EngineSetting.LIDL,
        )

    # -------------------------------------------------------------------------
    # Fuel Coefficient (Section 3.4.3)
    # -------------------------------------------------------------------------

    def _fuel_flow_coef(
        self,
        *,
        ct: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Fuel coefficient from thrust coefficient (Eq. 3.4-8).

        CF = CPSFC · CP · δ^-1 · θ^(-1/2)
        """
        cp = self._ct_to_cp(ct=ct, mach=mach, delta=delta, theta=theta)
        return self.data.cpsfc * cp / (delta * np.sqrt(theta))

    def _fuel_flow_min_coef(
        self,
        *,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Minimum (idle) fuel coefficient."""
        cp = self._power_coef(throttle=0.0, delta=delta, theta=theta)
        return self.data.cpsfc * cp / (delta * np.sqrt(theta))

    # -------------------------------------------------------------------------
    # Inverse: CT to CP (for fuel flow from thrust)
    # -------------------------------------------------------------------------

    def _ct_to_cp(
        self,
        *,
        ct: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Invert thrust coefficient to power coefficient.

        From CT = η · a0 · CP / (δ · V), solve for CP.

        By substituting WP = T·V/η into the propeller efficiency cubic
        A·η³ + η - η_max = 0, the equation simplifies to a quadratic:
        k·η² + η - η_max = 0, which is solved analytically.
        """
        v = mach * isa.A_SL * np.sqrt(theta)  # TAS = Mach * a = Mach * a_SL * sqrt(theta)
        sigma = delta / theta

        # Thrust from CT: T = δ · Wmref · CT
        thrust = ct * delta * self.data.wref

        # Coefficient for the quadratic: k*eta^2 + eta - eta_max = 0
        # k = 2*T / (n_eng * sigma * rho0 * D^2 * pi * V^2 * eta_max)
        k = (
            2.0
            * thrust
            / (
                self.data.engines
                * sigma
                * isa.RHO_SL
                * self.data.prop_dia**2
                * np.pi
                * v**2
                * self.data.max_eff
            )
        )

        # Solve quadratic: k·η² + η - η_max = 0
        # η = (-1 + sqrt(1 + 4·k·η_max)) / (2·k)
        # Numerically stable form (avoids 0/0 when k→0):
        # η = 2·η_max / (1 + sqrt(1 + 4·k·η_max))
        discriminant = 1.0 + 4.0 * k * self.data.max_eff
        eta = 2.0 * self.data.max_eff / (1.0 + np.sqrt(discriminant))

        # Clip efficiency to valid range
        eta = np.clip(eta, min=0.01, max=self.data.max_eff)

        # Power: WP = T · V / η
        wp = thrust * v / eta

        # CP = WP / (Wmref · a0)
        return wp / (self.data.wref * isa.A_SL)

    # -------------------------------------------------------------------------
    # Propeller Efficiency (Section 3.3.4)
    # -------------------------------------------------------------------------

    def _propeller_efficiency(
        self,
        *,
        wp: ArrayScalarLike,
        sigma: ArrayScalarLike,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Propeller efficiency via momentum theory (Eq. 3.3-20, 3.3-21).

        Solves the cubic: 2*WP/(n_eng*sigma*rho0*D^2*pi*V^3*eta_max)*eta^3 + eta - eta_max = 0
        """
        return self._mu(wp=wp, sigma=sigma, v=v)

    # -------------------------------------------------------------------------
    # Abstract method implementations for PropellerModel interface
    # -------------------------------------------------------------------------

    def _wp_to_cp(
        self,
        *,
        wp: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
        v: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Convert power to power coefficient (Eq. 3.3-15 inverted)."""
        return wp / (self.data.wref * isa.A_SL)

    def _throttle_to_cp(
        self,
        *,
        throttle: ArrayScalarLike,
        mach: ArrayScalarLike,
        delta: ArrayScalarLike,
        theta: ArrayScalarLike,
    ) -> ArrayScalarLike:
        """Convert throttle setting to power coefficient."""
        return self._power_coef(throttle=throttle, delta=delta, theta=theta)


class BADA4AeroModel(AerodynamicsModel[BADA4AerodynamicsData]):
    """BADA4 aerodynamic model.

    Implements drag coefficient calculations for different aircraft
    configurations and flight conditions using BADA4 aerodynamic database.

    BADA4 uses (hlid, lg) tuples to identify aerodynamic configurations:
    - hlid: High-lift device ID ("0" for clean, "1", "2", etc. for extended)
    - lg: Landing gear position (LandingGearPosition.UP or DOWN)

    The mapping from AerodynamicConfig enum to (hlid, lg) is aircraft-specific
    and defined in the ARPM (Airline Reference Procedure Model) configuration.
    """

    def __init__(self, data, arpm: BADA4ARPMConfig | None = None):
        """Initialize BADA4 aerodynamic model.

        Args:
            data: Aerodynamic data containing coefficients and surface area.
            arpm: Optional ARPM configuration for config-to-hlid/lg mapping.
        """
        super().__init__(data)
        self._arpm = arpm

    @requires("lift_coef", "mach_number")
    def drag_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate drag coefficient from flight state data.

        Extracts lift coefficient, Mach number, and configuration from
        the data dict and computes drag using BADA4 aerodynamic database.

        Configuration must be specified in one of two ways:
        1. Explicit hlid and lg in data dict
        2. aerodynamic_configuration enum in data (mapped via ARPM)

        Args:
            data: Dictionary containing:
                - lift_coef: Lift coefficient (required)
                - mach_number: Mach number (required)
                - aerodynamic_configuration: AerodynamicConfig enum, OR
                - hlid: High-lift device ID AND lg: Landing gear position

        Returns:
            Drag coefficient (dimensionless).

        Raises:
            MissingFieldError: If required fields are missing or configuration
                is not specified.
        """
        cl = data["lift_coef"]
        mach = data["mach_number"]

        # Check for hlid/lg - try to access them
        hlid = data.get("hlid")
        lg = data.get("lg")
        if hlid is not None and lg is not None:
            pass  # Use explicit hlid/lg
        elif self._arpm is not None:
            # Try to get aerodynamic_configuration (may be computed property)
            try:
                config = data["aerodynamic_configuration"]
                hlid, lg = self._hlid_lg_from_config(config)
            except KeyError:
                raise MissingFieldError(
                    field="aerodynamic_configuration",
                    method="drag_coef",
                    message="drag_coef requires either 'aerodynamic_configuration' or both 'hlid' and 'lg'",
                ) from None
        else:
            raise MissingFieldError(
                field="aerodynamic_configuration",
                method="drag_coef",
                message="drag_coef requires either 'aerodynamic_configuration' or both 'hlid' and 'lg'",
            )

        logger.debug("Computing drag_coef with hlid=%s, lg=%s", hlid, lg)

        return self.drag_coef_from_inputs(cl=cl, mach=mach, hlid=hlid, lg=lg)

    def _hlid_lg_from_config(
        self, config: ArrayScalarLike
    ) -> tuple[ArrayScalarLike, ArrayScalarLike]:
        """Convert AerodynamicConfig to (hlid, lg) using ARPM mappings.

        Args:
            config: AerodynamicConfig enum value(s).

        Returns:
            Tuple of (hlid, lg) strings.
        """
        # Build hlid mapping from ARPM with defaults
        hlid_mapping = {
            AerodynamicConfig.CR: self._arpm.hlid_map.get(AerodynamicConfig.CR, "0"),
            AerodynamicConfig.IC: self._arpm.hlid_map.get(AerodynamicConfig.IC, "0"),
            AerodynamicConfig.TO: self._arpm.hlid_map.get(AerodynamicConfig.TO, "0"),
            AerodynamicConfig.AP: self._arpm.hlid_map.get(AerodynamicConfig.AP, "0"),
            AerodynamicConfig.LD: self._arpm.hlid_map.get(AerodynamicConfig.LD, "0"),
        }

        # Build lg mapping from ARPM with defaults
        lg_mapping = {
            AerodynamicConfig.CR: self._arpm.lg_map.get(
                AerodynamicConfig.CR, LandingGearPosition.UP
            ),
            AerodynamicConfig.IC: self._arpm.lg_map.get(
                AerodynamicConfig.IC, LandingGearPosition.UP
            ),
            AerodynamicConfig.TO: self._arpm.lg_map.get(
                AerodynamicConfig.TO, LandingGearPosition.UP
            ),
            AerodynamicConfig.AP: self._arpm.lg_map.get(
                AerodynamicConfig.AP, LandingGearPosition.UP
            ),
            AerodynamicConfig.LD: self._arpm.lg_map.get(
                AerodynamicConfig.LD, LandingGearPosition.DOWN
            ),
        }

        # Map config to hlid and lg using utility function
        hlid = select_by_config(config, hlid_mapping, default="0")
        lg = select_by_config(config, lg_mapping, default=LandingGearPosition.UP)

        return hlid, lg

    def drag_coef_from_inputs(
        self,
        *,
        cl: ArrayScalarLike,
        mach: ArrayScalarLike,
        conf: ArrayScalarLike | None = None,
        hlid: ArrayScalarLike | None = None,
        lg: ArrayScalarLike | None = None,
    ) -> ArrayScalarLike:
        """Calculate drag coefficient for given aerodynamic configuration.

        Accepts either:
        - conf: AerodynamicConfig enum (like BADA3 API), or
        - hlid + lg: BADA4 native high-lift device ID and landing gear position

        If conf is provided, it takes precedence and is converted to (hlid, lg)
        using the aircraft-specific ARPM mappings.

        Args:
            cl: Lift coefficient (scalar or array).
            mach: Mach number (scalar or array).
            conf: AerodynamicConfig enum (TO, IC, CR, AP, LD). Takes precedence
                over hlid/lg if provided.
            hlid: High-lift device ID ("0" for clean, "1", "2", etc.). Used only
                if conf is not provided. Defaults to "0" (clean).
            lg: Landing gear position (LandingGearPosition.UP or DOWN). Used only
                if conf is not provided. Defaults to UP (retracted).

        Returns:
            Drag coefficient (scalar or array matching input shapes).

        Example:
            >>> # Using AerodynamicConfig (unified API with BADA3)
            >>> cd = aero.drag_coef_from_inputs(cl=0.5, mach=0.78, conf=AerodynamicConfig.CR)
            >>> # Using hlid/lg (BADA4 native)
            >>> cd = aero.drag_coef_from_inputs(
            ...     cl=0.5, mach=0.78, hlid="0", lg=LandingGearPosition.UP
            ... )
        """
        # If conf provided, convert to hlid/lg using aircraft-specific mapping
        if conf is not None:
            hlid, lg = self._hlid_lg_from_config(conf)
        else:
            # Default to clean configuration if neither provided
            hlid = hlid if hlid is not None else "0"
            lg = lg if lg is not None else LandingGearPosition.UP

        # Start with clean configuration as base
        cd = self._drag_coef_clean(cl=cl, mach=mach)

        # Apply each non-clean configuration using vectorized np.where
        for (hlid_key, lg_key), d_coefs in self.data.d_nonclean.items():
            # Compute drag for this configuration
            cd_nc = d_coefs[0] + d_coefs[1] * cl + d_coefs[2] * cl * cl

            # Select where hlid and lg match this configuration
            # np.equal handles both scalar and array comparisons
            is_this_config = np.logical_and(
                np.equal(hlid, hlid_key),
                np.equal(lg, lg_key),
            )
            cd = np.where(is_this_config, cd_nc, cd)

        return cd

    def _base_drag_coef_clean(
        self,
        *,
        cl: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        # param = 1 - M² (Prandtl-Glauert factor squared)
        # Clamp to small positive value to avoid division by zero at M → 1
        # np.where evaluates both branches, so we need this guard
        param = np.maximum(1 - mach * mach, EPSILON_CLAMP)
        d = self.data.d_clean
        c_0 = (
            d[0]
            + d[1] / np.sqrt(param)
            + d[2] / (param)
            + d[3] / np.power(param, 3.0 / 2.0)
            + d[4] / np.power(param, 2.0)
        )
        c_2 = (
            d[5]
            + d[6] / np.power(param, 3.0 / 2.0)
            + d[7] / np.power(param, 3.0)
            + d[8] / np.power(param, 9.0 / 2.0)
            + d[9] / np.power(param, 6.0)
        )
        c_6 = (
            d[10]
            + d[11] / np.power(param, 7.0)
            + d[12] / np.power(param, 15.0 / 2.0)
            + d[13] / np.power(param, 8.0)
            + d[14] / np.power(param, 17.0 / 2.0)
        )
        return self.data.scalar * (c_0 + c_2 * cl * cl + c_6 * np.power(cl, 6))

    def _drag_coef_clean(
        self,
        *,
        cl: ArrayScalarLike,
        mach: ArrayScalarLike,
    ) -> ArrayScalarLike:
        cd_nom = self._base_drag_coef_clean(cl=cl, mach=mach)

        mach_maxp = self.data.mach_max - 0.01
        cd_maxp = self._base_drag_coef_clean(cl=cl, mach=mach_maxp)
        cd_max = self._base_drag_coef_clean(mach=self.data.mach_max, cl=cl)

        # Use np.maximum to avoid negative base in power (np.where evaluates both branches)
        mach_diff = np.maximum((mach - mach_maxp) / 0.01, 0.0)

        return np.where(
            mach < self.data.mach_max,
            cd_nom,
            cd_maxp + np.power(mach_diff, 3 / 2) * (cd_max - cd_maxp),
        )

    def valid_hlid_values(self) -> set[str]:
        """Get valid high-lift device IDs for this aircraft.

        Returns the set of HLid values that this aircraft supports. These
        values can be used with drag_coef_from_inputs(hlid=...).

        Note:
            HLid values are aircraft-specific. A typical mapping might be:
            - "0": Clean (no flaps/slats)
            - "1": Take-off configuration
            - "2": Approach configuration
            - "3": Landing configuration

            But actual values depend on the aircraft's aerodynamic data.

        Returns:
            Set of valid HLid strings for this aircraft.

        Example:
            >>> aero = BADA4AeroModel(data)
            >>> aero.valid_hlid_values()
            {'0', '1', '2', '3'}
        """
        # Clean config always valid
        hlids = {"0"}

        # Add all non-clean configurations
        for hlid, _ in self.data.d_nonclean:
            hlids.add(hlid)

        return hlids


# =============================================================================
# FULL BADA4 AIRCRAFT MODEL
# =============================================================================


class BADA4Model(AircraftModel):
    """Complete BADA4 aircraft performance model.

    Integrates aircraft data, aerodynamics, engine models, and limitations
    to provide full aircraft performance calculations according to BADA4.
    """

    def __init__(self, icao: str):
        """Initialize BADA4 model from ICAO aircraft type code.

        Args:
            icao: ICAO aircraft type code (e.g., 'A320', 'B738').

        Raises:
            ConfigurationError: If BADA4 data path is not configured.
            FileNotFoundError: If aircraft not found in BADA4 data.

        Example:
            >>> model = BADA4Model("A320")
            >>> model.mtom
            78000.0
        """
        logger.info("Loading BADA4 model for %s", icao)
        xml_file = resolve_bada4_file(icao, data_path, synonym_path)
        logger.debug("Resolved XML file: %s", xml_file)
        data = parse_bada4_file(str(xml_file))
        self._init_from_data(data)

    def _init_from_data(self, data: BADA4Data) -> None:
        """Initialize model from BADA4 data object.

        Args:
            data: Parsed BADA4 data.
        """
        engine: EngineModel

        if data.aircraft.engine_type == EngineType.JET:
            logger.debug("Initializing jet engine model")
            engine = BADA4JetEngineModel(data.engine)
        elif data.aircraft.engine_type == EngineType.TURBOPROP:
            logger.debug("Initializing turboprop engine model")
            engine = BADA4TurbopropEngineModel(data.engine)
        elif data.aircraft.engine_type == EngineType.PISTON:
            logger.debug("Initializing piston engine model")
            engine = BADA4PistonEngineModel(data.engine)
        else:
            logger.error("Unsupported engine type: %s", data.aircraft.engine_type)
            raise ValueError(f"Unsupported engine type: {data.aircraft.engine_type}")

        # Store ARPM config (before aero_model init so it can be passed)
        self._arpm = data.arpm if data.arpm is not None else BADA4ARPMConfig()

        # Pass ARPM to aero model for config-to-hlid/lg mapping
        aero_model = BADA4AeroModel(data.aero, arpm=self._arpm)
        super().__init__(data.aircraft, engine, aero_model, data.limits)
        logger.info(
            "BADA4 model initialized: %s (engine=%s, mtom=%.0f kg)",
            data.aircraft.model,
            data.aircraft.engine_type.name,
            data.aircraft.mtom,
        )

    @classmethod
    def from_data(cls, data: BADA4Data) -> BADA4Model:
        """Create BADA4 model from pre-loaded data.

        Args:
            data: Parsed BADA4 data object.

        Returns:
            Initialized BADA4 aircraft model.
        """
        instance = object.__new__(cls)
        instance._init_from_data(data)
        return instance

    @classmethod
    def from_file(cls, file: str | Path, **kwargs) -> BADA4Model:
        """Load BADA4 model from XML file.

        Args:
            file: Path to BADA4 XML data file.
            **kwargs: Additional arguments (unused, for interface consistency).

        Returns:
            Initialized BADA4 aircraft model.
        """
        data = parse_bada4_file(str(file))
        return cls.from_data(data)

    @classmethod
    def available_aircraft(cls) -> list[str]:
        """List available BADA4 aircraft types.

        Returns:
            Sorted list of ICAO codes for available aircraft.

        Raises:
            ConfigurationError: If BADA4 data path is not configured.
        """
        return available_bada4_aircraft(data_path, synonym_path)

    # -------------------------------------------------------------------------
    # Mass Properties (from aircraft data, not limits)
    # -------------------------------------------------------------------------

    @property
    def mtom(self) -> float:
        """Maximum takeoff mass [kg]."""
        return self.data.mtom

    @property
    def oem(self) -> float:
        """Operating empty mass [kg]."""
        return self.data.oem

    @property
    def mpl(self) -> float:
        """Maximum payload [kg]."""
        return self.data.mpl

    # -------------------------------------------------------------------------
    # Limitations Access
    # -------------------------------------------------------------------------

    # Note: limits attribute is inherited from AircraftModel base class
    # Do not define a limits property here as it would shadow the attribute

    @property
    def has_limits(self) -> bool:
        """Check if limitations are available."""
        return self.limits is not None

    def max_altitude_for(
        self,
        hl_deployed: ArrayScalarLike = False,
    ) -> ArrayScalarLike:
        """Get maximum operating altitude in feet (GLM).

        Convenience method for direct parameter access.

        Args:
            hl_deployed: Whether high-lift devices are deployed.

        Returns:
            Maximum altitude in feet.
        """
        if self.limits is None:
            return self.hmo
        return self.limits.max_altitude({"hl_deployed": hl_deployed})

    def max_cas_for(
        self,
        hl_position: str = "0",
        lg_extended: ArrayScalarLike = False,
    ) -> ArrayScalarLike:
        """Get maximum calibrated airspeed in knots (KLM).

        Convenience method for direct parameter access.

        Args:
            hl_position: High-lift devices position identifier.
            lg_extended: Whether landing gear is extended.

        Returns:
            Maximum CAS in knots.
        """
        if self.limits is None:
            return self.vmo
        return self.limits.max_cas(
            {
                "hl_position": hl_position,
                "lg_extended": lg_extended,
            }
        )

    def max_mach_for(
        self,
        lg_extended: ArrayScalarLike = False,
    ) -> ArrayScalarLike:
        """Get maximum Mach number (KLM).

        Convenience method for direct parameter access.

        Args:
            lg_extended: Whether landing gear is extended.

        Returns:
            Maximum Mach number.
        """
        if self.limits is None:
            return self.mmo if self.mmo is not None else np.inf
        return self.limits.max_mach({"lg_extended": lg_extended})

    def cl_max(
        self,
        mach: ArrayScalarLike,
        hl_position: str = "0",
        lg_position: LandingGearPosition = LandingGearPosition.UP,
    ) -> ArrayScalarLike:
        """Get maximum lift coefficient (BLM) by HL/LG position.

        Args:
            mach: Mach number (used for clean configuration).
            hl_position: High-lift devices position ("0" for clean).
            lg_position: Landing gear position (LandingGearPosition enum).

        Returns:
            Maximum lift coefficient.
        """
        if self.limits is None:
            return np.inf

        if hl_position == "0" and lg_position == LandingGearPosition.UP:
            # Clean configuration
            return self.limits.buffet.cl_max_clean(mach)
        else:
            # Non-clean configuration
            return self.limits.buffet.cl_max_nonclean_config(hl_position, lg_position)

    def cl_max_config(
        self,
        config: AerodynamicConfig,
        mach: ArrayScalarLike = 0.3,
    ) -> ArrayScalarLike:
        """Get maximum lift coefficient (BLM) by aerodynamic configuration.

        Uses the AeroConfSchedule mapping to look up CL_max for the given
        configuration. For clean configurations (CR, IC), uses the Mach-dependent
        buffet model. For non-clean configurations (TO, AP, LD), uses the
        configuration-mapped CL_max values.

        Args:
            config: Aerodynamic configuration.
            mach: Mach number (used for clean configuration only).

        Returns:
            Maximum lift coefficient.
        """
        if self.limits is None or self.limits.buffet is None:
            return np.inf

        return self.limits.cl_max_by_config(config, mach)

    # -------------------------------------------------------------------------
    # ARPM Configuration
    # -------------------------------------------------------------------------

    @property
    def arpm(self) -> BADA4ARPMConfig:
        """Airline Procedure Model configuration."""
        return self._arpm

    def stall_speed(
        self,
        mass: ArrayScalarLike,
        mach: ArrayScalarLike = 0.3,
        config: ArrayScalarLike = AerodynamicConfig.CR,
        load_factor: ArrayScalarLike = None,
    ) -> ArrayScalarLike:
        """Calculate stall speed in CAS from CL_max.

        Uses BLM (Buffet Limit Model) to get CL_max and calculates stall speed:
            V_stall = sqrt(2 * m * g * n / (rho0 * S * CL_max))

        For clean configuration with BLM model available, a 1.2g buffet margin
        is applied by default (n=1.2) per BADA4 Appendix B "Operational envelope
        limitations". For non-clean configurations, n=1.0 is used.

        Args:
            mass: Aircraft mass [kg].
            mach: Mach number for CL_max lookup (clean config only).
            config: Aerodynamic configuration (scalar or array).
            load_factor: Load factor n [-]. If None, uses 1.2 for clean config
                        with BLM model, 1.0 otherwise.

        Returns:
            Stall speed in knots CAS.

        Note:
            Fully vectorized - config can be scalar or array/CasADi symbolic.
        """
        # Determine if we're using clean configuration with BLM model (vectorized)
        is_clean = np.logical_or(
            np.equal(config, AerodynamicConfig.CR),
            np.equal(config, AerodynamicConfig.IC),
        )
        has_blm = (
            self.limits is not None
            and self.limits.buffet is not None
            and self.limits.buffet.has_clean_model
        )

        # Get CL_max for the configuration using config-based lookup
        cl_max = self.cl_max_config(config=config, mach=mach)

        # Handle case where CL_max is not available (threshold comparison for CasADi)
        cl_max = np.where(cl_max > _INF_THRESHOLD, _CL_MAX_FALLBACK, cl_max)

        # Determine load factor for buffet margin (vectorized)
        # Per BADA4 Appendix B: clean config with BLM uses n=1.2 (1.2g buffet margin)
        if load_factor is None:
            # Use np.where for vectorized selection: n=1.2 if clean AND has_blm, else 1.0
            n = np.where(is_clean, 1.2, 1.0) if has_blm else 1.0
        else:
            n = load_factor

        # Stall speed formula: V_stall = sqrt(2 * m * g * n / (rho0 * S * CL_max))
        v_stall_m_s = np.sqrt(
            2.0 * mass * isa.G * n / (isa.RHO_SL * self.aerodynamics.surface_area * cl_max)
        )

        return v_stall_m_s / KT

    def aerodynamic_configuration(self, data: KeyedData) -> ArrayScalarLike:
        """Determine BADA4 aerodynamic configuration from flight state (ARPM Table 2).

        The configuration is selected based on flight phase, altitude AGL,
        and calibrated airspeed according to BADA4 Section 8 / Appendix B.

        Minimum speeds are calculated from CL_max (BLM) and mass:
            V_stall = sqrt(2 * m * g / (rho0 * S * CL_max))
            V_min = CVmin * V_stall

        Configuration selection rules:
        - TO: Climb, altitude <= Hmax_TO
        - IC: Climb, Hmax_TO < altitude <= Hmax_IC
        - CR: Climb above Hmax_IC, or descent above Hmax_AP,
              or descent with CAS >= Vmin_CR + 10 kt
        - AP: Descent, Hmax_LD < altitude <= Hmax_AP, CAS < Vmin_CR + 10 kt
        - LD: Descent, altitude <= Hmax_LD, CAS < Vmin_AP + 10 kt

        Args:
            data: Flight state dictionary with keys:
                Required keys:
                - flight_phase: FlightPhase enum (CLIMB, CRUISE, DESCENT)
                - calibrated_airspeed: Calibrated airspeed [m/s]
                - aircraft_mass: Aircraft mass [kg]
                - mach_number: Mach number

                Optional keys:
                - aerodynamic_configuration: Explicit override (bypasses calculation)
                - altitude_agl: Altitude above ground level [ft] (preferred)
                - altitude: Geometric altitude [m] (fallback, converted to ft)

        Returns:
            AerodynamicConfig enum value (BADA-specific, vectorized).
        """
        # If explicit configuration provided, use it
        if "aerodynamic_configuration" in data:
            logger.debug("Using explicit aerodynamic_configuration from data")
            return data["aerodynamic_configuration"]

        # Determine flight phase from flight_phase key (required)
        phase = data["flight_phase"]

        # Get altitude AGL - prefer altitude_agl, fall back to altitude (converted)
        if "altitude_agl" in data:
            h_agl = data["altitude_agl"]
        else:
            altitude_m = data["altitude"]
            h_agl = altitude_m * M_TO_FT  # Convert to feet

        # Get calibrated airspeed (required)
        calibrated_airspeed = data["calibrated_airspeed"] * MS_TO_KT

        mass = data["aircraft_mass"]
        mach = data["mach_number"]

        # Calculate minimum speeds from CL_max
        vs_cr = self.stall_speed(mass, mach, AerodynamicConfig.CR)
        vmin_cr = self._arpm.cvmin * vs_cr

        vs_ap = self.stall_speed(mass, mach, AerodynamicConfig.AP)
        vmin_ap = self._arpm.cvmin * vs_ap

        # ARPM thresholds
        hmax_to = self._arpm.hmax_to
        hmax_ic = self._arpm.hmax_ic
        hmax_ap = self._arpm.hmax_ap
        hmax_ld = self._arpm.hmax_ld

        # Speed margins (from common thresholds)
        speed_margin = AERO_CONFIG_THRESHOLDS.speed_margin
        v_cr_margin = vmin_cr + speed_margin  # kt margin above Vmin_CR
        v_ap_margin = vmin_ap + speed_margin  # kt margin above Vmin_AP

        # Boolean conditions for flight phase
        is_descent = np.equal(phase, FlightPhase.DESCENT)
        is_cruise = np.equal(phase, FlightPhase.CRUISE)

        # Altitude conditions
        in_to_zone = np.less_equal(h_agl, hmax_to)
        in_ic_zone = np.logical_and(
            np.greater(h_agl, hmax_to),
            np.less_equal(h_agl, hmax_ic),
        )
        above_ap_zone = np.greater(h_agl, hmax_ap)
        in_ld_zone = np.less_equal(h_agl, hmax_ld)

        # Speed conditions for descent
        fast_for_cr = np.greater_equal(calibrated_airspeed, v_cr_margin)
        slow_for_ld = np.less(calibrated_airspeed, v_ap_margin)

        # Climb configuration selection (based on altitude only)
        climb_config = np.where(
            in_to_zone,
            AerodynamicConfig.TO,
            np.where(
                in_ic_zone,
                AerodynamicConfig.IC,
                AerodynamicConfig.CR,
            ),
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
            np.where(
                is_cruise,
                AerodynamicConfig.CR,
                climb_config,
            ),
        )

        return config

    def high_lift(self, data: KeyedData) -> ArrayScalarLike:
        """Determine high-lift device ID (HLid) from flight state.

        Returns the HLid string as parsed from the BADA4 XML AeroConfSchedule.
        The mapping is aircraft-specific and comes directly from the XML file.

        This is the native BADA4 representation used for indexing into
        GLM (Geometric Limits Model) and BLM (Buffet Limits Model) tables.

        Args:
            data: Flight state dictionary.

        Returns:
            HLid string from parsed XML (e.g., "0", "1", "2", etc.).

        Raises:
            ValueError: If HLid mapping is not available in parsed XML data.
        """
        if not self._arpm.hlid_map:
            raise ValueError(
                "HLid mapping not available. The BADA4 XML file does not contain "
                "AeroConfSchedule with HLid values, or the file was not parsed correctly."
            )

        config = self.aerodynamic_configuration(data)

        # Build hlid mapping from ARPM
        hlid_mapping = {
            AerodynamicConfig.CR: self._arpm.hlid_map.get(AerodynamicConfig.CR),
            AerodynamicConfig.IC: self._arpm.hlid_map.get(AerodynamicConfig.IC),
            AerodynamicConfig.TO: self._arpm.hlid_map.get(AerodynamicConfig.TO),
            AerodynamicConfig.AP: self._arpm.hlid_map.get(AerodynamicConfig.AP),
            AerodynamicConfig.LD: self._arpm.hlid_map.get(AerodynamicConfig.LD),
        }

        return select_by_config(config, hlid_mapping)

    def landing_gear(self, data: KeyedData) -> ArrayScalarLike:
        """Determine landing gear state from flight state.

        Uses the LG mapping from the BADA4 XML AeroConfSchedule to determine
        gear position for each aerodynamic configuration.

        Args:
            data: Flight state dictionary (same as aerodynamic_configuration).

        Returns:
            Boolean indicating gear extended (True = DOWN) or retracted (False = UP).

        Raises:
            ValueError: If LG mapping is not available in parsed XML data.
        """
        if not self._arpm.lg_map:
            raise ValueError(
                "LG mapping not available. The BADA4 XML file does not contain "
                "AeroConfSchedule with LG values, or the file was not parsed correctly."
            )

        config = self.aerodynamic_configuration(data)

        # Convert LG values to boolean: True if DOWN (gear down), False if UP (gear up)
        def lg_to_bool(lg_val):
            return lg_val == LandingGearPosition.DOWN if lg_val is not None else None

        lg_mapping = {
            AerodynamicConfig.CR: lg_to_bool(self._arpm.lg_map.get(AerodynamicConfig.CR)),
            AerodynamicConfig.IC: lg_to_bool(self._arpm.lg_map.get(AerodynamicConfig.IC)),
            AerodynamicConfig.TO: lg_to_bool(self._arpm.lg_map.get(AerodynamicConfig.TO)),
            AerodynamicConfig.AP: lg_to_bool(self._arpm.lg_map.get(AerodynamicConfig.AP)),
            AerodynamicConfig.LD: lg_to_bool(self._arpm.lg_map.get(AerodynamicConfig.LD)),
        }

        return select_by_config(config, lg_mapping)
