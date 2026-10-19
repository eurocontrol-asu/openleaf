# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Poll-Schumann (PS) data structures.

Provides typed data classes for PS aircraft parameters following
the openleaf schema conventions.

Reference:
    Poll, D.I.A., and Schumann, U. (2021). "An estimation method for the fuel burn
    and other performance characteristics of civil transport aircraft in the cruise."
    The Aeronautical Journal, 125(1284), 257-295.
    https://doi.org/10.1017/aer.2020.62
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict

from openleaf.performance.core.perfos import AerodynamicsData
from openleaf.performance.core.perfos import AircraftData
from openleaf.performance.core.perfos import EngineData
from openleaf.performance.core.perfos import LimitsData


__all__ = [
    "PSAerodynamicsData",
    "PSAircraftData",
    "PSData",
    "PSEngineData",
    "PSLimitsData",
]


class PSEngineData(EngineData):
    """Poll-Schumann engine data.

    All parameters are derived from pycontrails database.
    No defaults are provided - all fields are required.

    Attributes:
        n_engines: Number of engines.
        m_des: Design Mach number.
        c_t_des: Thrust coefficient at design point.
        eta_1: Propulsion efficiency coefficient 1.
        eta_2: Propulsion efficiency coefficient 2.
        tet_mcc: Maximum combustor temperature [K].
        tr_ec: Engine temperature ratio at max efficiency.
        m_ec: Mach number at engine corner point.
        ff_idle_sls: Idle fuel flow at sea level static [kg/s].
        ff_max_sls: Maximum fuel flow at sea level static [kg/s].
        p_i_max: Maximum impact pressure [Pa].
        p_inf_co: Crossover pressure [Pa].
        thrust_buffer: Safety buffer for max thrust coefficient calculation.
            Default 0.0 gives thrust consistent with BADA MCC rating.
            pycontrails uses 0.20 (20% increase) as an upper safety bound.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_engines: int
    m_des: float
    c_t_des: float
    eta_1: float
    eta_2: float
    tet_mcc: float
    tr_ec: float
    m_ec: float
    ff_idle_sls: float
    ff_max_sls: float
    p_i_max: float
    p_inf_co: float
    thrust_buffer: float = 0.0


class PSAerodynamicsData(AerodynamicsData):
    """Poll-Schumann aerodynamic data.

    All parameters are derived from pycontrails database.
    No defaults are provided - all fields are required.

    Attributes:
        surface_area: Wing reference area [m**2].
        wing_span: Wing span [m].
        wing_aspect_ratio: Wing aspect ratio.
        cos_sweep: Cosine of wing sweep angle.
        psi_0: Parasite drag area factor.
        wing_constant: Wing constant for wave drag.
        delta_2: Induced drag factor increment.
        winglets: Whether aircraft has winglets.
        j_1: Wave drag coefficient 1.
        j_2: Wave drag coefficient 2.
        j_3: Wave drag coefficient 3.
        x_ref: Reference x for wave drag.
        c_l_do: Design optimum lift coefficient.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    wing_span: float
    wing_aspect_ratio: float
    cos_sweep: float
    psi_0: float
    wing_constant: float
    delta_2: float
    winglets: bool
    j_1: float
    j_2: float
    j_3: float
    x_ref: float
    c_l_do: float


class PSAircraftData(AircraftData):
    """Poll-Schumann aircraft data.

    Extends base aircraft data with PS-specific parameters.
    All parameters are derived from pycontrails database.

    Attributes:
        mtom: Maximum takeoff mass [kg].
        oem: Operating empty mass [kg].
        mpl: Maximum payload [kg].
        vmo: Maximum operating calibrated airspeed [kt].
        mmo: Maximum operating Mach number.
        hmo: Maximum operating altitude [ft].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mtom: float
    oem: float
    mpl: float
    vmo: float
    mmo: float
    hmo: float


class PSLimitsData(LimitsData):
    """Poll-Schumann aircraft limits data.

    Extends base limits with PS-specific fields for impact pressure constraints.

    Attributes:
        vmo: Maximum operating speed [kt].
        mmo: Maximum operating Mach number.
        hmo: Maximum operating altitude [ft].
        p_i_max: Maximum impact pressure [Pa].
        p_inf_co: Crossover pressure [Pa].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_i_max: float
    p_inf_co: float


class PSData(BaseModel):
    """Complete Poll-Schumann model data container.

    Attributes:
        aircraft: Aircraft specification data.
        engine: Engine performance data.
        aero: Aerodynamic data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    aircraft: PSAircraftData
    engine: PSEngineData
    aero: PSAerodynamicsData
