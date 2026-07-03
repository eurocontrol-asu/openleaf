# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""Poll-Schumann (PS) data loading from pycontrails.

This module handles loading aircraft parameters from the pycontrails
database for the Poll-Schumann performance model.

Reference:
    Poll, D.I.A., and Schumann, U. (2021). "An estimation method for the fuel burn
    and other performance characteristics of civil transport aircraft in the cruise."
    The Aeronautical Journal, 125(1284), 550-577.
"""

from __future__ import annotations

from openleaf.performance.core.perfos import EngineType
from openleaf.performance.models.ps.schemas import PSAerodynamicsData
from openleaf.performance.models.ps.schemas import PSAircraftData
from openleaf.performance.models.ps.schemas import PSData
from openleaf.performance.models.ps.schemas import PSEngineData


__all__ = [
    "load_from_pycontrails",
]


def load_from_pycontrails(aircraft_type: str) -> PSData:
    """Load PS data from pycontrails.

    All aircraft parameters come directly from the pycontrails database.
    No defaults are invented - if a parameter is missing, an error is raised.

    Args:
        aircraft_type: ICAO aircraft type code.

    Returns:
        PSData object with aircraft parameters.

    Raises:
        ImportError: If pycontrails is not installed.
        KeyError: If aircraft not found.
        AttributeError: If required parameter is missing from pycontrails.
    """
    try:
        from pycontrails.models.ps_model.ps_aircraft_params import load_aircraft_engine_params
        from pycontrails.models.ps_model.ps_model import get_aircraft_synonym_dict_ps
    except ImportError as e:
        raise ImportError(
            "pycontrails is required for PSModel. Install with: pip install pycontrails"
        ) from e

    aircraft_engine_params = load_aircraft_engine_params()
    synonym_dict = get_aircraft_synonym_dict_ps()

    atyp_ps = synonym_dict.get(aircraft_type) or aircraft_type

    if atyp_ps not in aircraft_engine_params:
        available = sorted(aircraft_engine_params.keys())[:20]
        raise KeyError(
            f"Aircraft '{aircraft_type}' not found in pycontrails database.\n"
            f"Available: {', '.join(available)}, ..."
        )

    params = aircraft_engine_params[atyp_ps]

    # Build PS data structures
    # Note: PS model is cruise-focused; VMO is not provided by pycontrails.
    # At cruise altitudes, MMO is always more restrictive than VMO, so we
    # set VMO to infinity and rely on Mach limits. WTC is also not provided.
    aircraft = PSAircraftData(
        model=aircraft_type,
        engine_type=EngineType.JET,
        wtc="M",  # Wake turbulence category - not in pycontrails
        oem=params.amass_oew,
        mtom=params.amass_mtow,
        mpl=params.amass_mpl,
        vmo=float("inf"),  # Not in pycontrails - PS model uses MMO at cruise
        mmo=params.max_mach_num,
        hmo=params.fl_max * 100.0,  # Convert FL to feet
    )

    # Compute wing aspect ratio from span and area
    wing_aspect_ratio = params.wing_span**2 / params.wing_surface_area

    aero = PSAerodynamicsData(
        surface_area=params.wing_surface_area,
        wing_span=params.wing_span,
        wing_aspect_ratio=wing_aspect_ratio,
        cos_sweep=params.cos_sweep,
        psi_0=params.psi_0,
        wing_constant=params.wing_constant,
        delta_2=params.delta_2,
        winglets=params.winglets,
        j_1=params.j_1,
        j_2=params.j_2,
        j_3=params.j_3,
        x_ref=params.x_ref,
    )

    engine = PSEngineData(
        n_engines=params.n_engine,
        m_des=params.m_des,
        c_t_des=params.c_t_des,
        eta_1=params.eta_1,
        eta_2=params.eta_2,
        tet_mcc=params.tet_mcc,
        tr_ec=params.tr_ec,
        m_ec=params.m_ec,
        ff_idle_sls=params.ff_idle_sls,
        ff_max_sls=params.ff_max_sls,
        p_i_max=params.p_i_max,
        p_inf_co=params.p_inf_co,
    )

    return PSData(aircraft=aircraft, engine=engine, aero=aero)
