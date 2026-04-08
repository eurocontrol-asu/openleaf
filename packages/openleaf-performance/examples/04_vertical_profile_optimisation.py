#!/usr/bin/env python3
"""
Vertical Profile Optimisation with Fixed Horizontal Route.

This example demonstrates how to optimise the altitude-speed profile (vertical
profile) of a flight given a fixed horizontal route, using CasADi optimization.

The horizontal route (latitude, longitude) is extracted from observed flight data,
and the along-path distance is computed using the haversine formula. This distance
becomes the independent variable instead of time, so the route is fixed and we
only optimise the vertical profile.

Method:
    1. Load flight data with lat/lon coordinates
    2. Compute along-path (cumulative) distance using haversine formula
    3. Build symbolic wind(h) and temperature(h) interpolants from met.npz
    4. Formulate optimal control problem with distance as independent variable
    5. Minimise total fuel consumption

State Variables: (h, v, m)
    - h: Altitude [m]
    - v: True airspeed [m/s]
    - m: Aircraft mass [kg]

Control Variables: (T, f) at each waypoint
    - T: Thrust [N], bounded by T_min and T_max
    - f: Energy share factor [-], bounded by [0, large_value]

Boundary Conditions:
    - At s=0: h, v, m are FIXED to observed values
    - At s=s_final: h, v are FIXED to observed values (m is free)

Dynamics (with respect to distance s):
    The TEM dynamics dX/dt are converted to dX/ds using ds/dt = gs (ground speed):
        gs = v + wind  (positive wind = tailwind)
        dh/ds = f * Ps / (g * gs)
        dv/ds = (1 - f) * Ps / (v * gs)
        dm/ds = -FF / gs
    where Ps = (T - D) * v / m is the specific excess power [W/kg]

Integration Scheme:
    Forward Euler method:
        x[k+1] = x[k] + ds * f(x[k], u[k])

Speed Constraints:
    - CAS <= VMO (max operating CAS)
    - Mach <= MMO (max operating Mach)

Objective:
    Minimise total fuel consumption: m[0] - m[N]

Requirements:
    - casadi: For symbolic computation and NLP solving
    - aetherx: For aircraft performance models and ISA

Usage:
    python examples/04_vertical_profile_optimisation.py [--model ps|bada3|bada4]

References:
    - BADA User Manual, Section 3.1: Total Energy Model
    - Haversine formula for great-circle distance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Literal

import casadi as ca
import cyclopts
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

import openleaf.performance.numpy as anp
import numpy as np
from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.physics import isa
from openleaf.performance.physics.units import FT
from openleaf.performance.physics.units import MS_TO_KT
from openleaf.performance.physics.units import G


if TYPE_CHECKING:
    from openleaf.performance.core.perfos import AircraftModel

console = Console()


# =============================================================================
# Haversine Distance Calculation
# =============================================================================

# Earth radius in meters
EARTH_RADIUS_M = 6371000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points using haversine formula.

    Args:
        lat1: Latitude of first point [degrees].
        lon1: Longitude of first point [degrees].
        lat2: Latitude of second point [degrees].
        lon2: Longitude of second point [degrees].

    Returns:
        Distance between points [m].
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def compute_along_path_distance(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    """Compute cumulative along-path distance from lat/lon coordinates.

    Args:
        latitudes: Array of latitudes [degrees].
        longitudes: Array of longitudes [degrees].

    Returns:
        Cumulative distance array [m], starting from 0.
    """
    n = len(latitudes)
    distances = np.zeros(n)

    for i in range(1, n):
        segment = haversine_distance(
            latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i]
        )
        distances[i] = distances[i - 1] + segment

    return distances


# =============================================================================
# Data Loading
# =============================================================================


class FlightData:
    """Container for flight trajectory data with meteorological profiles."""

    def __init__(
        self,
        along_path_distance: np.ndarray,
        altitude: np.ndarray,
        true_airspeed: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        wind_data: np.ndarray,
        temp_data: np.ndarray,
        pressure_levels: np.ndarray,
        aircraft_type: str,
    ) -> None:
        self.along_path_distance = along_path_distance  # [m]
        self.altitude = altitude  # [m]
        self.true_airspeed = true_airspeed  # [m/s]
        self.latitude = latitude  # [deg]
        self.longitude = longitude  # [deg]
        self.wind_data = wind_data  # shape: (N, N_levels), longitudinal wind [m/s]
        self.temp_data = temp_data  # shape: (N, N_levels), temperature [K]
        self.pressure_levels = pressure_levels  # [hPa]
        self.aircraft_type = aircraft_type

    def __len__(self) -> int:
        return len(self.along_path_distance)


def load_flight_data(
    data_path: Path,
    met_path: Path,
    initial_mass: float,
) -> tuple[FlightData, float]:
    """Load flight trajectory and meteorological data.

    Args:
        data_path: Path to flight_data.json.
        met_path: Path to met.npz with wind/temperature profiles.
        initial_mass: Initial aircraft mass [kg].

    Returns:
        Tuple of (FlightData, initial_mass).
    """
    # Load flight data
    with data_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Extract arrays
    latitude = np.array(raw["latitude"])
    longitude = np.array(raw["longitude"])
    altitude_ft = np.array(raw["altitude_ft"])
    true_airspeed_ms = np.array(raw["true_airspeed"])
    aircraft_type = str(raw.get("aircraft_type", "B738"))

    # Load meteorological data
    met = np.load(met_path)
    wind_data = met["longitudinal_wind_data"]  # shape: (N_met, N_levels)
    temp_data = met["air_temperature_data"]  # shape: (N_met, N_levels)
    pressure_levels = met["pressure_levels"]  # [hPa]

    # Align met data with flight data length before filtering
    n_flight_raw = len(latitude)
    n_met = len(wind_data)
    if n_met >= n_flight_raw:
        wind_data = wind_data[:n_flight_raw]
        temp_data = temp_data[:n_flight_raw]
    else:
        # Extend met data to match flight data
        wind_extend = np.tile(wind_data[-1], (n_flight_raw - n_met, 1))
        temp_extend = np.tile(temp_data[-1], (n_flight_raw - n_met, 1))
        wind_data = np.vstack([wind_data, wind_extend])
        temp_data = np.vstack([temp_data, temp_extend])

    # Filter out points with NaN in true_airspeed or met data
    valid_tas = ~np.isnan(true_airspeed_ms)
    valid_wind = ~np.isnan(wind_data).any(axis=1)
    valid_temp = ~np.isnan(temp_data).any(axis=1)
    valid_mask = valid_tas & valid_wind & valid_temp

    if not valid_mask.all():
        n_removed = (~valid_mask).sum()
        console.print(
            f"[yellow]Removing {n_removed} points with NaN data "
            f"(TAS: {(~valid_tas).sum()}, wind: {(~valid_wind).sum()}, "
            f"temp: {(~valid_temp).sum()})[/yellow]"
        )
        latitude = latitude[valid_mask]
        longitude = longitude[valid_mask]
        altitude_ft = altitude_ft[valid_mask]
        true_airspeed_ms = true_airspeed_ms[valid_mask]
        wind_data = wind_data[valid_mask]
        temp_data = temp_data[valid_mask]

    # Convert altitude to meters
    altitude_m = altitude_ft * FT

    # Compute along-path distance (after filtering)
    along_path_distance = compute_along_path_distance(latitude, longitude)

    flight_data = FlightData(
        along_path_distance=along_path_distance,
        altitude=altitude_m,
        true_airspeed=true_airspeed_ms,
        latitude=latitude,
        longitude=longitude,
        wind_data=wind_data,
        temp_data=temp_data,
        pressure_levels=pressure_levels,
        aircraft_type=aircraft_type,
    )

    return flight_data, initial_mass


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_type: str, aircraft_type: str) -> AircraftModel:
    """Load aircraft performance model.

    Args:
        model_type: One of 'ps', 'bada3', 'bada4'.
        aircraft_type: ICAO aircraft type code.

    Returns:
        Initialized aircraft model.
    """
    model_type = model_type.lower()

    if model_type == "ps":
        from openleaf.performance.models.ps import PSModel

        return PSModel(aircraft_type)

    if model_type == "bada3":
        from openleaf.performance.models.bada3 import BADA3Model

        return BADA3Model(aircraft_type)

    if model_type == "bada4":
        from openleaf.performance.models.bada4 import BADA4Model

        return BADA4Model(aircraft_type)

    msg = f"Unknown model type: {model_type}"
    raise ValueError(msg)


# =============================================================================
# Symbolic Interpolants
# =============================================================================


def build_met_interpolants(
    pressure_levels: np.ndarray,
    wind_row: np.ndarray,
    temp_row: np.ndarray,
    point_idx: int,
) -> tuple[ca.Function, ca.Function, float, float]:
    """Build CasADi interpolants for wind and temperature as functions of altitude.

    The met data is given as a function of pressure level. We create interpolants
    that map pressure [hPa] -> wind [m/s] and pressure [hPa] -> temperature [K].

    CasADi interpolant requires strictly increasing grid points, so we sort
    the pressure levels if needed.

    To get wind(h) and temp(h), the caller should:
        p_hPa = isa.air_pressure(h) / 100
        p_clamped = clamp(p_hPa, p_min, p_max)  # Avoid extrapolation!
        wind = wind_interp(p_clamped)
        temp = temp_interp(p_clamped)

    Args:
        pressure_levels: Pressure levels [hPa], shape (N_levels,).
        wind_row: Longitudinal wind at each pressure level [m/s].
        temp_row: Temperature at each pressure level [K].
        point_idx: Index for naming the interpolants.

    Returns:
        Tuple of (wind_interpolant, temp_interpolant, p_min, p_max).
    """
    # Sort by pressure (CasADi requires strictly increasing grid)
    sort_idx = np.argsort(pressure_levels)
    p_sorted = pressure_levels[sort_idx]
    wind_sorted = wind_row[sort_idx]
    temp_sorted = temp_row[sort_idx]

    p_list = list(p_sorted)
    wind_list = list(wind_sorted)
    temp_list = list(temp_sorted)

    wind_interp = ca.interpolant(f"wind_{point_idx}", "linear", [p_list], wind_list)
    temp_interp = ca.interpolant(f"temp_{point_idx}", "linear", [p_list], temp_list)

    p_min = float(p_sorted[0])
    p_max = float(p_sorted[-1])

    return wind_interp, temp_interp, p_min, p_max


# =============================================================================
# Dynamics Builder
# =============================================================================


def build_dynamics_expressions(
    model: AircraftModel,
    temp_interpolant: ca.Function,
    wind_interpolant: ca.Function,
    h_sym: ca.MX,
    v_sym: ca.MX,
    m_sym: ca.MX,
    T_sym: ca.MX,
    f_sym: ca.MX,
) -> dict[str, ca.MX]:
    """Build symbolic expressions for dynamics with respect to distance.

    The TEM dynamics are converted from time-based to distance-based using
    ground speed: ds/dt = gs = v + wind (positive wind = tailwind).

        dh/ds = f * Ps / (g * gs)
        dv/ds = (1 - f) * Ps / (v * gs)
        dm/ds = -FF / gs

    where Ps = (T - D) * v / m is the specific excess power.

    Args:
        model: Aircraft performance model.
        temp_interpolant: CasADi interpolant: pressure_hPa -> temperature [K].
        wind_interpolant: CasADi interpolant: pressure_hPa -> wind [m/s].
        h_sym: Symbolic altitude [m].
        v_sym: Symbolic TAS [m/s].
        m_sym: Symbolic mass [kg].
        T_sym: Symbolic thrust [N].
        f_sym: Symbolic energy share factor [-].

    Returns:
        Dictionary with dynamics expressions and constraint values:
        - dh_ds, dv_ds, dm_ds: state derivatives
        - T_min, T_max: thrust bounds
        - p_hpa: pressure in hPa (for met interpolation constraints)
        - mach: Mach number (for MMO constraint)
        - cas: calibrated airspeed [m/s] (for VMO constraint)
    """
    # Atmospheric state
    p = isa.air_pressure(h_sym)  # [Pa]
    p_hpa = p / 100.0  # [hPa] for interpolant

    # Temperature from met data (no clamping - use constraints instead)
    T_air = temp_interpolant(p_hpa)

    # ISA temperature and delta_ISA
    T_isa = isa.air_temperature(h_sym)
    delta_isa = T_air - T_isa

    # Air density from ideal gas law
    rho = p / (isa.R * T_air)

    # Speed of sound and Mach number
    a = anp.sqrt(isa.GAMMA * isa.R * T_air)
    mach = v_sym / a

    # Calibrated airspeed (CAS) from TAS using impact pressure method
    # CAS = a0 * sqrt(5 * ((q_c/p0 + 1)^((gamma-1)/gamma) - 1))
    # where q_c = p * ((1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1)) - 1) is impact pressure
    p0 = isa.P_SL  # Sea level pressure [Pa]
    a0 = isa.A_SL  # Sea level speed of sound [m/s]
    gamma = isa.GAMMA

    # Impact pressure
    q_c = p * (
        anp.power(1.0 + (gamma - 1.0) / 2.0 * mach**2, gamma / (gamma - 1.0)) - 1.0
    )

    # CAS from impact pressure
    cas = a0 * anp.sqrt(5.0 * (anp.power(q_c / p0 + 1.0, (gamma - 1.0) / gamma) - 1.0))

    # Dynamic pressure
    q = 0.5 * rho * v_sym**2

    # Wing reference area
    S = model.aerodynamics.data.surface_area

    # Lift coefficient (steady flight)
    data_for_cl = {
        "aircraft_mass": m_sym,
        "true_airspeed": v_sym,
        "air_density": rho,
        "load_factor": 1.0,
    }
    cl = model.lift_coef(data_for_cl)

    # Data dict for model calls
    data = {
        "aircraft_mass": m_sym,
        "altitude": h_sym,
        "true_airspeed": v_sym,
        "air_pressure": p,
        "air_temperature": T_air,
        "air_density": rho,
        "aerodynamic_configuration": AerodynamicConfig.CR,
        "mach_number": mach,
        "lift_coef": cl,
        "delta_isa": delta_isa,
        "engine_setting": EngineSetting.MCRZ,
    }

    # Drag coefficient and force
    cd = model.aerodynamics.drag_coef(data)
    drag = cd * q * S

    # Thrust limits
    T_min = model.thrust_min(data)
    T_max = model.thrust_max(data)

    # Fuel flow at given thrust
    data_with_thrust = {**data, "thrust": T_sym}
    fuel_flow = model.engine.fuel_flow(data_with_thrust)

    # Specific excess power [W/kg]
    Ps = (T_sym - drag) * v_sym / m_sym

    # Get wind from met data (positive = tailwind)
    wind = wind_interpolant(p_hpa)

    # Ground speed: gs = TAS + wind (tailwind increases ground speed)
    gs = v_sym + wind

    # TEM dynamics with respect to distance (ds = gs * dt)
    # dh/ds = (dh/dt) / gs = (f * Ps / g) / gs
    dh_ds = f_sym * Ps / (G * gs)

    # dv/ds = (dv/dt) / gs = ((1 - f) * Ps / v) / gs
    dv_ds = (1.0 - f_sym) * Ps / (v_sym * gs)

    # dm/ds = (dm/dt) / gs = -FF / gs
    dm_ds = -fuel_flow / gs

    return {
        "dh_ds": dh_ds,
        "dv_ds": dv_ds,
        "dm_ds": dm_ds,
        "T_min": T_min,
        "T_max": T_max,
        "p_hpa": p_hpa,
        "mach": mach,
        "cas": cas,
    }


# =============================================================================
# Vertical Profile Optimisation
# =============================================================================


def optimise_vertical_profile(
    data: FlightData,
    model: AircraftModel,
    initial_mass: float,
    verbose: bool = True,
) -> dict:
    """Optimise vertical profile to minimise fuel consumption.

    Formulation:
        Independent variable: along-path distance s
        State variables: (h, v, m)
        Control variables: (T, f) at each waypoint

        Boundary conditions:
            - s=0: h[0], v[0], m[0] FIXED
            - s=s_final: h[N], v[N] FIXED, m[N] FREE

        Constraints:
            - T_min[k] <= T[k] <= T_max[k]
            - 0 <= f[k] <= large_value (e.g., 10)

        Objective:
            min (m[0] - m[N]) = min fuel consumption

    Args:
        data: FlightData container.
        model: Aircraft performance model.
        initial_mass: Initial aircraft mass [kg].
        verbose: Print optimization progress.

    Returns:
        Dictionary with optimisation results.
    """
    N = len(data)
    s = data.along_path_distance
    ds = np.diff(s)

    # Reference values from observations
    h_obs = data.altitude
    v_obs = data.true_airspeed

    # Model mass bounds
    oem = model.oem
    mtow = model.mtom

    if verbose:
        console.print(f"[dim]Waypoints: N={N}[/dim]")
        console.print(f"[dim]Total distance: {s[-1] / 1000:.1f} km[/dim]")
        console.print(f"[dim]Mass bounds: OEM={oem:.0f} kg, MTOW={mtow:.0f} kg[/dim]")

    # Build met interpolants for each waypoint
    pressure_levels = data.pressure_levels
    wind_interpolants = []
    temp_interpolants = []
    p_min = None
    p_max = None
    for k in range(N):
        wind_interp, temp_interp, p_min_k, p_max_k = build_met_interpolants(
            pressure_levels,
            data.wind_data[k],
            data.temp_data[k],
            k,
        )
        wind_interpolants.append(wind_interp)
        temp_interpolants.append(temp_interp)
        # All waypoints have same pressure range
        if p_min is None:
            p_min = p_min_k
            p_max = p_max_k

    if verbose:
        console.print(f"[dim]Pressure range: {p_min:.0f} - {p_max:.0f} hPa[/dim]")

    # --- Speed limits (VMO/MMO) ---
    # Get from model - VMO may be inf for cruise-only models (PS)
    vmo = getattr(model, "vmo", float("inf"))
    mmo = getattr(model, "mmo", None)

    if mmo is None:
        msg = f"Model {type(model).__name__} does not provide MMO"
        raise ValueError(msg)

    if verbose:
        vmo_str = (
            f"{vmo * MS_TO_KT:.0f} kt" if np.isfinite(vmo) else "N/A (cruise model)"
        )
        console.print(f"[dim]VMO: {vmo_str}, MMO: {mmo:.2f}[/dim]")

    # --- CasADi Opti formulation ---
    opti = ca.Opti()

    # State variables at each waypoint
    h = opti.variable(N)  # Altitude [m]
    v = opti.variable(N)  # TAS [m/s]
    m = opti.variable(N)  # Mass [kg]

    # Control variables at each segment (N-1 controls for N waypoints)
    T = opti.variable(N - 1)  # Thrust [N]
    f = opti.variable(N - 1)  # Energy share factor [-]

    # --- Boundary conditions ---
    # At s=0: h, v, m are FIXED
    opti.subject_to(h[0] == h_obs[0])
    opti.subject_to(v[0] == v_obs[0])
    opti.subject_to(m[0] == initial_mass)

    # At s=s_final: h, v are FIXED (m is free)
    opti.subject_to(h[N - 1] == h_obs[N - 1])
    opti.subject_to(v[N - 1] == v_obs[N - 1])

    # --- State bounds ---
    for k in range(N):
        opti.subject_to(m[k] >= oem)
        opti.subject_to(m[k] <= mtow)
        # Altitude bounds (reasonable operational envelope)
        opti.subject_to(h[k] >= 1000 * FT)  # min 1000 ft
        opti.subject_to(h[k] <= 45000 * FT)  # max 45000 ft
        # TAS bounds (loose bounds, VMO/MMO handled via CAS/Mach constraints)
        opti.subject_to(v[k] >= 100.0)  # min ~200 kt
        opti.subject_to(v[k] <= 350.0)  # max ~680 kt (relaxed for high altitude)

    # --- Dynamics and control constraints (Forward Euler) ---
    for k in range(N - 1):
        # Get dynamics at waypoint k
        dyn = build_dynamics_expressions(
            model,
            temp_interpolants[k],
            wind_interpolants[k],
            h[k],
            v[k],
            m[k],
            T[k],
            f[k],
        )

        # Forward Euler integration: x[k+1] = x[k] + ds * f(x[k], u[k])
        opti.subject_to(h[k + 1] == h[k] + dyn["dh_ds"] * ds[k])
        opti.subject_to(v[k + 1] == v[k] + dyn["dv_ds"] * ds[k])
        opti.subject_to(m[k + 1] == m[k] + dyn["dm_ds"] * ds[k])

        # Thrust constraints
        opti.subject_to(T[k] >= dyn["T_min"])
        opti.subject_to(T[k] <= dyn["T_max"])

        # ESF constraints: 0 <= f <= large_value
        # f can be > 1 when decelerating while climbing (kinetic -> potential)
        opti.subject_to(f[k] >= 0.0)
        opti.subject_to(f[k] <= 2.0)  # Large upper bound

        # Pressure constraints (keep within met data interpolation range)
        opti.subject_to(dyn["p_hpa"] >= p_min)
        opti.subject_to(dyn["p_hpa"] <= p_max)

        # Speed envelope constraints: CAS <= VMO (if finite), Mach <= MMO
        if np.isfinite(vmo):
            opti.subject_to(dyn["cas"] <= vmo)
        opti.subject_to(dyn["mach"] <= mmo)

    # --- Objective: minimise total fuel consumption ---
    # Fuel = m[0] - m[N-1]
    opti.minimize(m[0] - m[N - 1])

    # --- Initial guesses ---
    opti.set_initial(h, h_obs)
    opti.set_initial(v, v_obs)

    # Linear mass decrease guess
    m_guess = np.linspace(initial_mass, initial_mass * 0.95, N)
    opti.set_initial(m, m_guess)

    # Thrust guess: roughly drag at cruise
    T_guess = 50000.0  # Typical cruise thrust for narrowbody
    opti.set_initial(T, T_guess * np.ones(N - 1))

    # ESF guess: mostly climb/descent (f close to 1)
    opti.set_initial(f, 0.5 * np.ones(N - 1))

    # --- Solve ---
    opts = {
        "ipopt.print_level": 5 if verbose else 0,
        "ipopt.max_iter": 2000,
        "ipopt.tol": 1e-3,
        "ipopt.acceptable_tol": 1e-3,
        "print_time": verbose,
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
    except RuntimeError:
        console.print(
            "[yellow]Solver did not converge within iteration limit; "
            "using best iterate found.[/yellow]"
        )
        sol = opti.debug

    # Extract solution
    h_opt = np.array(sol.value(h)).flatten()
    v_opt = np.array(sol.value(v)).flatten()
    m_opt = np.array(sol.value(m)).flatten()
    T_opt = np.array(sol.value(T)).flatten()
    f_opt = np.array(sol.value(f)).flatten()

    # Compute fuel consumption
    fuel_consumed = m_opt[0] - m_opt[-1]

    return {
        "h_opt": h_opt,
        "v_opt": v_opt,
        "m_opt": m_opt,
        "T_opt": T_opt,
        "f_opt": f_opt,
        "fuel_consumed": fuel_consumed,
        "distance": s,
        "h_obs": h_obs,
        "v_obs": v_obs,
        "solver_stats": sol.stats(),
    }


# =============================================================================
# Display Results
# =============================================================================


def display_results(result: dict, model_name: str) -> None:
    """Display optimisation results."""
    table = Table(title=f"Vertical Profile Optimisation - {model_name.upper()}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Unit", style="dim")

    table.add_row("Total Distance", f"{result['distance'][-1] / 1000:.1f}", "km")
    table.add_row("Initial Mass", f"{result['m_opt'][0]:,.0f}", "kg")
    table.add_row("Final Mass", f"{result['m_opt'][-1]:,.0f}", "kg")
    table.add_row("Fuel Consumed", f"{result['fuel_consumed']:,.0f}", "kg")

    # Altitude range
    h_min_ft = result["h_opt"].min() / FT
    h_max_ft = result["h_opt"].max() / FT
    table.add_row("Altitude Range", f"{h_min_ft:.0f} - {h_max_ft:.0f}", "ft")

    # TAS range
    v_min_kt = result["v_opt"].min() * MS_TO_KT
    v_max_kt = result["v_opt"].max() * MS_TO_KT
    table.add_row("TAS Range", f"{v_min_kt:.0f} - {v_max_kt:.0f}", "kt")

    # Thrust range
    T_min_kn = result["T_opt"].min() / 1000
    T_max_kn = result["T_opt"].max() / 1000
    table.add_row("Thrust Range", f"{T_min_kn:.1f} - {T_max_kn:.1f}", "kN")

    # ESF range
    f_min = result["f_opt"].min()
    f_max = result["f_opt"].max()
    table.add_row("ESF Range", f"{f_min:.3f} - {f_max:.3f}", "-")

    console.print(table)


def plot_results(
    result: dict,
    save_path: Path | None = None,
) -> None:
    """Plot vertical profile optimisation results.

    Args:
        result: Dictionary with optimisation results.
        save_path: Path to save the plot (PNG). If None, displays interactively.
    """
    # Convert distance to km for plotting
    s_km = result["distance"] / 1000.0

    _, axes = plt.subplots(3, 2, figsize=(14, 12))

    # --- Altitude profile ---
    ax = axes[0, 0]
    ax.plot(s_km, result["h_obs"] / FT, "o", label="Observed", markersize=3, alpha=0.5)
    ax.plot(s_km, result["h_opt"] / FT, "-", label="Optimised", linewidth=2)
    ax.set_xlabel("Along-path distance [km]")
    ax.set_ylabel("Altitude [ft]")
    ax.set_title("Altitude Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- TAS profile ---
    ax = axes[0, 1]
    ax.plot(
        s_km, result["v_obs"] * MS_TO_KT, "o", label="Observed", markersize=3, alpha=0.5
    )
    ax.plot(s_km, result["v_opt"] * MS_TO_KT, "-", label="Optimised", linewidth=2)
    ax.set_xlabel("Along-path distance [km]")
    ax.set_ylabel("True Airspeed [kt]")
    ax.set_title("Speed Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Mass profile ---
    ax = axes[1, 0]
    ax.plot(s_km, result["m_opt"] / 1000, "-", color="purple", linewidth=2)
    ax.set_xlabel("Along-path distance [km]")
    ax.set_ylabel("Mass [tonnes]")
    ax.set_title(f"Mass Profile (fuel burned: {result['fuel_consumed']:.0f} kg)")
    ax.grid(True, alpha=0.3)

    # --- Thrust profile ---
    ax = axes[1, 1]
    # Thrust is defined at segment midpoints (N-1 values)
    s_mid = (s_km[:-1] + s_km[1:]) / 2
    ax.plot(s_mid, result["T_opt"] / 1000, "-", color="red", linewidth=2)
    ax.set_xlabel("Along-path distance [km]")
    ax.set_ylabel("Thrust [kN]")
    ax.set_title("Thrust Profile")
    ax.grid(True, alpha=0.3)

    # --- Energy Share Factor profile ---
    ax = axes[2, 0]
    ax.plot(s_mid, result["f_opt"], "-", color="green", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="f=1 (all to climb)")
    ax.set_xlabel("Along-path distance [km]")
    ax.set_ylabel("Energy Share Factor [-]")
    ax.set_title("ESF Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Altitude vs TAS (hodograph-style) ---
    ax = axes[2, 1]
    scatter = ax.scatter(
        result["v_opt"] * MS_TO_KT,
        result["h_opt"] / FT,
        c=s_km,
        cmap="viridis",
        s=20,
    )
    ax.plot(
        result["v_obs"] * MS_TO_KT,
        result["h_obs"] / FT,
        "x",
        color="red",
        markersize=4,
        alpha=0.3,
        label="Observed",
    )
    ax.set_xlabel("True Airspeed [kt]")
    ax.set_ylabel("Altitude [ft]")
    ax.set_title("Altitude vs Speed (color = distance)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Distance [km]")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        console.print(f"[green]Plot saved to {save_path}[/green]")
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================


_DEFAULT_DATA = Path(__file__).parent.parent / "data" / "flight_data.json"
_DEFAULT_MET = Path(__file__).parent.parent / "data" / "met.npz"

app = cyclopts.App(
    name="vertical-profile-optimisation",
    help="Optimise vertical profile for minimum fuel consumption.",
)


@app.default
def main(
    model: Annotated[
        Literal["ps", "bada3", "bada4"],
        cyclopts.Parameter(help="Performance model to use"),
    ] = "ps",
    data: Annotated[
        Path,
        cyclopts.Parameter(help="Path to flight data JSON file"),
    ] = _DEFAULT_DATA,
    met: Annotated[
        Path,
        cyclopts.Parameter(help="Path to meteorological data (met.npz)"),
    ] = _DEFAULT_MET,
    mass: Annotated[
        float,
        cyclopts.Parameter(help="Initial aircraft mass [kg]"),
    ] = 69000.0,
    quiet: Annotated[
        bool,
        cyclopts.Parameter(help="Suppress solver output"),
    ] = False,
    save_plot: Annotated[
        Path | None,
        cyclopts.Parameter(help="Path to save result plot (PNG)"),
    ] = None,
) -> None:
    """Run vertical profile optimisation example."""
    console.rule("[bold]Vertical Profile Optimisation[/bold]")
    console.print(f"Model: [cyan]{model.upper()}[/cyan]")
    console.print(f"Data: [dim]{data}[/dim]")
    console.print(f"Met: [dim]{met}[/dim]")
    console.print(f"Initial mass: [dim]{mass:.0f} kg[/dim]")

    # Load data
    try:
        flight_data, initial_mass = load_flight_data(data, met, mass)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        sys.exit(1)

    console.print(f"Aircraft: [cyan]{flight_data.aircraft_type}[/cyan]")
    console.print(
        f"Flight data: [green]{len(flight_data)} waypoints[/green], "
        f"distance {flight_data.along_path_distance[-1] / 1000:.1f} km"
    )
    console.print(
        f"Altitude range: {flight_data.altitude.min() / FT:.0f} - "
        f"{flight_data.altitude.max() / FT:.0f} ft"
    )

    # Load model
    try:
        perf_model = load_model(model, flight_data.aircraft_type)
    except ImportError as e:
        console.print(f"[red]Cannot load {model} model: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Model loading failed: {e}[/red]")
        sys.exit(1)

    console.print(f"Model loaded: MTOW={perf_model.mtom:.0f} kg, OEM={perf_model.oem:.0f} kg")

    # Run optimisation
    console.print("\n[bold]Running optimisation...[/bold]")
    result = optimise_vertical_profile(
        flight_data, perf_model, initial_mass, verbose=not quiet
    )

    # Display results
    console.print()
    display_results(result, model)

    console.print(
        f"\n[bold green]Optimal fuel: {result['fuel_consumed']:,.0f} kg[/bold green]"
    )

    # Plot results
    path = save_plot or Path(__file__).parent / "vertical_profile_optimisation.png"
    plot_results(result, save_path=path)


if __name__ == "__main__":
    app()
