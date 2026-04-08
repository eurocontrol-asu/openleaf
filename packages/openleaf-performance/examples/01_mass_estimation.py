#!/usr/bin/env python3
"""
Aircraft Mass Estimation from Climb Trajectory.

This example demonstrates how to estimate aircraft takeoff mass from observed
ADS-B climb data using CasADi optimization.

Method:
    - Use climb phase only (above FL100, positive ROCD)
    - Assume maximum climb thrust (T_max) throughout
    - Initial conditions FIXED: h[0] = h_obs[0], V[0] = V_obs[0]
    - Decision variables: initial mass m0, energy share factor f[t]
    - State propagation via TEM dynamics (h, V evolve from initial conditions)

The Total Energy Model (TEM) propagates the state:
    Ps = (T_max - D) * V / m     (specific excess power)
    dh/dt = f * Ps / g           (rate of climb)
    dV/dt = (1 - f) * Ps / V     (acceleration)
    dm/dt = -FF                   (fuel consumption)

Note: T_max and D depend on current state (h, V, m), creating coupled dynamics.

The energy share factor f[t] absorbs:
    - ADS-B measurement noise
    - Pilot control variations
    - Performance model errors

Uses actual meteorological data (air temperature) from met.npz for realistic
atmospheric conditions.

Model Compatibility:
    - BADA3/BADA4: Full support (recommended for climb estimation)
    - PS: Full support
    - OpenAP: Supported via implicit NLP constraint (vs ↔ thrust resolved by IPOPT)

Usage:
    python examples/01_mass_estimation.py [--model bada3|bada4|ps|openap]

References:
    - BADA User Manual, Section 3.1: Total Energy Model
    - Poles et al. (2010): "Aircraft Mass Estimation from Trajectory Data"
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

import numpy as np
from openleaf.performance.physics import isa
from openleaf.performance.physics.units import FPM
from openleaf.performance.physics.units import FT
from openleaf.performance.physics.units import MS_TO_KT
from openleaf.performance.physics.units import G


if TYPE_CHECKING:
    from openleaf.performance.core.perfos import AircraftModel

console = Console()


# =============================================================================
# Data Containers
# =============================================================================


class ClimbData:
    """Container for climb trajectory data with meteorological data."""

    def __init__(
        self,
        time: np.ndarray,
        altitude: np.ndarray,
        true_airspeed: np.ndarray,
        vertical_speed: np.ndarray,
        temp_data: np.ndarray | None,
        pressure_levels: np.ndarray | None,
        climb_indices: np.ndarray,
        aircraft_type: str,
        use_isa: bool = False,
    ) -> None:
        self.time = time
        self.altitude = altitude
        self.true_airspeed = true_airspeed
        self.vertical_speed = vertical_speed
        self.temp_data = temp_data  # shape: (N_climb, N_levels) or None for ISA
        self.pressure_levels = pressure_levels  # shape: (N_levels,) in hPa or None
        self.climb_indices = climb_indices  # Original indices into met data
        self.aircraft_type = aircraft_type
        self.use_isa = use_isa  # True = use ISA temperature, False = use met data
        # NaN in true_airspeed marks an unreliable observation. The point stays in
        # the time grid (altitude constraint active) but is excluded from the Ek
        # term of the objective via this mask.
        self.tas_valid = ~np.isnan(true_airspeed)

    def __len__(self) -> int:
        return len(self.time)

    def get_temperature_at_obs(self) -> np.ndarray:
        """Get temperature at observed altitudes (for display/validation)."""
        temps = np.zeros(len(self.altitude))
        for i, h in enumerate(self.altitude):
            if self.use_isa or self.temp_data is None or self.pressure_levels is None:
                # Use ISA temperature
                temps[i] = float(isa.air_temperature(h))
            else:
                # Interpolate from met data
                p_hpa = float(isa.air_pressure(h)) / 100.0
                temps[i] = np.interp(p_hpa, self.pressure_levels, self.temp_data[i])
        return temps


# =============================================================================
# Data Loading
# =============================================================================


def load_climb_data(
    data_path: Path,
    met_path: Path | None = None,
    min_altitude_ft: float = 10000.0,
    max_altitude_ft: float = 38000.0,
    min_rocd_ms: float = 2.0,
) -> ClimbData:
    """Load and filter flight data for climb phase with meteorological data.

    NaN values in true_airspeed are preserved: the point stays in the time grid
    so the altitude constraint and dt structure remain intact, but its TAS is
    excluded from the kinetic-energy term of the objective (tas_valid mask).

    Args:
        data_path: Path to flight_data.json.
        met_path: Path to met.npz with temperature/wind data. If None, use ISA.
        min_altitude_ft: Minimum altitude for climb phase [ft].
        max_altitude_ft: Maximum altitude for climb phase [ft].
        min_rocd_ms: Minimum rate of climb [m/s].

    Returns:
        ClimbData container with filtered data and met profiles.
    """
    # Load flight data
    with data_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Extract arrays
    time = np.array(raw["time"])
    altitude_ft = np.array(raw["altitude_ft"])
    vertical_speed_fpm = np.array(raw["vertical_speed"])
    # TAS in flight_data.json is already in m/s (not knots)
    true_airspeed_ms = np.array(raw["true_airspeed"])

    # Convert to SI
    altitude_m = altitude_ft * FT
    vertical_speed_ms = vertical_speed_fpm * FPM

    # Load meteorological data if provided
    use_isa = met_path is None
    if use_isa:
        temp_data = None
        pressure_levels = None
    else:
        met = np.load(met_path)
        temp_data = met["air_temperature_data"]  # shape: (N_points, N_levels)
        pressure_levels = met["pressure_levels"]  # hPa

    # Filter: altitude and ROCD must be valid; NaN TAS is allowed (marks outlier).
    valid = ~np.isnan(altitude_ft) & ~np.isnan(vertical_speed_fpm)
    climb = (
        valid
        & (vertical_speed_ms > min_rocd_ms)
        & (altitude_ft > min_altitude_ft)
        & (altitude_ft < max_altitude_ft)
    )

    climb_idx = np.where(climb)[0]

    if climb.sum() < 3:
        msg = (
            f"Insufficient climb data: only {climb.sum()} points. "
            f"Need at least 3 points for estimation."
        )
        raise ValueError(msg)

    # Extract climb segment indices
    idx = np.where(climb)[0]
    t0 = time[idx[0]]

    # Extract temperature profiles for climb points (if met data provided)
    if temp_data is not None:
        # Ensure all climb indices have corresponding met data
        valid_met_idx = idx[idx < len(temp_data)]
        if len(valid_met_idx) < len(idx):
            console.print(
                f"[yellow]Warning: {len(idx) - len(valid_met_idx)} climb points "
                f"exceed met data range, falling back to ISA[/yellow]"
            )
            use_isa = True
            climb_temp_data = None
            pressure_levels = None
        else:
            # Direct indexing - idx[k] maps to temp_data[idx[k]]
            climb_temp_data = temp_data[idx]
    else:
        climb_temp_data = None

    return ClimbData(
        time=time[idx] - t0,  # Relative time from first climb point
        altitude=altitude_m[idx],
        true_airspeed=true_airspeed_ms[idx],  # may contain NaN (marks outlier)
        vertical_speed=vertical_speed_ms[idx],
        temp_data=climb_temp_data,
        pressure_levels=pressure_levels,
        climb_indices=idx,
        aircraft_type=str(raw.get("aircraft_type", "B738")),
        use_isa=use_isa,
    )


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

    Raises:
        ImportError: If required dependencies are not available.
        ValueError: If model type is unknown or not supported.
    """
    model_type = model_type.lower()

    if model_type == "openap":
        from openleaf.performance.models.openap import OpenAPModel

        return OpenAPModel(aircraft_type)

    if model_type == "ps":
        from openleaf.performance.models.ps import PSModel

        return PSModel(aircraft_type)

    if model_type == "bada3":
        from openleaf.performance.models.bada3 import BADA3Model

        return BADA3Model(aircraft_type)

    if model_type == "bada4":
        from openleaf.performance.models.bada4 import BADA4Model

        return BADA4Model(aircraft_type)

    raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# CasADi Functions Builder
# =============================================================================


def build_dynamics_at_point(
    model: AircraftModel,
    temp_interpolant: ca.Function | None,
    m_sym: ca.MX,
    h_sym: ca.MX,
    V_sym: ca.MX,
    vs_sym: ca.MX | float = 0.0,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    """Build dynamics expressions for a single trajectory point.

    Creates symbolic expressions for drag, thrust, fuel flow using
    a temperature interpolant that maps pressure [hPa] -> temperature [K].

    Args:
        model: Aircraft performance model (BADA3, BADA4, PS, or OpenAP).
        temp_interpolant: CasADi interpolant: pressure_hPa -> temperature_K.
            If None, use ISA temperature.
        m_sym: Symbolic mass [kg].
        h_sym: Symbolic altitude [m].
        V_sym: Symbolic TAS [m/s].
        vs_sym: Symbolic vertical speed [m/s]. For BADA3/4/PS this is unused
            (thrust_max has no ROCD dependency). For OpenAP this is required;
            pass the TEM-derived CasADi expression to let IPOPT resolve the
            implicit vs ↔ thrust relationship. Defaults to 0.0.

    Returns:
        Tuple of (drag_force, thrust_max, fuel_flow, temperature) as CasADi MX.
    """
    import openleaf.performance.numpy as anp
    from openleaf.performance.core.perfos import EngineSetting
    from openleaf.performance.core.types import AerodynamicConfig

    # Atmospheric state
    p = isa.air_pressure(h_sym)  # Pa
    p_hpa = p / 100.0  # hPa for interpolant

    # Get temperature from met data interpolant or ISA
    if temp_interpolant is not None:
        T = temp_interpolant(p_hpa)  # K
    else:
        T = isa.air_temperature(h_sym)  # ISA temperature [K]

    # ISA temperature and delta_ISA
    T_isa = isa.air_temperature(h_sym)
    delta_isa = T - T_isa  # Deviation from ISA [K] (0 for ISA mode)

    # Air density from ideal gas law
    rho = p / (isa.R * T)  # kg/m³

    # Speed of sound and Mach number
    a = anp.sqrt(isa.GAMMA * isa.R * T)
    mach = V_sym / a

    # Dynamic pressure
    q = 0.5 * rho * V_sym**2

    # Wing reference area
    S = model.aerodynamics.data.surface_area

    # Lift coefficient using model's generic interface
    # model.lift_coef requires air_density in data
    data_for_cl = {
        "aircraft_mass": m_sym,
        "true_airspeed": V_sym,
        "air_density": rho,
        "load_factor": 1.0,
    }
    cl = model.lift_coef(data_for_cl)  # type: ignore[arg-type]

    # Build data dict for model calls.
    data = {
        "aircraft_mass": m_sym,
        "altitude": h_sym,
        "true_airspeed": V_sym,
        "air_pressure": p,
        "air_temperature": T,
        "air_density": rho,
        "aerodynamic_configuration": AerodynamicConfig.CR,
        "mach_number": mach,
        "lift_coef": cl,
        "delta_isa": delta_isa,  # For BADA3/BADA4
        "engine_setting": EngineSetting.MCMB,  # Max climb thrust
        "vertical_speed": vs_sym,  # TEM-derived symbolic vs (for OpenAP)
    }

    # Drag coefficient from model
    cd = model.aerodynamics.drag_coef(data)  # type: ignore[arg-type]

    # Drag FORCE = CD * q * S
    drag_force = cd * q * S

    # Maximum climb thrust
    thrust_max = model.thrust_max(data)  # type: ignore[arg-type]

    # Note: PS model with default thrust_buffer=0.0 now aligns with BADA models
    # for max continuous climb (MCC) thrust. No scaling needed.

    # Fuel flow at max thrust
    data_with_thrust = {**data, "thrust": thrust_max}
    fuel_flow = model.engine.fuel_flow(data_with_thrust)  # type: ignore[arg-type]

    return drag_force, thrust_max, fuel_flow, T


# =============================================================================
# Mass Estimation via Multiple Shooting
# =============================================================================


def estimate_mass(
    data: ClimbData,
    model: AircraftModel,
    verbose: bool = True,
) -> dict:
    """Estimate initial aircraft mass from observed climb trajectory.

    Formulation:
        Initial conditions (FIXED at k=0 only):
            - h[0] = h_obs[0]
            - V[0] = V_obs[0]

        Decision variables:
            - m0: Initial mass [kg]
            - f[0:N-1]: Energy share factors [-]

        State propagation (TEM dynamics):
            Ps[k] = (T_max[k] - D[k]) * V[k] / m[k]  (specific excess power)
            h[k+1] = h[k] + f[k] * Ps[k] / g * dt[k]
            V[k+1] = V[k] + (1 - f[k]) * Ps[k] / V[k] * dt[k]
            m[k+1] = m[k] - FF[k] * dt[k]

        Note: T_max and D depend on current state (h[k], V[k], m[k])

        Objective:
            min sum((Ep_model - Ep_obs)^2) + sum((Ek_model - Ek_obs)^2)
            where Ep = g*h (specific potential energy)
                  Ek = 0.5*V^2 (specific kinetic energy)

    Args:
        data: ClimbData container from load_climb_data().
        model: Aircraft performance model.
        verbose: Print optimization progress.

    Returns:
        Dictionary with estimation results.
    """
    # Extract observations (used for initial conditions and objective)
    t_obs = data.time
    h_obs = data.altitude
    V_obs = data.true_airspeed
    # NaN TAS points are excluded from the Ek term (weight = 0). Replace NaN with
    # 0 so CasADi receives a finite value — the zero weight makes the choice moot.
    V_obs_safe = np.where(data.tas_valid, V_obs, 0.0)
    N = len(t_obs)

    # Time steps
    dt = np.diff(t_obs)

    # Create temperature interpolants for each time point (or None for ISA)
    if data.use_isa or data.temp_data is None or data.pressure_levels is None:
        temp_interpolants: list[ca.Function | None] = [None] * N
        if verbose:
            console.print("[dim]Using ISA temperature model[/dim]")
    else:
        pressure_levels = list(data.pressure_levels)
        temp_interpolants = []
        for k in range(N):
            temp_profile = list(data.temp_data[k])
            interp = ca.interpolant(f"T_{k}", "linear", [pressure_levels], temp_profile)
            temp_interpolants.append(interp)
        if verbose:
            console.print(
                f"[dim]Using met data: {len(data.pressure_levels)} levels[/dim]"
            )

    # Detect whether model needs symbolic vertical speed (OpenAP thrust_max
    # requires ROCD; we introduce it as an opti variable and constrain it to
    # the TEM-derived expression, letting IPOPT solve the implicit equation).
    try:
        from openleaf.performance.models.openap import OpenAPModel
        _needs_vs = isinstance(model, OpenAPModel)
    except ImportError:
        _needs_vs = False

    # Get mass bounds from model
    oem = model.oem
    mtow = model.mtom
    m_guess = 0.8 * mtow  # Typical operating mass

    if verbose:
        console.print(f"[dim]Mass bounds: OEM={oem:.0f} kg, MTOW={mtow:.0f} kg[/dim]")
        console.print(f"[dim]Data points: N={N}, dt_mean={np.mean(dt):.1f} s[/dim]")

    # --- CasADi Opti formulation ---
    opti = ca.Opti()

    # Decision variables: m0, f, and (for OpenAP) per-step vertical speed.
    # h and V are state variables that evolve via dynamics (not decision variables).
    # For OpenAP, vs_k is introduced as an explicit decision variable because
    # thrust_max(vs_k) and vs_k = f*Ps(thrust_max)/g form an implicit equation
    # that IPOPT resolves as part of the NLP.
    m0 = opti.variable()  # Initial mass [kg]
    esf = opti.variable(N - 1)  # Energy share factors [-]
    if _needs_vs:
        vs_opti = opti.variable(N - 1)  # Vertical speed at each interval [m/s]
    else:
        vs_opti = None

    # --- State propagation using TEM dynamics ---
    # Initialize states at first observation
    h = [h_obs[0]]  # h[0] is FIXED at observed value
    V = [V_obs[0]]  # V[0] is FIXED at observed value
    m = [m0]

    # Sub-step size cap: ADS-B gaps larger than this are split into equal pieces.
    # Removing an outlier can merge two 120 s intervals into a single 240 s step;
    # forward-Euler accuracy degrades sharply, causing large TAS prediction errors.
    MAX_DT_SUB = 60.0  # seconds

    for k in range(N - 1):
        # Current state values (h[k], V[k] are symbolic for k>0)
        h_k = h[-1]
        V_k = V[-1]
        m_k = m[-1]

        # Sub-step this interval so no single Euler step exceeds MAX_DT_SUB.
        # n_sub is a Python int (dt is numpy), so the loop unrolls at graph-
        # build time and remains fully CasADi-symbolic.
        n_sub = max(1, int(round(dt[k] / MAX_DT_SUB)))
        dt_sub = dt[k] / n_sub

        # For OpenAP, vs is a symbolic decision variable constrained to match the
        # TEM-derived ROCD at the START of the interval (entry-state evaluation).
        vs_k = vs_opti[k] if vs_opti is not None else 0.0

        for s in range(n_sub):
            # Use the met interpolant for the start of this ADS-B interval (k).
            D_s, T_max_s, ff_s, T_s = build_dynamics_at_point(
                model, temp_interpolants[k], m_k, h_k, V_k, vs_sym=vs_k
            )

            # Specific excess power [W/kg]
            Ps_s = (T_max_s - D_s) * V_k / m_k

            # Temperature correction for pressure altitude rate
            T_isa_s = isa.air_temperature(h_k)

            # Add OpenAP vs constraint once, at the first sub-step.
            if vs_opti is not None and s == 0:
                vs_tem = esf[k] * (Ps_s / G) * (T_isa_s / T_s)
                opti.subject_to(vs_opti[k] == vs_tem)
                # After the constraint is set, future sub-steps reuse the same vs_k
                # (ESF is constant within the ADS-B interval).

            # TEM forward-Euler sub-step
            h_k = h_k + esf[k] * (Ps_s / G) * (T_isa_s / T_s) * dt_sub
            V_k = V_k + (1 - esf[k]) * (Ps_s / V_k) * dt_sub
            m_k = m_k - ff_s * dt_sub

        # Append end-of-interval state to trajectory
        h.append(h_k)
        V.append(V_k)
        m.append(m_k)

    # Convert to CasADi vectors
    h = ca.vertcat(*h)
    V = ca.vertcat(*V)
    m = ca.vertcat(*m)

    # --- Objective: minimize potential + kinetic energy errors separately ---
    # This prevents trading off altitude against speed.
    # TAS-invalid points (local outliers) are excluded from the Ek term:
    # their altitude constraint remains active but their bad TAS does not
    # pull the optimiser toward a physically implausible velocity trajectory.
    Ep_obs = G * h_obs  # Specific potential energy [J/kg]
    Ep_model = G * h
    Ek_obs = 0.5 * V_obs_safe**2  # Specific kinetic energy [J/kg]
    Ek_model = 0.5 * V**2

    Ep_err = Ep_model - Ep_obs
    Ek_err = Ek_model - Ek_obs

    # Zero the Ek weight for flagged samples (tas_valid=False)
    ek_weights = ca.DM(data.tas_valid.astype(float))
    obj = ca.sumsqr(Ep_err) + ca.dot(ek_weights, Ek_err * Ek_err)
    opti.minimize(obj)

    # --- Bounds ---
    opti.subject_to(opti.bounded(oem, m0, mtow))
    for k in range(N):
        opti.subject_to(m[k] >= oem)
    for k in range(N - 1):
        # ESF in [0, 1.5]: 0 = all to acceleration, 1 = all to climb
        # Values > 1 can occur due to tailwinds, updrafts, or model errors
        opti.subject_to(opti.bounded(0.0, esf[k], 1.5))

    # --- Initial guesses ---
    opti.set_initial(m0, m_guess)
    opti.set_initial(esf, 0.9 * np.ones(N - 1))  # Start with mostly climbing
    if vs_opti is not None:
        opti.set_initial(vs_opti, 5.0 * np.ones(N - 1))  # ~1000 fpm initial guess

    # --- Solve ---
    opts = {
        "ipopt.print_level": 5 if verbose else 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "print_time": verbose,
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
    except RuntimeError as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        raise

    # Extract solution
    m0_opt = float(sol.value(m0))
    f_opt = np.array(sol.value(esf)).flatten()
    h_opt = np.array(sol.value(h)).flatten()
    V_opt = np.array(sol.value(V)).flatten()
    m_traj = np.array(sol.value(m)).flatten()

    # Compute specific energy error (use safe TAS to avoid NaN in Es_obs)
    Es_obs = G * h_obs + 0.5 * V_obs_safe**2
    Es_opt = G * h_opt + 0.5 * V_opt**2
    Es_error = Es_opt - Es_obs

    return {
        "m0": m0_opt,
        "m_trajectory": m_traj,
        "h_opt": h_opt,
        "V_opt": V_opt,
        "f_opt": f_opt,
        "Es_error": Es_error,
        "solver_stats": sol.stats(),
    }


# =============================================================================
# Display and Plotting
# =============================================================================


def display_results(result: dict, model_name: str, aircraft_type: str) -> None:
    """Display estimation results in a formatted table."""
    table = Table(title=f"Mass Estimation Results - {model_name.upper()}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Unit", style="dim")

    table.add_row(
        "Initial Mass (m0)",
        f"{result['m0']:,.0f}",
        "kg",
    )
    table.add_row(
        "Final Mass",
        f"{result['m_trajectory'][-1]:,.0f}",
        "kg",
    )
    table.add_row(
        "Fuel Burned",
        f"{result['m0'] - result['m_trajectory'][-1]:,.0f}",
        "kg",
    )
    f_mean = np.mean(result["f_opt"])
    table.add_row(
        "Mean Energy Share (f)",
        f"{f_mean:.3f}",
        "-",
    )
    Es_rmse = np.sqrt(np.mean(result["Es_error"] ** 2))
    table.add_row(
        "Specific Energy RMSE",
        f"{Es_rmse:,.0f}",
        "J/kg",
    )

    console.print(table)

    # Additional statistics
    f_opt = result["f_opt"]
    console.print(
        f"\nEnergy share f: min={f_opt.min():.3f}, max={f_opt.max():.3f}, "
        f"std={f_opt.std():.3f}"
    )


def plot_results(
    data: ClimbData,
    result: dict,
    save_path: Path | None = None,
) -> None:
    """Plot mass estimation results."""
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Altitude comparison
    ax = axes[0, 0]
    ax.plot(data.altitude / FT, "o", label="Observed", markersize=4)
    ax.plot(result["h_opt"] / FT, "s-", label="Model", markersize=4)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Altitude [ft]")
    ax.set_title("Altitude Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TAS comparison (convert m/s to kt for display)
    ax = axes[0, 1]
    ax.plot(data.true_airspeed * MS_TO_KT, "o", label="Observed", markersize=4)
    ax.plot(result["V_opt"] * MS_TO_KT, "s-", label="Model", markersize=4)
    ax.set_xlabel("Sample")
    ax.set_ylabel("TAS [kt]")
    ax.set_title("True Airspeed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy share factor
    ax = axes[1, 0]
    ax.plot(result["f_opt"], "o-", color="green")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="f=1")
    ax.set_xlabel("Sample")
    ax.set_ylabel("f [-]")
    ax.set_title("Energy Share Factor")
    ax.set_ylim(0, max(1.1, result["f_opt"].max() * 1.1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mass trajectory
    ax = axes[1, 1]
    ax.plot(result["m_trajectory"] / 1000, "o-", color="purple")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Mass [tonnes]")
    ax.set_title(f"Mass Trajectory (m0 = {result['m0'] / 1000:.1f} t)")
    ax.grid(True, alpha=0.3)

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

app = cyclopts.App(
    name="mass-estimation",
    help="Estimate aircraft mass from climb trajectory data.",
)


@app.default
def main(
    model: Annotated[
        Literal["ps", "bada3", "bada4", "openap"],
        cyclopts.Parameter(help="Performance model."),
    ] = "bada3",
    data: Annotated[
        Path,
        cyclopts.Parameter(help="Path to flight data JSON file"),
    ] = _DEFAULT_DATA,
    met: Annotated[
        Path | None,
        cyclopts.Parameter(help="Path to meteorological data (met.npz). Uses ISA if omitted."),
    ] = None,
    save_plot: Annotated[
        Path | None,
        cyclopts.Parameter(help="Path to save result plot"),
    ] = None,
    quiet: Annotated[
        bool,
        cyclopts.Parameter(help="Suppress solver output"),
    ] = False,
) -> None:
    """Run mass estimation example."""
    console.rule("[bold]Aircraft Mass Estimation from Climb Data[/bold]")
    console.print(f"Model: [cyan]{model.upper()}[/cyan]")
    console.print(f"Data: [dim]{data}[/dim]")
    if met:
        console.print(f"Met: [dim]{met}[/dim]")
    else:
        console.print("Met: [dim]ISA (no met data)[/dim]")

    # Load data
    try:
        climb_data = load_climb_data(data, met)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    aircraft_type = climb_data.aircraft_type
    console.print(f"Aircraft: [cyan]{aircraft_type}[/cyan]")
    console.print(
        f"Climb data: [green]{len(climb_data)} points[/green], "
        f"altitude {climb_data.altitude.min() / FT:.0f}-{climb_data.altitude.max() / FT:.0f} ft"
    )

    # Show temperature info (at observed altitudes)
    T_obs = climb_data.get_temperature_at_obs()
    T_isa = np.array([float(isa.air_temperature(h)) for h in climb_data.altitude])
    delta_T = T_obs - T_isa
    console.print(
        f"Temperature: actual={T_obs.mean():.1f} K, delta_ISA={delta_T.mean():+.1f} K"
    )

    # Load model
    try:
        perf_model = load_model(model, aircraft_type)
    except ImportError as e:
        console.print(f"[red]Cannot load {model} model: {e}[/red]")
        console.print("[yellow]Try --model ps (requires fewer dependencies)[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Model loading failed: {e}[/red]")
        sys.exit(1)

    console.print(f"Model loaded: MTOW={perf_model.mtom:.0f} kg, OEM={perf_model.oem:.0f} kg")

    # Run estimation
    console.print("\n[bold]Running optimization...[/bold]")
    result = estimate_mass(climb_data, perf_model, verbose=not quiet)

    # Display results
    display_results(result, model, aircraft_type)

    # Plot
    path = save_plot or Path(__file__).parent / "mass_estimation.png"
    plot_results(climb_data, result, save_path=path)


if __name__ == "__main__":
    app()
