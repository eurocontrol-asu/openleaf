#!/usr/bin/env python3
"""
Cost Index Estimation using AetherX PS Model.

This example demonstrates how to estimate the cost index (CI) from observed
flight data using AetherX - whose functions work with both numpy AND CasADi!

The cost index represents the ratio of time-related costs to fuel costs:
    CI = C_time / C_fuel [kg/min]

At the optimal cruise condition, the gradient of specific cost with respect
to speed and altitude should be zero. This example uses this optimality
condition to infer the CI that best explains the observed cruise behavior.

Method:
    1. Extract cruise phase from flight trajectory
    2. Build symbolic specific cost function: q = (FF + CI/60) / GS
    3. Compute gradients dq/dV and dq/dh at each cruise point
    4. Find CI that minimizes sum of squared gradients (optimality condition)

Requirements:
    - pycontrails: For Flight class and phase detection
    - casadi: For symbolic differentiation
    - aetherx: For PS model, ISA, and aero functions

Model support:
    - PS (Poll-Schumann): Primary model, designed for cruise optimization
    - BADA3/BADA4: Also supported for comparison

Usage:
    python examples/03_cost_index_estimation.py [--model ps|bada3|bada4]

References:
    - Poles et al. (2010): "Aircraft Mass Estimation from Trajectory Data"
    - Poll & Schumann (2021): "An estimation method for the fuel burn..."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated
from typing import Literal

import casadi as ca
import cyclopts
from rich.console import Console
from rich.table import Table

import openleaf.performance.numpy as anp
import numpy as np
from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.physics import isa
from openleaf.performance.physics import units


console = Console()


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_type: str, aircraft_type: str):
    """Load aircraft performance model.

    Args:
        model_type: One of 'ps', 'bada3', 'bada4', 'openap'.
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

    if model_type == "openap":
        from openleaf.performance.models.openap import OpenAPModel

        return OpenAPModel(aircraft_type)

    msg = f"Unknown model type: {model_type}"
    raise ValueError(msg)


# =============================================================================
# Cruise Phase Extraction
# =============================================================================


def extract_cruise_data(
    flight_data_path: Path,
    met_data_path: Path,
    aircraft_mass: float,
) -> dict:
    """Extract cruise phase data from flight trajectory.

    Args:
        flight_data_path: Path to flight_data.json.
        met_data_path: Path to met.npz.
        aircraft_mass: Assumed aircraft mass [kg].

    Returns:
        Dictionary with cruise phase data arrays.
    """
    # Load flight data
    with flight_data_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Extract arrays
    time = np.array(raw["time"])
    altitude_ft = np.array(raw["altitude_ft"])
    vertical_speed_fpm = np.array(raw["vertical_speed"])
    true_airspeed_ms = np.array(raw["true_airspeed"])

    # Convert to SI
    altitude_m = altitude_ft * units.FT
    vertical_speed_ms = vertical_speed_fpm * units.FPM

    # Load met data
    # Note: Met data may be pre-filtered to cruise points only
    met = np.load(met_data_path)
    wind_data = met["longitudinal_wind_data"]  # shape: (N_cruise, N_levels)
    temp_data = met["air_temperature_data"]
    pressure_levels = met["pressure_levels"]  # hPa

    # Filter cruise phase: stable altitude (small vertical speed)
    # and above FL250 (typical cruise)
    cruise_mask = (np.abs(vertical_speed_ms) < 1.0) & (altitude_ft > 25000)

    if cruise_mask.sum() < 5:
        msg = f"Insufficient cruise data: only {cruise_mask.sum()} points"
        raise ValueError(msg)

    idx = np.where(cruise_mask)[0]
    n_cruise = len(idx)

    # Handle met data that may be pre-filtered to cruise points
    # If met data size matches cruise size, use direct indexing (0..N-1)
    # Otherwise, try to use original indices
    n_met = len(wind_data)
    if n_met == n_cruise:
        # Met data is already cruise-only, use sequential indices
        met_wind = wind_data
        met_temp = temp_data
    elif n_met >= idx.max() + 1:
        # Met data covers full flight, index with cruise indices
        met_wind = wind_data[idx]
        met_temp = temp_data[idx]
    else:
        # Fallback: use last available met data for all points
        console.print(
            f"[yellow]Warning: Met data ({n_met} points) doesn't match "
            f"cruise ({n_cruise} points). Using repeated met data.[/yellow]"
        )
        met_wind = np.tile(wind_data[-1], (n_cruise, 1))
        met_temp = np.tile(temp_data[-1], (n_cruise, 1))

    return {
        "time": time[idx],
        "altitude": altitude_m[idx],
        "true_airspeed": true_airspeed_ms[idx],
        "aircraft_mass": np.full(n_cruise, aircraft_mass),
        "wind_data": met_wind,
        "temp_data": met_temp,
        "pressure_levels": pressure_levels,
        "aircraft_type": str(raw.get("aircraft_type", "B738")),
        "n_points": n_cruise,
    }


# =============================================================================
# Specific Cost Gradient Function
# =============================================================================


def build_specific_cost_gradients(
    model,
    wind_interpolant: ca.Function,
    temp_interpolant: ca.Function,
) -> ca.Function:
    """Build CasADi function for specific cost gradients.

    The specific cost is:
        q = (fuel_flow + CI/60) / ground_speed

    At optimal cruise, dq/dV = 0 and dq/dh = 0.

    Args:
        model: Aircraft performance model (PSModel, BADA3Model, etc.).
        wind_interpolant: CasADi interpolant: pressure_hPa -> wind_speed [m/s].
        temp_interpolant: CasADi interpolant: pressure_hPa -> temperature [K].

    Returns:
        CasADi function F(x, ci) -> (dq/dV, dq/dh).
    """
    # Symbolic state variables
    altitude = ca.SX.sym("h")
    true_airspeed = ca.SX.sym("v")
    aircraft_mass = ca.SX.sym("m")
    ci = ca.SX.sym("CI")

    # Atmospheric state using AetherX ISA (works with CasADi!)
    air_pressure = isa.air_pressure(altitude)
    pressure_hpa = air_pressure / 100.0

    # Temperature from met data interpolant
    air_temperature = temp_interpolant(pressure_hpa)

    # ISA temperature and delta_ISA
    temp_isa = isa.air_temperature(altitude)
    delta_isa = air_temperature - temp_isa

    # Air density and speed of sound
    air_density = air_pressure / (isa.R * air_temperature)
    speed_of_sound = anp.sqrt(isa.GAMMA * isa.R * air_temperature)
    mach = true_airspeed / speed_of_sound

    # Dynamic pressure
    q_dyn = 0.5 * air_density * true_airspeed**2

    # Wing area
    S = model.aerodynamics.data.surface_area

    # Lift coefficient (steady level flight)
    c_lift = (aircraft_mass * isa.G) / (q_dyn * S)

    # Build data dict for model calls (common interface for all models)
    data = {
        "aircraft_mass": aircraft_mass,
        "altitude": altitude,
        "true_airspeed": true_airspeed,
        "air_pressure": air_pressure,
        "air_temperature": air_temperature,
        "air_density": air_density,
        "aerodynamic_configuration": AerodynamicConfig.CR,
        "mach_number": mach,
        "lift_coef": c_lift,
        "delta_isa": delta_isa,
        "engine_setting": EngineSetting.MCRZ,  # Max Continuous Cruise
        "vertical_speed": 0.0,  # Level cruise
    }

    # Drag coefficient (unified interface for all models)
    c_drag = model.aerodynamics.drag_coef(data)

    # Drag force
    drag = c_drag * q_dyn * S

    # Thrust = Drag for steady level flight
    thrust = drag

    # Fuel flow from engine model
    data_with_thrust = {**data, "thrust": thrust}
    fuel_flow = model.engine.fuel_flow(data_with_thrust)

    # Ground speed (TAS + headwind component)
    wind_component = wind_interpolant(pressure_hpa)
    ground_speed = true_airspeed + wind_component

    # Specific cost: cost per unit distance
    # CI in kg/min, fuel_flow in kg/s, ground_speed in m/s
    specific_cost = (fuel_flow + ci / 60.0) / ground_speed

    # Build CasADi function returning gradients
    x_state = ca.vertcat(true_airspeed, altitude, aircraft_mass)
    return ca.Function(
        "dq",
        [x_state, ci],
        [
            ca.gradient(specific_cost, true_airspeed),
            ca.gradient(specific_cost, altitude),
        ],
        ["x", "ci"],
        ["dqdv", "dqdh"],
    )


# =============================================================================
# Cost Index Estimation
# =============================================================================


def estimate_cost_index(
    data: dict,
    model,
    verbose: bool = True,
) -> dict:
    """Estimate cost index from cruise data.

    At optimal cruise conditions, the gradient of specific cost with
    respect to speed and altitude should be zero. We find the CI that
    minimizes the sum of squared gradients.

    Args:
        data: Cruise data dictionary from extract_cruise_data().
        model: Aircraft performance model.
        verbose: Whether to print progress.

    Returns:
        Dictionary with estimated CI and optimization details.
    """
    pressure_levels = list(data["pressure_levels"])
    n_points = data["n_points"]

    # Set up CasADi optimization
    opti = ca.Opti()
    ci = opti.variable()
    opti.set_initial(ci, 30.0)  # Typical CI starting point
    opti.subject_to(opti.bounded(0.0, ci, 999.0))

    # Compute gradients at each cruise point
    dqdv_list = []
    dqdh_list = []

    for i in range(n_points - 1):
        # Create interpolants for this point's met data
        wind_row = (
            data["wind_data"][i] if data["wind_data"].ndim > 1 else data["wind_data"]
        )
        temp_row = (
            data["temp_data"][i] if data["temp_data"].ndim > 1 else data["temp_data"]
        )

        wind_interp = ca.interpolant("w", "linear", [pressure_levels], list(wind_row))
        temp_interp = ca.interpolant("t", "linear", [pressure_levels], list(temp_row))

        # Build gradient function for this point
        F = build_specific_cost_gradients(model, wind_interp, temp_interp)

        # Evaluate at observed state
        x = ca.vertcat(
            data["true_airspeed"][i],
            data["altitude"][i],
            data["aircraft_mass"][i],
        )
        dq = F(x, ci)

        dqdv_list.append(dq[0])
        dqdh_list.append(dq[1])

    # Objective: minimize sum of squared gradients
    # At optimal cruise: dq/dV = dq/dh = 0
    cost = ca.sumsqr(ca.hcat(dqdv_list)) + ca.sumsqr(ca.hcat(dqdh_list))
    opti.minimize(cost)

    # Solve
    opti.solver(
        "ipopt", {"print_time": False}, {"print_level": 0 if not verbose else 3}
    )

    if verbose:
        console.print("[bold]Optimizing cost index...[/bold]")

    try:
        sol = opti.solve()
        estimated_ci = float(sol.value(ci))
        final_cost = float(sol.value(cost))
        success = True
    except RuntimeError as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        estimated_ci = 0.0
        final_cost = float("inf")
        success = False

    return {
        "cost_index": estimated_ci,
        "final_cost": final_cost,
        "n_points": n_points,
        "success": success,
    }


# =============================================================================
# Display Results
# =============================================================================


def display_results(result: dict, model_name: str) -> None:
    """Display estimation results."""
    table = Table(title=f"Cost Index Estimation - {model_name.upper()}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Unit", style="dim")

    table.add_row("Estimated Cost Index", f"{result['cost_index']:.1f}", "kg/min")
    table.add_row("Cruise Points Used", f"{result['n_points']}", "-")
    table.add_row("Final Residual", f"{result['final_cost']:.2e}", "-")

    console.print(table)

    # Interpretation
    ci = result["cost_index"]
    console.print("\n[bold]Interpretation:[/bold]")
    if ci < 20:
        console.print("  Low CI: Fuel-efficient cruise (long-haul, low fuel cost)")
    elif ci < 50:
        console.print("  Medium CI: Balanced time/fuel tradeoff (typical operations)")
    elif ci < 100:
        console.print("  High CI: Time-sensitive operations (short-haul, delays)")
    else:
        console.print("  Very high CI: Maximum speed priority (rare)")


# =============================================================================
# Main
# =============================================================================


_DEFAULT_DATA = Path(__file__).parent.parent / "data" / "flight_data.json"
_DEFAULT_MET = Path(__file__).parent.parent / "data" / "met.npz"

app = cyclopts.App(
    name="cost-index-estimation",
    help="Estimate airline cost index from cruise flight data.",
)


@app.default
def main(
    model: Annotated[
        Literal["ps", "bada3", "bada4"],
        cyclopts.Parameter(help="Performance model"),
    ] = "ps",
    data: Annotated[
        Path,
        cyclopts.Parameter(help="Path to flight data JSON"),
    ] = _DEFAULT_DATA,
    met: Annotated[
        Path,
        cyclopts.Parameter(help="Path to meteorological data"),
    ] = _DEFAULT_MET,
    mass: Annotated[
        float,
        cyclopts.Parameter(help="Aircraft mass [kg]"),
    ] = 69000.0,
) -> None:
    """Run cost index estimation example."""
    console.rule("[bold]Cost Index Estimation from Cruise Data[/bold]")
    console.print(f"Model: [cyan]{model.upper()}[/cyan]")
    console.print(f"Data: [dim]{data}[/dim]")
    console.print(f"Mass: [dim]{mass:.0f} kg[/dim]")

    # Load model
    try:
        # Get aircraft type from data
        with data.open(encoding="utf-8") as f:
            raw = json.load(f)
        aircraft_type = str(raw.get("aircraft_type", "B738"))

        perf_model = load_model(model, aircraft_type)
        console.print(f"Aircraft: [cyan]{aircraft_type}[/cyan]")
        console.print(
            f"Model: S={perf_model.aerodynamics.data.surface_area:.1f} m², "
            f"MTOW={perf_model.mtom:.0f} kg"
        )
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        sys.exit(1)

    # Extract cruise data
    try:
        cruise_data = extract_cruise_data(data, met, mass)
        console.print(
            f"Cruise data: [green]{cruise_data['n_points']} points[/green], "
            f"altitude {cruise_data['altitude'].min() / units.FT:.0f}-"
            f"{cruise_data['altitude'].max() / units.FT:.0f} ft"
        )
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        sys.exit(1)

    # Estimate cost index
    console.print()
    result = estimate_cost_index(cruise_data, perf_model, verbose=True)

    if not result["success"]:
        console.print("[red]Cost index estimation failed[/red]")
        sys.exit(1)

    # Display results
    console.print()
    display_results(result, model)

    console.print(
        f"\n[bold green]Final Result: CI = {result['cost_index']:.1f} kg/min[/bold green]"
    )


if __name__ == "__main__":
    app()
