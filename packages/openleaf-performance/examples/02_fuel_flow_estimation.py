#!/usr/bin/env python3
"""
Fuel Flow Evaluation from Flight Trajectory.

This example demonstrates how to compute fuel flow using AetherX's TrajectoryData
workflow. Given a known aircraft mass (e.g., from Example 01), it computes:
    - Required thrust from TEM (Total Energy Model)
    - Fuel flow at that thrust level
    - Total fuel consumption over the flight

This example shows the simple evaluation workflow WITHOUT optimization.
It uses actual meteorological data (air temperature) from met.npz.

Method:
    - Load full flight trajectory (all phases)
    - Provide known mass trajectory (from mass estimation or assumed)
    - Use TrajectoryData.with_aircraft() to attach performance model
    - Access .fuel_flow property for automatic computation

The computation chain is:
    1. TrajectoryData holds: altitude, TAS, vertical_speed, mass, f
    2. Model computes: drag_coef -> drag -> thrust (via TEM) -> fuel_flow

Supports all AetherX performance models:
    - OpenAP (open source, always available)
    - PS (Poll-Schumann, requires pycontrails)
    - BADA3/BADA4 (requires EUROCONTROL license)

Usage:
    python examples/02_fuel_flow_estimation.py [--model openap|ps|bada3|bada4]

References:
    - BADA User Manual, Section 3.7: Fuel Consumption Model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated
from typing import Literal

import cyclopts
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

import numpy as np
from openleaf.performance.core.trajectory import TrajectoryData
from openleaf.performance.physics import isa
from openleaf.performance.physics.units import FPM
from openleaf.performance.physics.units import FT
from openleaf.performance.physics.units import G


console = Console()


# =============================================================================
# Data Loading
# =============================================================================


def compute_acceleration_trapezoidal(
    velocity: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Compute acceleration using trapezoidal differentiation.

    Uses central differences for interior points and forward/backward
    differences at boundaries.

    Args:
        velocity: True airspeed array [m/s].
        time: Time array [s].

    Returns:
        Acceleration array [m/s²], same length as input.
    """
    n = len(velocity)
    accel = np.zeros(n)

    # Forward difference for first point
    if n > 1:
        accel[0] = (velocity[1] - velocity[0]) / (time[1] - time[0])

    # Central differences for interior points (trapezoidal)
    for i in range(1, n - 1):
        dt = time[i + 1] - time[i - 1]
        if dt > 0:
            accel[i] = (velocity[i + 1] - velocity[i - 1]) / dt

    # Backward difference for last point
    if n > 1:
        accel[-1] = (velocity[-1] - velocity[-2]) / (time[-1] - time[-2])

    return accel


def compute_energy_share_factor(
    true_airspeed: np.ndarray,
    vertical_speed: np.ndarray,
    acceleration: np.ndarray,
) -> np.ndarray:
    """Compute energy share factor from kinematics.

    The energy share factor f determines how thrust is distributed between
    climbing (potential energy) and accelerating (kinetic energy):

        f = 1 - (V/g) * (dV/dh)

    where dV/dh = (dV/dt) / (dh/dt) = acceleration / vertical_speed

    For level flight (vertical_speed ≈ 0), f = 1.0.
    For constant-speed climb, f = 1.0.
    For accelerating climb, f < 1.0.

    Args:
        true_airspeed: True airspeed [m/s].
        vertical_speed: Rate of climb/descent [m/s].
        acceleration: Longitudinal acceleration dV/dt [m/s²].

    Returns:
        Energy share factor array, clipped to [0, 1].
    """
    # Avoid division by zero for level flight
    # When |vertical_speed| < threshold, use f = 1.0
    vs_threshold = 0.5  # m/s (~100 fpm)

    # Compute dV/dh = acceleration / vertical_speed
    # For small vertical speeds, this ratio can be undefined/large
    dv_dh = np.where(
        np.abs(vertical_speed) > vs_threshold,
        acceleration / vertical_speed,
        0.0,  # For level flight, assume no altitude-speed tradeoff
    )

    # f = 1 - (V/g) * (dV/dh)
    f = 1.0 - (true_airspeed / G) * dv_dh

    # Clip to physically reasonable range [0, 1]
    # f < 0 would mean decelerating while climbing (unusual)
    # f > 1 would mean accelerating while climbing faster than energy allows
    return np.clip(f, 0.0, 1.0)


def load_flight_data(
    data_path: Path,
    met_path: Path,
    initial_mass: float,
) -> dict:
    """Load flight data with meteorological conditions and mass trajectory.

    Args:
        data_path: Path to flight_data.json.
        met_path: Path to met.npz with temperature/wind data.
        initial_mass: Initial aircraft mass [kg].

    Returns:
        Dictionary with trajectory data in SI units, including:
            - time, altitude, true_airspeed, vertical_speed
            - air_temperature (from met data)
            - aircraft_mass (propagated from initial)
            - f (energy share factor, derived)
            - aircraft_type
    """
    # Load flight data
    with data_path.open(encoding="utf-8") as file:
        raw = json.load(file)

    # Extract arrays
    time = np.array(raw["time"])
    altitude_ft = np.array(raw["altitude_ft"])
    vertical_speed_fpm = np.array(raw["vertical_speed"])
    # TAS in flight_data.json is already in m/s (not knots)
    true_airspeed_ms = np.array(raw["true_airspeed"])

    # Convert to SI
    altitude_m = altitude_ft * FT
    vertical_speed_ms = vertical_speed_fpm * FPM

    # Load meteorological data
    met = np.load(met_path)
    temp_data = met["air_temperature_data"]  # shape: (N_points, N_levels)
    pressure_levels = met["pressure_levels"]  # hPa

    # Interpolate temperature at each point
    N = len(time)
    air_temperature = np.zeros(N)
    for i in range(N):
        # Get pressure at this altitude
        p_pa = isa.air_pressure(altitude_m[i])
        p_hpa = float(p_pa) / 100.0

        # Interpolate temperature from met data
        if i < len(temp_data):
            air_temperature[i] = np.interp(p_hpa, pressure_levels, temp_data[i])
        else:
            air_temperature[i] = float(isa.air_temperature(altitude_m[i]))

    # Filter valid data first (before computing derived quantities)
    valid = (
        ~np.isnan(altitude_ft)
        & ~np.isnan(vertical_speed_fpm)
        & ~np.isnan(true_airspeed_ms)
        & (true_airspeed_ms > 50)  # Filter unrealistic speeds
    )

    # Apply filter to all arrays
    time_filt = time[valid]
    altitude_filt = altitude_m[valid]
    tas_filt = true_airspeed_ms[valid]
    vs_filt = vertical_speed_ms[valid]
    temp_filt = air_temperature[valid]

    # Compute energy share factor from kinematics on filtered data
    # f = 1 - (V/g) * (dV/dh) where dV/dh = acceleration / vertical_speed
    #
    # This properly accounts for acceleration/deceleration during climb/descent
    # rather than assuming fixed values.
    acceleration = compute_acceleration_trapezoidal(tas_filt, time_filt)
    esf = compute_energy_share_factor(tas_filt, vs_filt, acceleration)

    # Initialize mass trajectory (constant mass as placeholder)
    aircraft_mass = np.full(len(time_filt), initial_mass)

    aircraft_type = str(raw.get("aircraft_type", "B738"))

    return {
        "time": time_filt,
        "altitude": altitude_filt,
        "true_airspeed": tas_filt,
        "vertical_speed": vs_filt,
        "air_temperature": temp_filt,
        "aircraft_mass": aircraft_mass,
        "f": esf,
        "aircraft_type": aircraft_type,
    }


# =============================================================================
# Fuel Flow Computation
# =============================================================================


def compute_fuel_flow(
    data: dict,
    model_type: str,
    aircraft_type: str,
) -> dict:
    """Compute fuel flow using TrajectoryData workflow.

    This demonstrates the standard AetherX workflow:
    1. Create TrajectoryData from observations
    2. Attach a pre-built aircraft model
    3. Access computed properties (drag, thrust, fuel_flow)

    Args:
        data: Flight data dictionary from load_flight_data().
        model_type: Performance model type.
        aircraft_type: ICAO aircraft type code.

    Returns:
        Dictionary with computed values:
            - fuel_flow: Fuel flow at each point [kg/s]
            - thrust: Required thrust [N]
            - drag: Aerodynamic drag [N]
            - total_fuel: Total fuel consumed [kg]
            - flight_duration: Total flight time [s]
    """
    # Import the specific model class to avoid factory loading all models
    model_type = model_type.upper()
    if model_type == "OPENAP":
        from openleaf.performance.models.openap import OpenAPModel

        model = OpenAPModel(aircraft_type)
    elif model_type == "PS":
        from openleaf.performance.models.ps import PSModel

        model = PSModel(aircraft_type)
    elif model_type == "BADA3":
        from openleaf.performance.models.bada3 import BADA3Model

        model = BADA3Model(aircraft_type)
    elif model_type == "BADA4":
        from openleaf.performance.models.bada4 import BADA4Model

        model = BADA4Model(aircraft_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create TrajectoryData
    traj = TrajectoryData(
        {  # type: ignore[arg-type]
            "altitude": data["altitude"],
            "true_airspeed": data["true_airspeed"],
            "vertical_speed": data["vertical_speed"],
            "aircraft_mass": data["aircraft_mass"],
            "air_temperature": data["air_temperature"],
            "f": data["f"],
        }
    )

    # Attach aircraft model using pre-built model instance
    from openleaf.performance.core.trajectory import AircraftTrajectoryData

    perf = AircraftTrajectoryData(traj, model)

    # Access computed properties
    # The chain is: drag_coef -> drag -> thrust (TEM) -> fuel_flow
    fuel_flow_raw = np.asarray(perf.fuel_flow)
    thrust_raw = np.asarray(perf.thrust)
    drag = np.asarray(perf.drag)

    # Get thrust limits for clamping
    thrust_max = np.asarray(perf.thrust_max)
    thrust_min = np.asarray(perf.thrust_min)  # Idle thrust
    fuel_flow_min = np.asarray(perf.fuel_flow_min)  # Idle fuel flow

    # Clamp thrust to physical limits [idle, max]
    thrust = np.clip(thrust_raw, thrust_min, thrust_max)

    # For descent (negative vertical speed), use idle fuel flow when thrust
    # would be below idle. This is physically correct: during descent,
    # aircraft typically fly at idle thrust, using potential energy.
    vs = data["vertical_speed"]
    is_descent = vs < -1.0
    fuel_flow = np.where(
        is_descent & (thrust_raw < thrust_min),
        fuel_flow_min,
        fuel_flow_raw,
    )
    # Also clamp fuel flow to be non-negative
    fuel_flow = np.maximum(fuel_flow, fuel_flow_min)

    # Compute total fuel consumption (trapezoidal integration)
    time = data["time"]
    dt = np.diff(time)
    ff_avg = (fuel_flow[:-1] + fuel_flow[1:]) / 2
    total_fuel = np.sum(ff_avg * dt)

    return {
        "fuel_flow": fuel_flow,
        "thrust": thrust,
        "drag": drag,
        "total_fuel": total_fuel,
        "flight_duration": time[-1] - time[0],
    }


# =============================================================================
# Display and Plotting
# =============================================================================


def display_results(
    data: dict,
    result: dict,
    model_name: str,
    aircraft_type: str,
) -> None:
    """Display fuel flow results in a formatted table."""
    table = Table(title=f"Fuel Flow Results - {model_name.upper()}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Unit", style="dim")

    table.add_row(
        "Flight Duration",
        f"{result['flight_duration'] / 60:.1f}",
        "min",
    )
    table.add_row(
        "Total Fuel Consumed",
        f"{result['total_fuel']:,.0f}",
        "kg",
    )
    table.add_row(
        "Mean Fuel Flow",
        f"{np.mean(result['fuel_flow']) * 3600:.0f}",
        "kg/h",
    )
    table.add_row(
        "Max Fuel Flow",
        f"{np.max(result['fuel_flow']) * 3600:.0f}",
        "kg/h",
    )
    table.add_row(
        "Min Fuel Flow",
        f"{np.min(result['fuel_flow']) * 3600:.0f}",
        "kg/h",
    )
    table.add_row(
        "Mean Thrust",
        f"{np.mean(result['thrust']) / 1000:.1f}",
        "kN",
    )
    table.add_row(
        "Mean Drag",
        f"{np.mean(result['drag']) / 1000:.1f}",
        "kN",
    )

    console.print(table)

    # Flight phase breakdown
    vs = data["vertical_speed"]
    climb_mask = vs > 1.0
    cruise_mask = np.abs(vs) <= 1.0
    descent_mask = vs < -1.0

    console.print("\n[bold]Fuel by Flight Phase:[/bold]")
    for name, mask in [
        ("Climb", climb_mask),
        ("Cruise", cruise_mask),
        ("Descent", descent_mask),
    ]:
        if mask.any():
            ff_phase = result["fuel_flow"][mask]
            console.print(
                f"  {name}: mean={np.mean(ff_phase) * 3600:.0f} kg/h, points={mask.sum()}"
            )


def plot_results(
    data: dict,
    result: dict,
    save_path: Path | None = None,
) -> None:
    """Plot fuel flow analysis results."""
    time_min = (data["time"] - data["time"][0]) / 60

    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Altitude profile
    ax = axes[0, 0]
    ax.plot(time_min, data["altitude"] / FT, "b-", linewidth=1)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Altitude [ft]")
    ax.set_title("Flight Altitude Profile")
    ax.grid(True, alpha=0.3)

    # Fuel flow over time
    ax = axes[0, 1]
    ax.plot(time_min, result["fuel_flow"] * 3600, "r-", linewidth=1)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Fuel Flow [kg/h]")
    ax.set_title("Fuel Flow")
    ax.grid(True, alpha=0.3)

    # Thrust and Drag
    ax = axes[1, 0]
    ax.plot(time_min, result["thrust"] / 1000, "g-", label="Thrust", linewidth=1)
    ax.plot(time_min, result["drag"] / 1000, "b-", label="Drag", linewidth=1)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Force [kN]")
    ax.set_title("Thrust and Drag")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative fuel consumption
    ax = axes[1, 1]
    dt = np.diff(data["time"])
    ff_avg = (result["fuel_flow"][:-1] + result["fuel_flow"][1:]) / 2
    cumulative_fuel = np.concatenate([[0], np.cumsum(ff_avg * dt)])
    ax.plot(time_min, cumulative_fuel, "m-", linewidth=1.5)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Cumulative Fuel [kg]")
    ax.set_title(f"Total Fuel: {result['total_fuel']:.0f} kg")
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
_DEFAULT_MET = Path(__file__).parent.parent / "data" / "met.npz"

app = cyclopts.App(
    name="fuel-flow-estimation",
    help="Compute fuel flow from flight trajectory data.",
)


@app.default
def main(
    model: Annotated[
        Literal["openap", "ps", "bada3", "bada4"],
        cyclopts.Parameter(help="Performance model to use"),
    ] = "openap",
    data: Annotated[
        Path,
        cyclopts.Parameter(help="Path to flight data JSON file"),
    ] = _DEFAULT_DATA,
    met: Annotated[
        Path,
        cyclopts.Parameter(help="Path to meteorological data (met.npz)"),
    ] = _DEFAULT_MET,
    mass: Annotated[
        float | None,
        cyclopts.Parameter(help="Initial aircraft mass [kg]. Default: 85%% MTOW"),
    ] = None,
    save_plot: Annotated[
        Path | None,
        cyclopts.Parameter(help="Path to save result plot"),
    ] = None,
) -> None:
    """Run fuel flow evaluation example."""
    console.rule("[bold]Fuel Flow Evaluation from Flight Trajectory[/bold]")
    console.print(f"Model: [cyan]{model.upper()}[/cyan]")
    console.print(f"Data: [dim]{data}[/dim]")
    console.print(f"Met: [dim]{met}[/dim]")

    # Load flight data to get aircraft type
    with data.open(encoding="utf-8") as file:
        raw = json.load(file)
    aircraft_type = str(raw.get("aircraft_type", "B738"))

    # Get model to determine default mass
    model_upper = model.upper()
    try:
        if model_upper == "OPENAP":
            from openleaf.performance.models.openap import OpenAPModel

            perf_model = OpenAPModel(aircraft_type)
        elif model_upper == "PS":
            from openleaf.performance.models.ps import PSModel

            perf_model = PSModel(aircraft_type)
        elif model_upper == "BADA3":
            from openleaf.performance.models.bada3 import BADA3Model

            perf_model = BADA3Model(aircraft_type)
        elif model_upper == "BADA4":
            from openleaf.performance.models.bada4 import BADA4Model

            perf_model = BADA4Model(aircraft_type)
        else:
            console.print(f"[red]Unknown model: {model_upper}[/red]")
            sys.exit(1)
    except ImportError as e:
        console.print(f"[red]Cannot import {model_upper} model: {e}[/red]")
        sys.exit(1)

    console.print(f"Aircraft: [cyan]{aircraft_type}[/cyan]")
    console.print(f"Model: MTOW={perf_model.mtom:.0f} kg, OEM={perf_model.oem:.0f} kg")

    # Determine initial mass
    if mass is not None:
        initial_mass = mass
    else:
        initial_mass = 0.85 * perf_model.mtom
        console.print(
            f"[dim]Using default mass: {initial_mass:.0f} kg (85% MTOW)[/dim]"
        )

    # Load data with met
    try:
        flight_data = load_flight_data(data, met, initial_mass)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        sys.exit(1)

    console.print(
        f"Flight data: [green]{len(flight_data['time'])} points[/green], "
        f"altitude {flight_data['altitude'].min() / FT:.0f}-{flight_data['altitude'].max() / FT:.0f} ft"
    )

    # Show temperature info
    T_isa = np.array([float(isa.air_temperature(h)) for h in flight_data["altitude"]])
    delta_T = flight_data["air_temperature"] - T_isa
    console.print(
        f"Temperature: mean={flight_data['air_temperature'].mean():.1f} K, "
        f"delta_ISA={delta_T.mean():+.1f} K"
    )

    # Compute fuel flow
    console.print("\n[bold]Computing fuel flow...[/bold]")
    try:
        result = compute_fuel_flow(flight_data, model, aircraft_type)
    except Exception as e:
        console.print(f"[red]Computation failed: {e}[/red]")
        sys.exit(1)

    # Display results
    display_results(flight_data, result, model, aircraft_type)

    # Plot
    path = save_plot or Path(__file__).parent / "fuel_flow_evaluation.png"
    plot_results(flight_data, result, save_path=path)


if __name__ == "__main__":
    app()
