#!/usr/bin/env python3
"""
Parametric Cruise Simulation via JAX JIT + vmap.

Demonstrates that OpenLeaf's JAX backend enables JIT-compiled, vectorised
cruise simulations.  After the first call (JIT compilation), subsequent
evaluations run in microseconds — enabling parametric sweeps over
(mass, range, CI) at negligible cost.

Usage:
    uv run python examples/06_jax_parametric.py
    uv run python examples/06_jax_parametric.py --grid-size 1000
"""

from __future__ import annotations

import time
from typing import Annotated

import cyclopts
import jax
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from openleaf.performance.models.ps import PSModel
from openleaf.performance.physics import isa


console = Console()

model = PSModel("A320")
S = model.aerodynamics.surface_area
G = isa.G
R = isa.R
GAMMA_R = isa.GAMMA * isa.R


def simulate_cruise(mass, distance_km, ci):
    """Simulate a fixed-altitude cruise leg via forward Euler.

    Returns (fuel_burn [kg], flight_time [min]).
    """
    h = 10668.0   # FL350
    v = 230.0     # ~M0.78 at FL350
    m = mass
    total_dist = distance_km * 1000.0
    N_steps = 100
    ds = total_dist / N_steps

    flight_time = 0.0

    for _ in range(N_steps):
        T_air = isa.air_temperature(h)
        p_air = isa.air_pressure(h)
        rho_val = p_air / (R * T_air)
        mach_val = v / jnp.sqrt(GAMMA_R * T_air)
        q_dyn = 0.5 * rho_val * v**2
        cl = (m * G) / (q_dyn * S)

        data = {
            "altitude": h,
            "true_airspeed": v,
            "aircraft_mass": m,
            "air_temperature": T_air,
            "air_pressure": p_air,
            "air_temperature_ratio": isa.theta(h),
            "air_pressure_ratio": isa.delta(h),
            "air_density": rho_val,
            "mach_number": mach_val,
            "lift_coef": cl,
            "vertical_speed": 0.0,
            "f": 1.0,
        }
        cd = model.drag_coef(data)
        drag = cd * q_dyn * S
        ff = model.fuel_flow({**data, "thrust": drag})

        gs = v
        dt = ds / gs
        m = m - ff * dt
        flight_time = flight_time + dt

    fuel_burn = mass - m
    return fuel_burn, flight_time / 60.0


app = cyclopts.App()


@app.default
def main(
    grid_size: Annotated[int, cyclopts.Parameter(name="--grid-size")] = 100,
) -> None:
    """Run parametric cruise sweep with JIT + vmap."""

    # --- Single trajectory ---
    console.print("[bold]Single trajectory (eager)[/bold]")
    t0 = time.perf_counter()
    fb, ft = simulate_cruise(70000.0, 1500.0, 30.0)
    t_eager = time.perf_counter() - t0
    console.print(f"  fuel={float(fb):.1f} kg, time={float(ft):.1f} min, elapsed={t_eager*1000:.1f} ms")

    # --- JIT single ---
    console.print("\n[bold]JIT compilation (first call)[/bold]")
    sim_jit = jax.jit(simulate_cruise)
    t0 = time.perf_counter()
    fb, ft = sim_jit(70000.0, 1500.0, 30.0)
    t_compile = time.perf_counter() - t0
    console.print(f"  fuel={float(fb):.1f} kg, time={float(ft):.1f} min, compile={t_compile:.3f} s")

    console.print("\n[bold]JIT subsequent calls (1000×)[/bold]")
    t0 = time.perf_counter()
    for _ in range(1000):
        sim_jit(70000.0, 1500.0, 30.0)
    t_jit = (time.perf_counter() - t0) / 1000
    console.print(f"  {t_jit*1000:.4f} ms/call")

    # --- vmap parametric sweep ---
    N = grid_size
    console.print(f"\n[bold]vmap parametric sweep ({N} trajectories)[/bold]")
    sim_batch = jax.jit(jax.vmap(simulate_cruise))
    masses = jnp.linspace(55000.0, 79000.0, N)
    ranges = jnp.linspace(500.0, 3000.0, N)
    cis = jnp.linspace(0.0, 80.0, N)

    t0 = time.perf_counter()
    fb_batch, ft_batch = sim_batch(masses, ranges, cis)
    t_vmap_compile = time.perf_counter() - t0
    console.print(f"  Compile: {t_vmap_compile:.3f} s")

    t0 = time.perf_counter()
    for _ in range(100):
        fb_batch, ft_batch = sim_batch(masses, ranges, cis)
    t_vmap = (time.perf_counter() - t0) / 100
    console.print(f"  Subsequent: {t_vmap*1000:.3f} ms/call ({N} trajectories)")

    # --- Summary ---
    table = Table(title="JAX JIT Benchmark")
    table.add_column("Method")
    table.add_column("Time")
    table.add_column("Speedup vs eager")
    table.add_row("Eager (1 traj)", f"{t_eager*1000:.1f} ms", "1×")
    table.add_row("JIT first call", f"{t_compile:.1f} s", "(compilation)")
    table.add_row("JIT subsequent", f"{t_jit*1000:.4f} ms", f"{t_eager/t_jit:.0f}×")
    table.add_row(f"vmap ({N} traj)", f"{t_vmap*1000:.3f} ms", f"{t_eager*N/t_vmap:.0f}×")
    console.print(table)

    console.print(f"\nFuel range: {float(fb_batch.min()):.0f} – {float(fb_batch.max()):.0f} kg")
    console.print(f"Time range: {float(ft_batch.min()):.0f} – {float(ft_batch.max()):.0f} min")


if __name__ == "__main__":
    app()
