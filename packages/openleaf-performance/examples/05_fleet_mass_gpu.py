#!/usr/bin/env python3
"""
Fleet-Scale Mass Estimation on GPU via OpenLeaf's PyTorch Backend.

Estimates takeoff mass for N flights simultaneously using gradient descent
through the TEM forward simulation on PyTorch tensors.

The formulation mirrors example 01 (CasADi):
    - IC: h[0] = h_obs[0], V[0] = V_obs[0], m[0] = m0 (unknown)
    - Decision variables: m0 and ESF[t] per flight
    - Forward-Euler TEM integration: (h, V, m) propagated step by step
    - Objective: ||h_sim - h_obs||² + ||V_sim - V_obs||²
    - OpenLeaf's thrust_max(), drag_coef(), fuel_flow() run on batched tensors

The key difference: CasADi + IPOPT solves one flight at a time.
PyTorch stacks the fleet as a batch dimension and backpropagates
through the entire forward simulation for all flights simultaneously.

Usage:
    uv run python examples/05_fleet_mass_gpu.py
    uv run python examples/05_fleet_mass_gpu.py --fleet-size 1000
    uv run python examples/05_fleet_mass_gpu.py --fleet-size 10000 --device cuda
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import cyclopts
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from openleaf.performance.physics import isa
from openleaf.performance.physics import units
from openleaf.performance.physics.aero import aerodynamic_coefficient
from openleaf.performance.physics.aero import dynamic_pressure


console = Console()

G = isa.G
GAMMA_R = isa.GAMMA * isa.R


def load_climb_data(flight_data_path: Path) -> dict[str, np.ndarray]:
    """Extract climb phase from a single flight."""
    with flight_data_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    alt_ft = np.array(raw["altitude_ft"], dtype=float)
    vs_fpm = np.array(raw["vertical_speed"], dtype=float)
    v_tas = np.array(raw["true_airspeed"], dtype=float)
    t_sec = np.array(raw["time"], dtype=float)

    alt_m = alt_ft * units.FT
    vs_ms = vs_fpm * units.FPM

    valid = np.isfinite(alt_m) & np.isfinite(vs_ms) & np.isfinite(v_tas)
    climb = valid & (vs_ms > 0.5) & (alt_ft > 10000) & (alt_ft < 35000)
    idx = np.where(climb)[0]

    dt = np.diff(t_sec[idx])

    return {
        "altitude": alt_m[idx],
        "true_airspeed": v_tas[idx],
        "dt": dt,
        "n_points": len(idx),
    }


def tem_forward(
    model,
    m0: torch.Tensor,
    esf: torch.Tensor,
    h_obs: torch.Tensor,
    v_obs: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-Euler TEM integration for a batch of flights.

    Args:
        model: OpenLeaf performance model.
        m0: initial mass per flight [F]
        esf: energy share factor [F, N-1]
        h_obs: observed altitudes [F, N] (IC from h_obs[:, 0])
        v_obs: observed TAS [F, N] (IC from v_obs[:, 0])
        dt: timesteps [N-1]

    Returns:
        h_sim [F, N], v_sim [F, N] — simulated trajectories
    """
    F, N = h_obs.shape
    device = m0.device
    dtype = m0.dtype

    h_sim = [h_obs[:, 0]]  # IC: h[0] = h_obs[0]
    v_sim = [v_obs[:, 0]]  # IC: V[0] = V_obs[0]

    h_k = h_obs[:, 0]      # [F]
    v_k = v_obs[:, 0]       # [F]
    m_k = m0                 # [F]

    g = torch.tensor(G, dtype=dtype, device=device)
    gamma_r = torch.tensor(GAMMA_R, dtype=dtype, device=device)

    S = model.aerodynamics.surface_area
    N_SUB = 4  # sub-steps per interval

    for k in range(N - 1):
        f_k = esf[:, k]
        dt_sub = dt[k] / N_SUB

        for _s in range(N_SUB):
            # Atmosphere — OpenLeaf, PyTorch backend
            T_air = isa.air_temperature(h_k)
            p_air = isa.air_pressure(h_k)
            rho = p_air / (isa.R * T_air)
            mach = v_k / torch.sqrt(gamma_r * T_air)

            q_dyn = dynamic_pressure(v_k, rho)
            cl = aerodynamic_coefficient(m_k * g, q_dyn, S)

            perf_data = {
                "altitude": h_k,
                "true_airspeed": v_k,
                "aircraft_mass": m_k,
                "air_temperature": T_air,
                "air_pressure": p_air,
                "air_temperature_ratio": isa.theta(h_k),
                "air_pressure_ratio": isa.delta(h_k),
                "air_density": rho,
                "mach_number": mach,
                "lift_coef": cl,
                "vertical_speed": torch.zeros_like(h_k),
                "f": f_k,
            }

            # OpenLeaf model calls — all batched over fleet
            T_max = model.thrust_max(perf_data)
            cd = model.drag_coef(perf_data)
            drag = cd * q_dyn * S
            ff = model.fuel_flow({**perf_data, "thrust": T_max})

            # TEM: Ps [W/kg] = (T-D)*V/m
            Ps = (T_max - drag) * v_k / m_k

            # Pressure altitude rate = f * Ps/g * T_ISA/T
            T_isa = isa.air_temperature(h_k)
            dh = f_k * (Ps / g) * (T_isa / T_air) * dt_sub
            dV = (1.0 - f_k) * (Ps / v_k) * dt_sub
            dm = ff * dt_sub

            h_k = h_k + dh
            v_k = v_k + dV
            m_k = m_k - dm

        h_next = h_k
        v_next = v_k
        m_next = m_k

        h_sim.append(h_next)
        v_sim.append(v_next)

        h_k = h_next
        v_k = v_next
        m_k = m_next

    return torch.stack(h_sim, dim=1), torch.stack(v_sim, dim=1)


def fleet_mass_estimation(
    model,
    climb: dict[str, np.ndarray],
    fleet_size: int,
    device: torch.device,
    epochs: int = 300,
    lr: float = 100.0,
) -> tuple[torch.Tensor, list[float], float]:
    """Estimate mass for N flights simultaneously via gradient descent."""
    n = climb["n_points"]
    dtype = torch.float64

    # Observed trajectory — replicate for fleet [fleet_size, n_points]
    h_obs = torch.tensor(climb["altitude"], dtype=dtype, device=device).unsqueeze(0).expand(fleet_size, -1)
    v_obs = torch.tensor(climb["true_airspeed"], dtype=dtype, device=device).unsqueeze(0).expand(fleet_size, -1)
    dt = torch.tensor(climb["dt"], dtype=dtype, device=device)

    # Decision variables: use unconstrained parameterisation
    # m0_raw → m0 via sigmoid scaled to [OEM, MTOW]
    # esf_raw → esf via sigmoid scaled to [0, 1.5]
    OEM, MTOW = 41000.0, 79000.0

    # Inverse-sigmoid for initial guesses: m0≈70000 → raw≈1.54, esf≈0.9 → raw≈0.89
    def inv_sigmoid(x, lo, hi):
        t = (x - lo) / (hi - lo)
        return torch.log(t / (1 - t))

    m0_raw = torch.nn.Parameter(
        inv_sigmoid(torch.full((fleet_size,), 70000.0, dtype=dtype, device=device), OEM, MTOW)
    )
    esf_raw = torch.nn.Parameter(
        inv_sigmoid(torch.full((fleet_size, n - 1), 0.9, dtype=dtype, device=device), 0.0, 1.5)
    )

    optimiser = torch.optim.Adam([m0_raw, esf_raw], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    losses = []

    t0 = time.perf_counter()

    for epoch in range(epochs):
        optimiser.zero_grad()

        # Constrained variables via sigmoid
        m0 = OEM + (MTOW - OEM) * torch.sigmoid(m0_raw)
        esf = 1.5 * torch.sigmoid(esf_raw)

        # Forward TEM simulation — entire fleet in one pass
        h_sim, v_sim = tem_forward(model, m0, esf, h_obs, v_obs, dt)

        # Objective: specific energy error (same as CasADi formulation)
        Ep_err = G * (h_sim - h_obs)          # potential energy error [J/kg]
        Ek_err = 0.5 * (v_sim**2 - v_obs**2)  # kinetic energy error [J/kg]
        loss = (Ep_err**2).mean() + (Ek_err**2).mean()
        loss.backward()
        optimiser.step()
        scheduler.step()

        losses.append(loss.item())
        if epoch % 50 == 0:
            m0_val = (OEM + (MTOW - OEM) * torch.sigmoid(m0_raw.data)).mean().item()
            esf_val = (1.5 * torch.sigmoid(esf_raw.data)).mean().item()
            console.print(
                f"  epoch {epoch:4d}  loss={loss.item():.2e}  "
                f"m0={m0_val:.0f} kg  "
                f"esf_mean={esf_val:.3f}"
            )

    elapsed = time.perf_counter() - t0
    m0_final = OEM + (MTOW - OEM) * torch.sigmoid(m0_raw.data)
    return m0_final, losses, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = cyclopts.App()


@app.default
def main(
    model_type: Annotated[str, cyclopts.Parameter(name="--model")] = "ps",
    data: Annotated[Path, cyclopts.Parameter(name="--data")] = Path("data/flight_data.json"),
    fleet_size: Annotated[int, cyclopts.Parameter(name="--fleet-size")] = 100,
    epochs: Annotated[int, cyclopts.Parameter(name="--epochs")] = 300,
    device_str: Annotated[str, cyclopts.Parameter(name="--device")] = "cpu",
) -> None:
    """Estimate mass for a fleet of flights in parallel."""
    device = torch.device(device_str)
    console.print(f"Device: [bold]{device}[/bold]")
    console.print(f"Fleet size: [bold]{fleet_size}[/bold] flights")

    if model_type.lower() == "ps":
        from openleaf.performance.models.ps import PSModel
        perf_model = PSModel("B738")
    else:
        msg = f"Unsupported: {model_type}. Use ps."
        raise ValueError(msg)

    climb = load_climb_data(data)
    console.print(f"Climb points per flight: [bold]{climb['n_points']}[/bold]")

    console.print(f"\n[bold]Estimating mass for {fleet_size} flights…[/bold]")
    m0_est, losses, elapsed = fleet_mass_estimation(
        perf_model, climb, fleet_size, device, epochs=epochs,
    )

    # Compare with CasADi result from example 01
    casadi_time = 1.2  # seconds per flight (from paper)
    casadi_equiv = fleet_size * casadi_time

    table = Table(title="Fleet Mass Estimation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Fleet size", f"{fleet_size}")
    table.add_row("Total time", f"{elapsed:.1f} s")
    table.add_row("Per-flight time", f"{elapsed/fleet_size*1000:.1f} ms")
    table.add_row("CasADi equivalent", f"{casadi_equiv:.0f} s ({casadi_equiv/60:.1f} min)")
    table.add_row("Speedup", f"{casadi_equiv/elapsed:.0f}×")
    table.add_row("m₀ estimate", f"{m0_est.mean().item():.0f} kg")
    table.add_row("Final loss", f"{losses[-1]:.2e}")
    console.print(table)


if __name__ == "__main__":
    app()
