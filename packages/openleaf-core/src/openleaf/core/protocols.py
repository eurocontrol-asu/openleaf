"""Structural Protocols for aircraft performance model contracts.

These Protocols describe what an aircraft model offers without prescribing
how it is built. Concrete ABCs with physics-aware defaults live in
``openleaf-performance``; this module only declares the surface that any
downstream consumer (trajectory optimisation, emissions, web APIs) can
program against.

Protocols are ``@runtime_checkable`` for the sub-protocols (Engine,
Aerodynamics, Limits). ``AircraftModelProtocol`` aggregates them and is
not runtime-checkable because Python's ``@runtime_checkable`` doesn't
support multi-Protocol inheritance — use ``isinstance(model, sub_proto)``
on each individual sub-protocol if you need a runtime check.
"""

from __future__ import annotations


__all__ = [
    "AerodynamicsModelProtocol",
    "AircraftModelProtocol",
    "EngineModelProtocol",
    "LimitsModelProtocol",
]

from typing import Protocol
from typing import runtime_checkable

from openleaf.core.types import ArrayScalarLike
from openleaf.core.types import KeyedData


@runtime_checkable
class EngineModelProtocol(Protocol):
    """Protocol for engine models."""

    def thrust_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum available thrust [N]."""
        ...

    def thrust_max(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate maximum available thrust [N]."""
        ...

    def fuel_flow(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate fuel flow rate [kg/s]."""
        ...

    def fuel_flow_min(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate minimum fuel flow rate [kg/s]."""
        ...


@runtime_checkable
class AerodynamicsModelProtocol(Protocol):
    """Protocol for aerodynamic models."""

    @property
    def surface_area(self) -> float:
        """Wing reference area [m²]."""
        ...

    def drag_coef(self, data: KeyedData) -> ArrayScalarLike:
        """Calculate drag coefficient [-]."""
        ...


@runtime_checkable
class LimitsModelProtocol(Protocol):
    """Protocol for aircraft limits."""

    @property
    def mtom(self) -> float:
        """Maximum takeoff mass [kg]."""
        ...

    @property
    def oem(self) -> float:
        """Operating empty mass [kg]."""
        ...

    @property
    def mpl(self) -> float:
        """Maximum payload [kg]."""
        ...

    @property
    def vmo(self) -> float:
        """Maximum operating calibrated airspeed [kt]."""
        ...

    @property
    def mmo(self) -> float:
        """Maximum operating Mach number."""
        ...

    @property
    def hmo(self) -> float:
        """Maximum operating altitude [ft]."""
        ...

    def min_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get state-dependent minimum calibrated airspeed [kt]."""
        ...

    def min_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get state-dependent minimum Mach number."""
        ...

    def max_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get state-dependent maximum calibrated airspeed [kt]."""
        ...

    def max_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get state-dependent maximum Mach number."""
        ...

    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:
        """Get state-dependent maximum operating altitude [ft]."""
        ...


# NOTE: @runtime_checkable is not supported on multi-Protocol inheritance.
# Use isinstance() on individual sub-protocols if you need a runtime check.
# Explicit Protocol base is needed so mypy recognises the empty bodies as
# Protocol method stubs rather than missing implementations.
class AircraftModelProtocol(
    EngineModelProtocol,
    AerodynamicsModelProtocol,
    LimitsModelProtocol,
    Protocol,
):
    """Protocol for complete aircraft performance models."""

    @property
    def model(self) -> str:
        """Aircraft model name."""
        ...

    @property
    def wtc(self) -> str:
        """Wake turbulence category."""
        ...

    @property
    def engine_type(self) -> str:
        """Engine type."""
        ...

    def high_lift(self, data: KeyedData) -> ArrayScalarLike | None:
        """Determine high-lift device setting from flight state."""
        ...

    def landing_gear(self, data: KeyedData) -> ArrayScalarLike:
        """Determine landing gear state from flight state."""
        ...

    def aerodynamic_configuration(self, data: KeyedData) -> ArrayScalarLike:
        """Determine aerodynamic configuration from flight state."""
        ...
