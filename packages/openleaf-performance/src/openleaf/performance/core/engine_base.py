"""Skeleton base class for the Phase-4 engine-model unification.

Background
----------

Today every provider (BADA3, BADA4, OpenAP, PS) implements
``thrust_max`` / ``thrust_min`` / ``fuel_flow`` / ``fuel_flow_min`` from
scratch. The implementations have substantial structural duplication —
each unpacks fields from ``KeyedData``, runs provider-specific math,
and returns ``ArrayScalarLike``. That duplication is what Phase 4 of
the audit plan removes.

This module ships the *skeleton* of the future ``BaseEngineModel`` so
the migration infrastructure (characterization tests, generic typing)
is in place. **No provider migration happens here.** The migrations
themselves require running BADA-licensed data through golden
fixtures, which is maintainer-only work.

The shape
---------

``BaseEngineModel[CoeffSetT]`` is generic over a *coefficient set*. A
coefficient set is whatever data structure carries the per-aircraft
constants the provider needs (polynomial coefficients, idle/max
ratios, tabulated reference points, …). Each concrete provider binds
the generic to its own coefficient type.

The public methods (``thrust_max`` etc.) are abstract because the
field-extraction contract differs by provider — some need
``mach + altitude + delta_isa``, others need ``true_airspeed +
altitude + vertical_speed``, and so on. The plan calls for each
migration to add a ``@requires(...)`` decorator at the concrete class
level. Forcing that decoration is the job of the validation-enforcement
test in ``tests/unit/core/test_validation_enforcement.py`` — adding
``@requires`` is not optional once a method reads ``data[...]``.

Why no shared implementation yet
--------------------------------

The right abstraction emerges from the second migration, not the
first. Committing a non-trivial shared implementation now risks
designing for one provider's needs and forcing the other three to
fit. The skeleton therefore only commits to:

1. A generic type variable ``CoeffSetT`` so subclasses get typed
   coefficient access.
2. A constructor that stores both the engine data (inherited from
   ``EngineModel``) and the coefficient set.
3. A ``coefficients`` property as the single agreed-upon vocabulary
   for accessing provider-specific constants from shared helpers.

The Phase-4 PR that migrates BADA4 first will introduce shared
helpers in this module (polynomial evaluation, idle/max threshold
logic) once the cross-provider commonality is concrete.
"""

from __future__ import annotations


__all__ = ["BaseEngineModel", "CoeffSetT"]

from typing import Generic
from typing import TypeVar

from openleaf.performance.core.perfos import EngineDataT
from openleaf.performance.core.perfos import EngineModel


CoeffSetT = TypeVar("CoeffSetT")
"""Type variable for the provider-specific coefficient set.

Bound to whatever shape a provider uses to carry its constants —
typically a dataclass or pydantic model — so subclasses get typed
coefficient access without ``Any``-typing escape hatches.
"""


class BaseEngineModel(EngineModel[EngineDataT], Generic[EngineDataT, CoeffSetT]):
    """Skeleton for the Phase-4 unified engine model.

    Subclasses MUST:

    1. Bind both type variables, e.g.
       ``class BADA4Engine(BaseEngineModel[BADA4EngineData, BADA4Coeffs]): ...``
    2. Pass the coefficient set to ``__init__`` alongside the engine
       data so the base class can expose it via ``coefficients``.
    3. Implement ``thrust_max`` / ``thrust_min`` / ``fuel_flow`` /
       ``fuel_flow_min`` (inherited as ``@abstractmethod`` from
       ``EngineModel``), each decorated with ``@requires(...)`` for
       the fields the implementation reads.

    Subclasses MAY rely on (future) shared helpers exposed on this
    base — polynomial evaluation, idle/max selectors — once the
    Phase-4 migrations identify what those helpers should look like.

    Until then this class is *infrastructure*: it gives Phase 4
    something concrete to migrate towards without prematurely
    locking in an abstraction.
    """

    def __init__(self, data: EngineDataT, coefficients: CoeffSetT) -> None:
        """Initialise with both the engine data and the coefficient set.

        Args:
            data: Engine-data container the parent class expects.
            coefficients: Provider-specific constants used by the
                engine math. Stored as-is; how it is used is up to
                the concrete provider.
        """
        super().__init__(data)
        self._coefficients = coefficients

    @property
    def coefficients(self) -> CoeffSetT:
        """The provider's coefficient set, exactly as supplied at construction."""
        return self._coefficients
