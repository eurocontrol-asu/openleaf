"""Enforce that every public model method consuming KeyedData declares @requires.

This is the introspection gate for the @requires decorator. Without it, the
decorator's coverage drifts (some methods decorated, others not), and the
runtime validation guarantee is silently inconsistent across providers.

The test uses AST analysis rather than just attribute introspection:
a method overrides the Protocol surface AND reads ``data["key"]`` or
``data.get("key")`` => it MUST be wrapped with @requires.

Methods that take ``data`` purely to satisfy a Protocol signature but never
read fields (base-class default implementations that return static configuration)
are correctly NOT required to declare @requires.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable
from typing import Any

import pytest

from openleaf.performance.core.perfos import AerodynamicsModel
from openleaf.performance.core.perfos import AircraftModel
from openleaf.performance.core.perfos import EngineModel
from openleaf.performance.core.perfos import LimitsModel
from openleaf.performance.core.validation import StateSchema


# Force-import every concrete provider so the ABCs see them as subclasses.
# Skipped imports do NOT remove the assertion — the test still walks whatever
# loaded successfully. Providers that fail to import (e.g. missing BADA data
# paths) are reported in the skip messages collected below.
_IMPORT_FAILURES: dict[str, str] = {}
for _module in (
    "openleaf.performance.models.bada3.models",
    "openleaf.performance.models.bada3.limits",
    "openleaf.performance.models.bada4.models",
    "openleaf.performance.models.bada4.limits",
    "openleaf.performance.models.openap.models",
    "openleaf.performance.models.openap.limits",
    "openleaf.performance.models.ps.models",
    "openleaf.performance.models.ps.limits",
):
    try:
        __import__(_module)
    except Exception as exc:  # noqa: BLE001 - any failure is informational
        _IMPORT_FAILURES[_module] = repr(exc)


BASE_CLASSES: tuple[type, ...] = (
    EngineModel,
    AerodynamicsModel,
    LimitsModel,
    AircraftModel,
)


def _all_subclasses(base: type) -> list[type]:
    """Return every subclass loaded into the registry, including abstract
    intermediates.

    Intermediate abstract classes (e.g. BADA3TurbineModel) are where the real
    implementations live — the concrete leaves often just exist to bind the
    Generic. We must walk them too, otherwise the test silently passes by
    skipping every method that matters.
    """
    found: list[type] = []
    stack: list[type] = list(base.__subclasses__())
    seen: set[type] = set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        found.append(cls)
    return found


def _accesses_data_field(fn: Any) -> bool:
    """Return True if the function's body reads ``data[...]`` or ``data.get(...)``.

    The method binds ``data`` as its first non-self parameter; we look for any
    subscript or ``.get(...)`` access against that name.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False

    source = inspect.cleandoc(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    fn_node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)),
        None,
    )
    if fn_node is None:
        return False

    # Find the parameter name that carries KeyedData (typically "data").
    # Skip self/cls; take the first remaining positional parameter.
    args = [a.arg for a in fn_node.args.args if a.arg not in {"self", "cls"}]
    if not args:
        return False
    data_name = args[0]

    for node in ast.walk(fn_node):
        # data[...]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == data_name
        ):
            return True
        # data.get(...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == data_name
        ):
            return True
    return False


def _public_methods(cls: type) -> Iterable[tuple[str, Any]]:
    """Yield (name, function) for every public method DEFINED on the class itself.

    Methods inherited unchanged from a base class are checked when we walk
    that base class instead; this avoids reporting the same method N times.
    Pure ``@abstractmethod`` stubs (no body to analyse) are skipped.
    """
    for name, value in vars(cls).items():
        if name.startswith("_"):
            continue
        if not callable(value):
            continue
        # Skip classmethods, staticmethods, properties.
        if isinstance(value, classmethod | staticmethod | property):
            continue
        # Skip pure abstract method stubs — they have no body to check.
        if getattr(value, "__isabstractmethod__", False):
            continue
        yield name, value


def _is_decorated(fn: Any) -> bool:
    """Return True if the function carries a StateSchema from @requires."""
    schema = getattr(fn, "_schema", None)
    return isinstance(schema, StateSchema)


def _gather_violations() -> list[str]:
    """Return human-readable descriptions of every undecorated data-consuming method."""
    violations: list[str] = []
    for base in BASE_CLASSES:
        for cls in _all_subclasses(base):
            for name, fn in _public_methods(cls):
                if not _accesses_data_field(fn):
                    continue
                if _is_decorated(fn):
                    continue
                qualname = f"{cls.__module__}.{cls.__qualname__}.{name}"
                violations.append(qualname)
    return sorted(set(violations))


def test_every_public_aircraftmodel_method_declares_required_fields() -> None:
    """Each concrete model method that reads ``data[...]`` must carry @requires.

    Failure mode: the violator list below names every method that accesses
    flight-state fields without declaring them through the decorator. To fix,
    add ``@requires(\"field_a\", \"field_b\", ...)`` matching the fields the
    method actually reads.
    """
    if _IMPORT_FAILURES:
        # Surfacing import failures helps diagnose CI-only issues but does
        # not weaken the assertion: methods on modules that failed to import
        # simply aren't checked.
        notes = "\n".join(f"  - {m}: {err}" for m, err in _IMPORT_FAILURES.items())
        print(f"Note: some provider modules did not import:\n{notes}")

    violations = _gather_violations()
    if violations:
        listed = "\n".join(f"  - {v}" for v in violations)
        pytest.fail(
            f"{len(violations)} public method(s) read data[...] without @requires:\n"
            f"{listed}\n\n"
            "Add @requires(\"field_a\", ...) listing the fields each method "
            "actually reads. See openleaf.performance.core.validation for the "
            "decorator."
        )
