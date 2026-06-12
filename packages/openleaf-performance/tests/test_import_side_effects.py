"""Importing openleaf must not reconfigure the host application's logging.

Run in a subprocess so the assertions see a pristine interpreter, not state
left behind by other tests.
"""

from __future__ import annotations

import subprocess
import sys


_SCRIPT = """
import logging
import structlog

root_handlers_before = list(logging.getLogger().handlers)

import openleaf.performance  # noqa: F401
import openleaf.performance.config  # module-level get_logger() call

assert not structlog.is_configured(), (
    "importing openleaf must not call structlog.configure()"
)
assert logging.getLogger().handlers == root_handlers_before, (
    "importing openleaf must not add handlers to the root logger"
)

# The library-scoped fallback logger still works and filters at WARNING.
from openleaf.performance.logging import get_logger, disable_logging

log = get_logger("openleaf.test")
log.debug("must be dropped silently")
disable_logging()
log.warning("must be dropped after disable_logging")
print("NO-SIDE-EFFECTS-OK")
"""


def test_import_does_not_touch_global_logging():
    result = subprocess.run(  # noqa: S603 — fixed argv, runs our own interpreter
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "NO-SIDE-EFFECTS-OK" in result.stdout
    # Nothing else may have been printed by the dropped log calls.
    assert "must be dropped" not in result.stdout
    assert "must be dropped" not in result.stderr
