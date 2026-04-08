"""
BADA-specific type definitions.

This module re-exports the unified AerodynamicConfig from core.types
for backwards compatibility. New code should import from openleaf.performance.core.types.

The AerodynamicConfig IntEnum represents the five aerodynamic configurations
defined in the BADA manuals (Section 3.5 for BADA3, Section 6 for BADA4).
"""

from __future__ import annotations

# Re-export from unified types module
from openleaf.performance.core.types import AerodynamicConfig


__all__ = ["AerodynamicConfig"]
