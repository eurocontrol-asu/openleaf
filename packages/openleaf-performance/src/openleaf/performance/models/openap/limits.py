"""OpenAP aircraft limitations model.

Provides the OpenAPLimits class implementing constant operational limits
(VMO, MMO, ceiling) from OpenAP aircraft database.

Reference: OpenAP (https://github.com/junzis/openap)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openleaf.performance.core.perfos import LimitsModel
from openleaf.performance.models.openap.schemas import OpenAPLimitsData


if TYPE_CHECKING:
    from openleaf.performance.utils.types import ArrayScalarLike
    from openleaf.performance.utils.types import KeyedData

__all__ = ["OpenAPLimits"]


class OpenAPLimits(LimitsModel[OpenAPLimitsData]):
    """OpenAP aircraft operational limits.

    OpenAP provides constant limits (VMO, MMO, ceiling) without
    configuration-dependent adjustments.
    """

    def __init__(self, data: OpenAPLimitsData) -> None:
        """Initialize OpenAP limits.

        Args:
            data: OpenAP limits data container.
        """
        super().__init__(data)

    def max_mach(self, data: KeyedData) -> ArrayScalarLike:
        """Get maximum Mach number (constant MMO).

        Args:
            data: Flight state (unused - OpenAP uses constant MMO).

        Returns:
            Maximum operating Mach number.
        """
        return self.data.mmo

    def max_altitude(self, data: KeyedData) -> ArrayScalarLike:
        """Get service ceiling.

        Args:
            data: Flight state (unused - OpenAP uses constant ceiling).

        Returns:
            Service ceiling [ft].
        """
        return self.data.hmo

    def max_cas(self, data: KeyedData) -> ArrayScalarLike:
        """Get VMO (maximum operating CAS).

        Args:
            data: Flight state (unused - OpenAP uses constant VMO).

        Returns:
            VMO [kt].
        """
        return self.data.vmo
