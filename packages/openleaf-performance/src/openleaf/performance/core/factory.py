# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2026 EUROCONTROL

"""
Model factory for unified aircraft model loading.

Provides a single entry point for loading aircraft performance models
from different sources (BADA3, BADA4, OpenAP, PS) using a consistent
interface.

Usage:
    # Load by ICAO code (auto-detects best available model)
    model = ModelFactory.load("A320")

    # Load specific model type
    model = ModelFactory.load("A320", model_type="bada4")

    # Load from file
    model = ModelFactory.load_from_file("/path/to/A320.xml", model_type="bada4")

    # List available aircraft by model type
    aircraft = ModelFactory.available_aircraft("bada4")
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from openleaf.performance.exceptions import OpenLeafError


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from openleaf.performance.core.perfos import AircraftModel

    # All concrete model classes accept (icao: str) as constructor,
    # unlike AircraftModel ABC which takes (data, engine, aero, limits).
    type ModelConstructor = Callable[[str], AircraftModel]

__all__ = ["ModelFactory", "ModelType"]

_logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported aircraft performance model types."""

    BADA3 = "bada3"
    BADA4 = "bada4"
    OPENAP = "openap"
    PS = "ps"


# Priority order for model type detection
_MODEL_PRIORITY = [ModelType.BADA4, ModelType.BADA3, ModelType.OPENAP, ModelType.PS]


class ModelFactory:
    """Unified factory for loading aircraft performance models.

    Provides a consistent interface for loading models from different
    sources without requiring knowledge of the specific model classes.

    Attributes:
        _model_classes: Mapping from ModelType to model class (lazy-loaded).
    """

    _model_classes: dict[ModelType, ModelConstructor] | None = None

    @classmethod
    def _get_model_classes(cls) -> dict[ModelType, ModelConstructor | None]:
        """Lazy-load model classes to avoid circular imports.

        Returns:
            Dictionary mapping ModelType to model class (or None if unavailable).
        """
        if cls._model_classes is None:
            # Import here to avoid circular imports
            # Models may fail to import if their data paths aren't configured
            cls._model_classes = {}

            try:
                from openleaf.performance.models.bada3 import BADA3Model

                cls._model_classes[ModelType.BADA3] = BADA3Model
            except (ImportError, OpenLeafError) as e:
                _logger.debug("BADA3 model not available: %s", e)
                cls._model_classes[ModelType.BADA3] = None

            try:
                from openleaf.performance.models.bada4 import BADA4Model

                cls._model_classes[ModelType.BADA4] = BADA4Model
            except (ImportError, OpenLeafError) as e:
                _logger.debug("BADA4 model not available: %s", e)
                cls._model_classes[ModelType.BADA4] = None

            try:
                from openleaf.performance.models.openap import OpenAPModel

                cls._model_classes[ModelType.OPENAP] = OpenAPModel
            except (ImportError, OpenLeafError) as e:
                _logger.debug("OpenAP model not available: %s", e)
                cls._model_classes[ModelType.OPENAP] = None

            try:
                from openleaf.performance.models.ps import PSModel

                cls._model_classes[ModelType.PS] = PSModel
            except (ImportError, OpenLeafError) as e:
                _logger.debug("PS model not available: %s", e)
                cls._model_classes[ModelType.PS] = None

        return cls._model_classes

    @staticmethod
    def _resolve_model_type(
        model_type: str | ModelType | None,
    ) -> ModelType | None:
        """Convert a string model type to ModelType enum with validation."""
        if not isinstance(model_type, str):
            return model_type
        try:
            return ModelType(model_type.lower())
        except ValueError as e:
            valid_types = [t.value for t in ModelType]
            raise ValueError(
                f"Invalid model_type '{model_type}'. Valid types are: {valid_types}"
            ) from e

    @classmethod
    def _auto_detect(
        cls,
        icao: str,
        model_classes: dict[ModelType, ModelConstructor | None],
    ) -> AircraftModel:
        """Try each model type in priority order, return first success."""
        errors: list[str] = []
        for mtype in _MODEL_PRIORITY:
            model_class = model_classes[mtype]
            if model_class is None:
                errors.append(f"{mtype.value}: Model not available (not configured)")
                continue
            try:
                return model_class(icao)
            except (KeyError, FileNotFoundError, ValueError, OpenLeafError) as e:
                errors.append(f"{mtype.value}: {e}")

        raise ValueError(
            f"Aircraft '{icao}' not found in any model. Tried:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )

    @classmethod
    def load(
        cls,
        icao: str,
        model_type: str | ModelType | None = None,
    ) -> AircraftModel:
        """Load an aircraft model by ICAO code.

        If model_type is not specified, tries to load models in priority order
        (BADA4 > BADA3 > OpenAP > PS) and returns the first successful load.

        Args:
            icao: ICAO aircraft type code (e.g., "A320", "B738").
            model_type: Specific model type to load, or None for auto-detect.
                Can be a string ("bada3", "bada4", "openap", "ps") or ModelType enum.

        Returns:
            Loaded aircraft model implementing AircraftModel interface.

        Raises:
            ValueError: If model_type is invalid or aircraft not found in any model.

        Example:
            >>> model = ModelFactory.load("A320")
            >>> model = ModelFactory.load("A320", model_type="bada4")
            >>> model = ModelFactory.load("A320", model_type=ModelType.BADA4)
        """
        model_classes = cls._get_model_classes()
        resolved = cls._resolve_model_type(model_type)

        if resolved is not None:
            model_class = model_classes[resolved]
            if model_class is None:
                raise ValueError(
                    f"Model type '{resolved.value}' is not available. "
                    f"Check that the required data path is configured."
                )
            return model_class(icao)

        return cls._auto_detect(icao, model_classes)

    @classmethod
    def load_from_file(
        cls,
        file_path: str | Path,
        model_type: str | ModelType,
    ) -> AircraftModel:
        """Load an aircraft model from a specific data file.

        Args:
            file_path: Path to the model data file.
            model_type: Model type to use for parsing the file.
                Required because file format determines the parser.

        Returns:
            Loaded aircraft model.

        Raises:
            ValueError: If model_type is invalid.
            FileNotFoundError: If file does not exist.

        Example:
            >>> model = ModelFactory.load_from_file("/path/to/A320.xml", "bada4")
            >>> model = ModelFactory.load_from_file("/path/to/A320.opf", "bada3")
        """
        model_classes = cls._get_model_classes()

        # Convert string to ModelType if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError as e:
                valid_types = [t.value for t in ModelType]
                raise ValueError(
                    f"Invalid model_type '{model_type}'. Valid types are: {valid_types}"
                ) from e

        model_class = model_classes[model_type]

        if model_class is None:
            raise ValueError(
                f"Model type '{model_type.value}' is not available. "
                f"Check that the required data path is configured."
            )

        # Use from_file class method if available
        if hasattr(model_class, "from_file"):
            return model_class.from_file(file_path)
        else:
            raise ValueError(
                f"Model type '{model_type.value}' does not support loading from file."
            )

    @classmethod
    def available_aircraft(
        cls,
        model_type: str | ModelType | None = None,
    ) -> dict[ModelType, list[str]] | list[str]:
        """List available aircraft for one or all model types.

        Args:
            model_type: Specific model type, or None for all types.

        Returns:
            If model_type specified: List of ICAO codes for that model type.
            If model_type is None: Dictionary mapping ModelType to list of ICAO codes.

        Example:
            >>> all_aircraft = ModelFactory.available_aircraft()
            >>> bada4_aircraft = ModelFactory.available_aircraft("bada4")
        """
        model_classes = cls._get_model_classes()

        # Convert string to ModelType if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError as e:
                valid_types = [t.value for t in ModelType]
                raise ValueError(
                    f"Invalid model_type '{model_type}'. Valid types are: {valid_types}"
                ) from e

        def _get_aircraft_list(mtype: ModelType) -> list[str]:
            """Get aircraft list for a model type, handling errors."""
            model_class = model_classes[mtype]
            if model_class is None:
                return []
            if hasattr(model_class, "available_aircraft"):
                try:
                    return model_class.available_aircraft()
                except (ImportError, OSError, OpenLeafError) as e:
                    # ImportError: optional backend library (openap/pycontrails)
                    # not installed — same contract as unconfigured BADA paths.
                    _logger.debug("Failed to get aircraft list for %s: %s", mtype, e)
                    return []
            return []

        if model_type is not None:
            return _get_aircraft_list(model_type)

        # Return all model types
        return {mtype: _get_aircraft_list(mtype) for mtype in ModelType}

    @classmethod
    def is_available(
        cls,
        icao: str,
        model_type: str | ModelType | None = None,
    ) -> bool | dict[ModelType, bool]:
        """Check if an aircraft is available in one or all model types.

        Args:
            icao: ICAO aircraft type code.
            model_type: Specific model type, or None for all types.

        Returns:
            If model_type specified: True if aircraft available in that model.
            If model_type is None: Dictionary mapping ModelType to availability.

        Example:
            >>> ModelFactory.is_available("A320")
            {ModelType.BADA3: True, ModelType.BADA4: True, ...}
            >>> ModelFactory.is_available("A320", "bada4")
            True
        """
        all_aircraft = cls.available_aircraft(model_type)

        if isinstance(all_aircraft, list):
            return icao in all_aircraft

        return {mtype: icao in aircraft for mtype, aircraft in all_aircraft.items()}
