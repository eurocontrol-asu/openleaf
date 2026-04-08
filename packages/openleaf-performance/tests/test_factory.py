"""Tests for model factory."""

from __future__ import annotations

import pytest

from openleaf.performance.core.factory import ModelFactory
from openleaf.performance.core.factory import ModelType


class TestModelType:
    """Test ModelType enum."""

    def test_model_type_values(self):
        """Test ModelType enum has expected values."""
        assert ModelType.BADA3.value == "bada3"
        assert ModelType.BADA4.value == "bada4"
        assert ModelType.OPENAP.value == "openap"
        assert ModelType.PS.value == "ps"

    def test_model_type_from_string(self):
        """Test ModelType can be created from string."""
        assert ModelType("bada3") == ModelType.BADA3
        assert ModelType("bada4") == ModelType.BADA4
        assert ModelType("openap") == ModelType.OPENAP
        assert ModelType("ps") == ModelType.PS


class TestModelFactoryLoad:
    """Test ModelFactory.load method."""

    def test_load_with_string_model_type(self):
        """Test loading with string model type."""
        model = ModelFactory.load("A320", model_type="openap")
        assert model.model == "A320"

    def test_load_with_enum_model_type(self):
        """Test loading with ModelType enum."""
        model = ModelFactory.load("A320", model_type=ModelType.OPENAP)
        assert model.model == "A320"

    def test_load_invalid_model_type_raises(self):
        """Test loading with invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            ModelFactory.load("A320", model_type="invalid_type")

    def test_load_auto_detect(self):
        """Test auto-detection of model type."""
        # This should succeed with at least one model type
        model = ModelFactory.load("A320")
        assert model is not None
        # Model name varies by source (e.g., "A320", "A320-212", "a320")
        assert "320" in model.model or "320" in model.model.upper()

    def test_load_nonexistent_aircraft_raises(self):
        """Test loading non-existent aircraft raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            ModelFactory.load("ZZZZ_NONEXISTENT")


class TestModelFactoryLoadFromFile:
    """Test ModelFactory.load_from_file method."""

    def test_load_from_file_invalid_type_raises(self):
        """Test loading with invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            ModelFactory.load_from_file("/some/path.xml", model_type="invalid")


class TestModelFactoryAvailableAircraft:
    """Test ModelFactory.available_aircraft method."""

    def test_available_aircraft_all_types(self):
        """Test getting available aircraft for all model types."""
        result = ModelFactory.available_aircraft()
        assert isinstance(result, dict)
        assert ModelType.OPENAP in result
        assert ModelType.PS in result
        assert isinstance(result[ModelType.OPENAP], list)

    def test_available_aircraft_specific_type_string(self):
        """Test getting available aircraft for specific type (string)."""
        result = ModelFactory.available_aircraft("openap")
        assert isinstance(result, list)
        # OpenAP uses lowercase codes
        assert "a320" in result or "A320" in result

    def test_available_aircraft_specific_type_enum(self):
        """Test getting available aircraft for specific type (enum)."""
        result = ModelFactory.available_aircraft(ModelType.OPENAP)
        assert isinstance(result, list)
        # OpenAP uses lowercase codes
        assert "a320" in result or "A320" in result

    def test_available_aircraft_invalid_type_raises(self):
        """Test invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            ModelFactory.available_aircraft("invalid")


class TestModelFactoryIsAvailable:
    """Test ModelFactory.is_available method."""

    def test_is_available_all_types(self):
        """Test checking availability for all model types."""
        # Different models use different case conventions
        # PS uses "A320" (uppercase), OpenAP uses "a320" (lowercase)
        result_upper = ModelFactory.is_available("A320")
        result_lower = ModelFactory.is_available("a320")
        assert isinstance(result_upper, dict)
        assert result_upper[ModelType.PS] is True
        assert result_lower[ModelType.OPENAP] is True

    def test_is_available_specific_type(self):
        """Test checking availability for specific type."""
        # Use lowercase since OpenAP uses lowercase codes
        result = ModelFactory.is_available("a320", model_type="openap")
        assert result is True

    def test_is_available_nonexistent_aircraft(self):
        """Test checking availability for non-existent aircraft."""
        result = ModelFactory.is_available("ZZZZ_NONEXISTENT", model_type="openap")
        assert result is False
