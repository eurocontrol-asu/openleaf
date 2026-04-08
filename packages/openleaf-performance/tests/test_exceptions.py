"""Tests for AetherX exceptions."""

from __future__ import annotations

import pytest

from openleaf.performance.exceptions import OpenLeafError
from openleaf.performance.exceptions import BADA3ParseError
from openleaf.performance.exceptions import BADA4ParseError
from openleaf.performance.exceptions import ConfigurationError
from openleaf.performance.exceptions import EnvelopeError
from openleaf.performance.exceptions import MissingFieldError
from openleaf.performance.exceptions import ModelDataError
from openleaf.performance.exceptions import ModelError
from openleaf.performance.exceptions import ModelNotFoundError
from openleaf.performance.exceptions import ParsingError
from openleaf.performance.exceptions import ValidationError
from openleaf.performance.exceptions import ValueRangeError


class TestOpenLeafError:
    """Test base OpenLeafError."""

    def test_aetherx_error_is_exception(self):
        """Test OpenLeafError is an Exception."""
        err = OpenLeafError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"

    def test_catch_all_aetherx_errors(self):
        """Test catching all AetherX errors with base class."""
        with pytest.raises(OpenLeafError):
            raise ConfigurationError("config error")

        with pytest.raises(OpenLeafError):
            raise ModelError("model error")

        with pytest.raises(OpenLeafError):
            raise ParsingError("parsing error")


class TestConfigurationError:
    """Test ConfigurationError."""

    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        err = ConfigurationError("BADA path not configured")
        assert isinstance(err, OpenLeafError)
        assert "BADA path" in str(err)


class TestModelErrors:
    """Test model-related exceptions."""

    def test_model_error(self):
        """Test ModelError creation."""
        err = ModelError("model initialization failed")
        assert isinstance(err, OpenLeafError)

    def test_model_not_found_error(self):
        """Test ModelNotFoundError with attributes."""
        err = ModelNotFoundError(
            "Aircraft A320 not found",
            icao="A320",
            model_type="BADA4",
        )
        assert isinstance(err, ModelError)
        assert err.icao == "A320"
        assert err.model_type == "BADA4"

    def test_model_not_found_error_without_attributes(self):
        """Test ModelNotFoundError without optional attributes."""
        err = ModelNotFoundError("Aircraft not found")
        assert err.icao is None
        assert err.model_type is None

    def test_model_data_error(self):
        """Test ModelDataError creation."""
        err = ModelDataError("Invalid coefficient")
        assert isinstance(err, ModelError)


class TestParsingErrors:
    """Test parsing-related exceptions."""

    def test_parsing_error_basic(self):
        """Test ParsingError with basic message."""
        err = ParsingError("parse failed")
        assert str(err) == "parse failed"
        assert err.filepath is None
        assert err.line_number is None

    def test_parsing_error_with_filepath(self):
        """Test ParsingError with filepath."""
        err = ParsingError("parse failed", filepath="/path/to/file.xml")
        assert "file: /path/to/file.xml" in str(err)
        assert err.filepath == "/path/to/file.xml"

    def test_parsing_error_with_filepath_and_line(self):
        """Test ParsingError with filepath and line number."""
        err = ParsingError("parse failed", filepath="/path/to/file.xml", line_number=42)
        assert "file: /path/to/file.xml" in str(err)
        assert "line: 42" in str(err)

    def test_bada3_parse_error(self):
        """Test BADA3ParseError with section."""
        err = BADA3ParseError(
            "invalid coefficient",
            filepath="/path/to/A320__.OPF",
            section="AERO",
        )
        assert isinstance(err, ParsingError)
        assert err.section == "AERO"
        assert "file: /path/to/A320__.OPF" in str(err)
        assert "section: AERO" in str(err)

    def test_bada3_parse_error_minimal(self):
        """Test BADA3ParseError with minimal info."""
        err = BADA3ParseError("error")
        assert str(err) == "error"

    def test_bada4_parse_error(self):
        """Test BADA4ParseError with element."""
        err = BADA4ParseError(
            "invalid XML",
            filepath="/path/to/A320.xml",
            element="AERO",
        )
        assert isinstance(err, ParsingError)
        assert err.element == "AERO"
        assert "file: /path/to/A320.xml" in str(err)
        assert "element: AERO" in str(err)

    def test_bada4_parse_error_minimal(self):
        """Test BADA4ParseError with minimal info."""
        err = BADA4ParseError("error")
        assert str(err) == "error"


class TestValidationErrors:
    """Test validation-related exceptions."""

    def test_validation_error(self):
        """Test ValidationError creation."""
        err = ValidationError("validation failed")
        assert isinstance(err, OpenLeafError)

    def test_missing_field_error_basic(self):
        """Test MissingFieldError with field only."""
        err = MissingFieldError("altitude")
        assert isinstance(err, ValidationError)
        assert err.field == "altitude"
        assert err.method is None
        assert "altitude" in str(err)

    def test_missing_field_error_with_method(self):
        """Test MissingFieldError with method."""
        err = MissingFieldError("altitude", method="drag_coef")
        assert err.method == "drag_coef"
        assert "altitude" in str(err)
        assert "drag_coef" in str(err)

    def test_missing_field_error_with_custom_message(self):
        """Test MissingFieldError with custom message."""
        err = MissingFieldError("altitude", message="Custom error message")
        assert str(err) == "Custom error message"

    def test_value_range_error_both_bounds(self):
        """Test ValueRangeError with both bounds."""
        err = ValueRangeError("mach", 1.5, min_value=0.0, max_value=1.0)
        assert isinstance(err, ValidationError)
        assert err.field == "mach"
        assert err.value == 1.5
        assert err.min_value == 0.0
        assert err.max_value == 1.0
        assert "outside range" in str(err)

    def test_value_range_error_min_only(self):
        """Test ValueRangeError with min only."""
        err = ValueRangeError("mass", -100.0, min_value=0.0)
        assert "below minimum" in str(err)

    def test_value_range_error_max_only(self):
        """Test ValueRangeError with max only."""
        err = ValueRangeError("altitude", 50000.0, max_value=45000.0)
        assert "above maximum" in str(err)

    def test_value_range_error_no_bounds(self):
        """Test ValueRangeError with no bounds."""
        err = ValueRangeError("value", float("nan"))
        assert "is invalid" in str(err)

    def test_value_range_error_custom_message(self):
        """Test ValueRangeError with custom message."""
        err = ValueRangeError("mach", 1.5, message="Mach too high")
        assert str(err) == "Mach too high"


class TestEnvelopeError:
    """Test envelope-related exceptions."""

    def test_envelope_error_basic(self):
        """Test EnvelopeError with basic message."""
        err = EnvelopeError("speed exceeds limit")
        assert isinstance(err, OpenLeafError)

    def test_envelope_error_with_attributes(self):
        """Test EnvelopeError with all attributes."""
        err = EnvelopeError(
            "VMO exceeded",
            limit_type="VMO",
            value=350.0,
            limit=340.0,
        )
        assert err.limit_type == "VMO"
        assert err.value == 350.0
        assert err.limit == 340.0

    def test_envelope_error_partial_attributes(self):
        """Test EnvelopeError with partial attributes."""
        err = EnvelopeError("altitude exceeded", limit_type="HMO")
        assert err.limit_type == "HMO"
        assert err.value is None
        assert err.limit is None


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_all_errors_are_aetherx_errors(self):
        """Test all exceptions inherit from OpenLeafError."""
        exceptions = [
            ConfigurationError("test"),
            ModelError("test"),
            ModelNotFoundError("test"),
            ModelDataError("test"),
            ParsingError("test"),
            BADA3ParseError("test"),
            BADA4ParseError("test"),
            ValidationError("test"),
            MissingFieldError("field"),
            ValueRangeError("field", 0.0),
            EnvelopeError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, OpenLeafError), (
                f"{type(exc).__name__} should be OpenLeafError"
            )

    def test_model_errors_hierarchy(self):
        """Test model error hierarchy."""
        assert issubclass(ModelNotFoundError, ModelError)
        assert issubclass(ModelDataError, ModelError)
        assert issubclass(ModelError, OpenLeafError)

    def test_parsing_errors_hierarchy(self):
        """Test parsing error hierarchy."""
        assert issubclass(BADA3ParseError, ParsingError)
        assert issubclass(BADA4ParseError, ParsingError)
        assert issubclass(ParsingError, OpenLeafError)

    def test_validation_errors_hierarchy(self):
        """Test validation error hierarchy."""
        assert issubclass(MissingFieldError, ValidationError)
        assert issubclass(ValueRangeError, ValidationError)
        assert issubclass(ValidationError, OpenLeafError)
