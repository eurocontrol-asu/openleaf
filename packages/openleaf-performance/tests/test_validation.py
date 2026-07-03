"""Tests for validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from openleaf.performance.exceptions import MissingFieldError
from openleaf.performance.exceptions import ValueRangeError
from openleaf.performance.utils.validation import get_optional
from openleaf.performance.utils.validation import get_required
from openleaf.performance.utils.validation import require_one_of
from openleaf.performance.utils.validation import validate_positive
from openleaf.performance.utils.validation import validate_range


class TestGetRequired:
    """Test get_required function."""

    def test_get_required_success(self):
        """Test getting required field that exists."""
        data = {"altitude": 35000.0, "mach": 0.78}
        result = get_required(data, "altitude")
        assert result == 35000.0

    def test_get_required_missing_raises(self):
        """Test getting missing required field raises MissingFieldError."""
        data = {"altitude": 35000.0}
        with pytest.raises(MissingFieldError) as exc_info:
            get_required(data, "mach")
        assert exc_info.value.field == "mach"

    def test_get_required_with_method(self):
        """Test get_required includes method in error message."""
        data = {}
        with pytest.raises(MissingFieldError) as exc_info:
            get_required(data, "altitude", method="drag_coef")
        assert exc_info.value.method == "drag_coef"
        assert "drag_coef" in str(exc_info.value)


class TestGetOptional:
    """Test get_optional function."""

    def test_get_optional_exists(self):
        """Test getting optional field that exists."""
        data = {"altitude": 35000.0}
        result = get_optional(data, "altitude", default=0.0)
        assert result == 35000.0

    def test_get_optional_missing_returns_default(self):
        """Test getting missing optional field returns default."""
        data = {}
        result = get_optional(data, "altitude", default=0.0)
        assert result == 0.0

    def test_get_optional_default_none(self):
        """Test get_optional default is None."""
        data = {}
        result = get_optional(data, "altitude")
        assert result is None


class TestValidateRange:
    """Test validate_range function."""

    def test_validate_range_valid_scalar(self):
        """Test validating valid scalar value."""
        assert validate_range(0.78, "mach", min_value=0.0, max_value=1.0) is True

    def test_validate_range_valid_array(self):
        """Test validating valid array values."""
        values = np.array([0.5, 0.6, 0.7, 0.8])
        assert validate_range(values, "mach", min_value=0.0, max_value=1.0) is True

    def test_validate_range_below_min_raises(self):
        """Test value below min raises ValueRangeError."""
        with pytest.raises(ValueRangeError) as exc_info:
            validate_range(-0.1, "mach", min_value=0.0)
        assert exc_info.value.field == "mach"
        assert exc_info.value.min_value == 0.0

    def test_validate_range_above_max_raises(self):
        """Test value above max raises ValueRangeError."""
        with pytest.raises(ValueRangeError) as exc_info:
            validate_range(1.5, "mach", max_value=1.0)
        assert exc_info.value.field == "mach"
        assert exc_info.value.max_value == 1.0

    def test_validate_range_array_below_min_raises(self):
        """Test array with value below min raises."""
        values = np.array([0.5, -0.1, 0.8])
        with pytest.raises(ValueRangeError):
            validate_range(values, "mach", min_value=0.0)

    def test_validate_range_array_above_max_raises(self):
        """Test array with value above max raises."""
        values = np.array([0.5, 1.1, 0.8])
        with pytest.raises(ValueRangeError):
            validate_range(values, "mach", max_value=1.0)

    def test_validate_range_no_raise_returns_false(self):
        """Test validate_range with raise_error=False returns bool."""
        assert validate_range(-0.1, "mach", min_value=0.0, raise_error=False) is False
        assert validate_range(1.5, "mach", max_value=1.0, raise_error=False) is False
        assert validate_range(0.5, "mach", min_value=0.0, max_value=1.0, raise_error=False) is True

    def test_validate_range_min_only(self):
        """Test validating with min_value only."""
        assert validate_range(100.0, "mass", min_value=0.0) is True
        with pytest.raises(ValueRangeError):
            validate_range(-1.0, "mass", min_value=0.0)

    def test_validate_range_max_only(self):
        """Test validating with max_value only."""
        assert validate_range(35000.0, "altitude", max_value=45000.0) is True
        with pytest.raises(ValueRangeError):
            validate_range(50000.0, "altitude", max_value=45000.0)


class TestValidatePositive:
    """Test validate_positive function."""

    def test_validate_positive_valid(self):
        """Test validating positive value."""
        assert validate_positive(100.0, "mass") is True

    def test_validate_positive_zero_not_allowed(self):
        """Test zero is not allowed by default."""
        with pytest.raises(ValueRangeError):
            validate_positive(0.0, "mass")

    def test_validate_positive_zero_allowed(self):
        """Test zero is allowed when allow_zero=True."""
        assert validate_positive(0.0, "mass", allow_zero=True) is True

    def test_validate_positive_negative_raises(self):
        """Test negative value raises."""
        with pytest.raises(ValueRangeError):
            validate_positive(-100.0, "mass")

    def test_validate_positive_no_raise(self):
        """Test validate_positive with raise_error=False."""
        assert validate_positive(-100.0, "mass", raise_error=False) is False
        assert validate_positive(100.0, "mass", raise_error=False) is True


class TestRequireOneOf:
    """Test require_one_of function."""

    def test_require_one_of_first_found(self):
        """Test require_one_of returns first found field."""
        data = {"altitude": 35000.0, "pressure": 22600.0}
        field, value = require_one_of(data, ["altitude", "pressure"])
        assert field == "altitude"
        assert value == 35000.0

    def test_require_one_of_second_found(self):
        """Test require_one_of returns second if first missing."""
        data = {"pressure": 22600.0}
        field, value = require_one_of(data, ["altitude", "pressure"])
        assert field == "pressure"
        assert value == 22600.0

    def test_require_one_of_none_found_raises(self):
        """Test require_one_of raises if none found."""
        data = {"mach": 0.78}
        with pytest.raises(MissingFieldError) as exc_info:
            require_one_of(data, ["altitude", "pressure"])
        assert "altitude" in str(exc_info.value) or "pressure" in str(exc_info.value)

    def test_require_one_of_with_method(self):
        """Test require_one_of includes method in error."""
        data = {}
        with pytest.raises(MissingFieldError) as exc_info:
            require_one_of(data, ["altitude", "pressure"], method="compute")
        assert "compute" in str(exc_info.value)
