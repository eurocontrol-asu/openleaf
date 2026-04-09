from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openleaf.performance.core.factory import ModelFactory
from openleaf.performance.core.factory import ModelType


@pytest.fixture
def mock_model():
    """Return a callable that produces a mock AircraftModel."""
    model = MagicMock()
    return MagicMock(return_value=model)


@pytest.fixture
def model_classes(mock_model):
    """Model classes dict with all types available."""
    return {
        ModelType.BADA4: mock_model,
        ModelType.BADA3: mock_model,
        ModelType.OPENAP: mock_model,
        ModelType.PS: mock_model,
    }


# --- Unit tests ---


def test_load_explicit_type_string(monkeypatch, mock_model):
    """load('A320', model_type='openap') returns OpenAP model."""
    classes = {
        ModelType.BADA4: None,
        ModelType.BADA3: None,
        ModelType.OPENAP: mock_model,
        ModelType.PS: None,
    }
    monkeypatch.setattr(ModelFactory, "_get_model_classes", classmethod(lambda cls: classes))
    result = ModelFactory.load("A320", model_type="openap")
    mock_model.assert_called_once_with("A320")
    assert result is mock_model.return_value


def test_load_explicit_type_enum(monkeypatch, mock_model):
    """load('A320', model_type=ModelType.OPENAP) returns OpenAP model."""
    classes = {
        ModelType.BADA4: None,
        ModelType.BADA3: None,
        ModelType.OPENAP: mock_model,
        ModelType.PS: None,
    }
    monkeypatch.setattr(ModelFactory, "_get_model_classes", classmethod(lambda cls: classes))
    result = ModelFactory.load("A320", model_type=ModelType.OPENAP)
    mock_model.assert_called_once_with("A320")
    assert result is mock_model.return_value


def test_load_auto_detect(monkeypatch):
    """Auto-detect returns highest-priority available model."""
    bada4_model = MagicMock()
    openap_model = MagicMock()
    classes = {
        ModelType.BADA4: MagicMock(return_value=bada4_model),
        ModelType.BADA3: None,
        ModelType.OPENAP: MagicMock(return_value=openap_model),
        ModelType.PS: None,
    }
    monkeypatch.setattr(ModelFactory, "_get_model_classes", classmethod(lambda cls: classes))
    result = ModelFactory.load("A320")
    # BADA4 has highest priority, so it should be returned
    assert result is bada4_model
    classes[ModelType.BADA4].assert_called_once_with("A320")
    classes[ModelType.OPENAP].assert_not_called()


def test_load_invalid_type(monkeypatch, model_classes):
    """load('A320', model_type='invalid') raises ValueError with valid types."""
    monkeypatch.setattr(ModelFactory, "_get_model_classes", classmethod(lambda cls: model_classes))
    with pytest.raises(ValueError, match=r"Invalid model_type 'invalid'") as exc_info:
        ModelFactory.load("A320", model_type="invalid")
    # Check that valid types are listed in the error message
    assert "Valid types are" in str(exc_info.value)


# --- Edge cases ---


def test_load_no_models_available(monkeypatch):
    """All model classes return None -> ValueError listing all failures."""
    classes = {
        ModelType.BADA4: None,
        ModelType.BADA3: None,
        ModelType.OPENAP: None,
        ModelType.PS: None,
    }
    monkeypatch.setattr(ModelFactory, "_get_model_classes", classmethod(lambda cls: classes))
    with pytest.raises(ValueError, match=r"Aircraft 'A320' not found in any model") as exc_info:
        ModelFactory.load("A320")
    error_msg = str(exc_info.value)
    # All types should appear as not configured
    assert "not configured" in error_msg.lower() or "not available" in error_msg.lower()


def test_load_model_type_not_configured(monkeypatch):
    """Explicit type whose class is None -> ValueError about data path."""
    classes = {
        ModelType.BADA4: None,
        ModelType.BADA3: None,
        ModelType.OPENAP: None,
        ModelType.PS: None,
    }
    monkeypatch.setattr(ModelFactory, "_get_model_classes", classmethod(lambda cls: classes))
    with pytest.raises(ValueError, match=r"not available") as exc_info:
        ModelFactory.load("A320", model_type=ModelType.BADA4)
    assert "data path" in str(exc_info.value).lower()
