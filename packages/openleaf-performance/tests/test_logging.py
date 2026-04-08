"""Tests for aetherx.logging module."""

from __future__ import annotations

import logging

from openleaf.performance.logging import configure_logging
from openleaf.performance.logging import enable_logging
from openleaf.performance.logging import get_logger


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_configure_with_string_level(self):
        """Accepts string level like 'DEBUG'."""
        configure_logging(level="DEBUG")

    def test_configure_with_int_level(self):
        """Accepts integer level like logging.INFO."""
        configure_logging(level=logging.INFO)

    def test_configure_with_json_format(self):
        """JSON format mode uses JSONRenderer."""
        configure_logging(level="WARNING", json_format=True)

    def test_configure_with_no_colors(self):
        """Explicit colors=False disables color output."""
        configure_logging(level="WARNING", colors=False)

    def test_configure_default(self):
        """Default configuration uses environment or WARNING."""
        configure_logging()


class TestGetLogger:
    """Tests for get_logger()."""

    def test_get_logger_with_name(self):
        """Logger with module name gets aetherx prefix."""
        logger = get_logger("mymodule")
        assert logger is not None

    def test_get_logger_without_name(self):
        """Logger without name returns root aetherx logger."""
        logger = get_logger(None)
        assert logger is not None

    def test_get_logger_with_aetherx_prefix(self):
        """Logger already prefixed with aetherx keeps name."""
        logger = get_logger("openleaf.performance.models.ps")
        assert logger is not None

    def test_get_logger_with_context(self):
        """Logger with initial context binds it."""
        logger = get_logger("test", aircraft="A320")
        assert logger is not None


class TestEnableLogging:
    """Tests for enable_logging()."""

    def test_enable_logging(self):
        """enable_logging restores given level."""
        enable_logging(level="INFO")

    def test_enable_warning(self):
        """enable_logging with WARNING level."""
        enable_logging(level="WARNING")
