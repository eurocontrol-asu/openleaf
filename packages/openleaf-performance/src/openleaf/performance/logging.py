"""
Structured logging for AetherX using structlog.

Provides structured, contextual logging for the library with configurable
verbosity. Uses structlog for better debugging and log analysis.

Usage:
    from openleaf.performance.logging import get_logger, configure_logging

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.debug("Loading model", aircraft="A320", source="BADA4")
    logger.info("Computation complete", duration=0.5)
    logger.warning("Value exceeds limit", value=350, limit=340)

    # Enable debug logging
    configure_logging(level="DEBUG")

    # Or use environment variable
    # OPENLEAF_LOG_LEVEL=DEBUG python your_script.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


__all__ = [
    "configure_logging",
    "disable_logging",
    "enable_logging",
    "get_logger",
]

# Default log level (can be overridden by environment variable)
_DEFAULT_LOG_LEVEL = "WARNING"
_ENV_VAR = "OPENLEAF_LOG_LEVEL"
_LOGGER_NAME = "openleaf"

# Track if logging has been configured
_logging_configured = False


def _get_log_level() -> int:
    """Get log level from environment or default."""
    level_name = os.environ.get(_ENV_VAR, _DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.WARNING)


def configure_logging(
    level: str | int | None = None,
    json_format: bool = False,
    colors: bool | None = None,
) -> None:
    """Configure structured logging for AetherX.

    This function sets up structlog with sensible defaults. Call it once
    at application startup, or let it auto-configure on first logger use.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or None
            to use environment variable or default.
        json_format: If True, output logs as JSON (useful for log aggregation).
            If False (default), use human-readable console output.
        colors: Whether to use colors in console output. If None, auto-detect
            based on terminal capabilities.

    Example:
        >>> configure_logging(level="DEBUG")
        >>> configure_logging(level="INFO", json_format=True)
    """
    global _logging_configured

    # Determine log level
    if level is None:
        numeric_level = _get_log_level()
    elif isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.WARNING)
    else:
        numeric_level = level

    # Auto-detect colors
    if colors is None:
        colors = sys.stderr.isatty()

    # Configure processors based on output format
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        # JSON output for production/log aggregation
        processors = [
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=colors),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging for any third-party libraries
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    _logging_configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """Get a structured logger for AetherX modules.

    Returns a structlog bound logger with optional initial context.
    The logger supports structured key-value logging.

    Args:
        name: Logger name. Typically __name__ of the calling module.
            If None, returns the root AetherX logger.
        **initial_context: Initial context to bind to the logger.

    Returns:
        Configured structlog BoundLogger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Loading model data")
        >>> logger.info("Model loaded", aircraft="A320", source="BADA4")
        >>> logger.warning("Value exceeds VMO", value=350, vmo=340)

        # With initial context
        >>> logger = get_logger(__name__, aircraft="A320")
        >>> logger.info("Computing drag")  # aircraft="A320" included automatically
    """
    global _logging_configured

    # Auto-configure on first use
    if not _logging_configured:
        configure_logging()

    # Get or create logger
    if name is None:
        name = _LOGGER_NAME
    elif not name.startswith(_LOGGER_NAME):
        name = f"{_LOGGER_NAME}.{name}"

    logger = structlog.get_logger(name)

    # Bind initial context if provided
    if initial_context:
        logger = logger.bind(**initial_context)

    return logger


def disable_logging() -> None:
    """Disable all AetherX logging output.

    Useful for library users who don't want any output.
    Sets the log level to CRITICAL+1 to suppress all messages.
    """
    configure_logging(level=logging.CRITICAL + 1)


def enable_logging(level: str = "WARNING") -> None:
    """Re-enable AetherX logging after disable_logging().

    Args:
        level: Log level to restore. Defaults to WARNING.
    """
    configure_logging(level=level)
