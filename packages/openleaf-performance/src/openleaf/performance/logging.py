"""
Structured logging for OpenLeaf using structlog.

Provides structured, contextual logging for the library with configurable
verbosity, without touching global logging state: importing openleaf never
calls ``structlog.configure()`` nor ``logging.basicConfig()``, so host
applications keep full control of their own logging.

Behavior:
    - If the host application has configured structlog
      (``structlog.is_configured()``), OpenLeaf loggers flow through the
      application's pipeline untouched.
    - Otherwise, OpenLeaf loggers use a private, library-scoped pipeline
      writing to stderr, filtered at WARNING by default (override with the
      ``OPENLEAF_LOG_LEVEL`` environment variable or ``configure_logging()``).

Usage:
    from openleaf.performance.logging import get_logger, configure_logging

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.debug("Loading model", aircraft="A320", source="BADA4")
    logger.info("Computation complete", duration=0.5)
    logger.warning("Value exceeds limit", value=350, limit=340)

    # Enable debug logging (affects OpenLeaf's own loggers only)
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

# Map structlog method names to stdlib numeric levels for filtering.
_METHOD_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "msg": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "exception": logging.ERROR,
    "critical": logging.CRITICAL,
    "fatal": logging.CRITICAL,
}


def _get_log_level() -> int:
    """Get log level from environment or default."""
    level_name = os.environ.get(_ENV_VAR, _DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.WARNING)


# Library-scoped state. Mutated by configure_logging()/disable_logging();
# consulted dynamically by the processors below, so level changes apply to
# loggers that already exist.
_state: dict[str, Any] = {
    "level": _get_log_level(),
    "json_format": False,
    "colors": None,  # None = auto-detect at first render
    "renderer": None,  # lazily built from json_format/colors
}


def _filter_by_level(
    logger: Any, method_name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Drop events below the library's current log level."""
    if _METHOD_LEVELS.get(method_name, logging.INFO) < _state["level"]:
        raise structlog.DropEvent
    return event_dict


def _render(logger: Any, method_name: str, event_dict: structlog.types.EventDict) -> str:
    """Render with the renderer matching the current configuration."""
    renderer = _state["renderer"]
    if renderer is None:
        if _state["json_format"]:
            renderer = structlog.processors.JSONRenderer()
        else:
            colors = _state["colors"]
            if colors is None:
                colors = sys.stderr.isatty()
            renderer = structlog.dev.ConsoleRenderer(colors=colors)
        _state["renderer"] = renderer
    return renderer(logger, method_name, event_dict)


_PROCESSORS: list[structlog.types.Processor] = [
    _filter_by_level,
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.dev.set_exc_info,
    structlog.processors.TimeStamper(fmt="iso"),
    _render,
]


def configure_logging(
    level: str | int | None = None,
    json_format: bool = False,
    colors: bool | None = None,
) -> None:
    """Configure OpenLeaf's library-scoped logging.

    This adjusts only OpenLeaf's own loggers. It deliberately does NOT call
    ``structlog.configure()`` or ``logging.basicConfig()`` — a library must
    not reconfigure the host application's logging. If your application has
    configured structlog globally, OpenLeaf loggers use the application's
    pipeline and this function only records the requested level for the
    fallback pipeline.

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
    if level is None:
        _state["level"] = _get_log_level()
    elif isinstance(level, str):
        _state["level"] = getattr(logging, level.upper(), logging.WARNING)
    else:
        _state["level"] = level

    _state["json_format"] = json_format
    _state["colors"] = colors
    _state["renderer"] = None  # rebuild lazily with the new settings


def get_logger(name: str | None = None, **initial_context: Any) -> Any:
    """Get a structured logger for OpenLeaf modules.

    Returns a structlog bound logger with optional initial context. If the
    host application has configured structlog, the application's pipeline is
    used; otherwise a library-scoped stderr pipeline (WARNING by default).

    Args:
        name: Logger name. Typically __name__ of the calling module.
            If None, returns the root OpenLeaf logger.
        **initial_context: Initial context to bind to the logger.

    Returns:
        Configured structlog bound logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Loading model data")
        >>> logger.info("Model loaded", aircraft="A320", source="BADA4")
        >>> logger.warning("Value exceeds VMO", value=350, vmo=340)

        # With initial context
        >>> logger = get_logger(__name__, aircraft="A320")
        >>> logger.info("Computing drag")  # aircraft="A320" included automatically
    """
    if name is None:
        name = _LOGGER_NAME
    elif not name.startswith(_LOGGER_NAME):
        name = f"{_LOGGER_NAME}.{name}"

    if structlog.is_configured():
        # Respect the application's structlog configuration.
        logger = structlog.get_logger(name)
    else:
        # Library-scoped fallback: never touches global structlog/stdlib state.
        logger = structlog.wrap_logger(
            structlog.PrintLogger(file=sys.stderr),
            processors=_PROCESSORS,
        ).bind(logger=name)

    if initial_context:
        logger = logger.bind(**initial_context)

    return logger


def disable_logging() -> None:
    """Disable all OpenLeaf logging output.

    Useful for library users who don't want any output.
    Sets the log level to CRITICAL+1 to suppress all messages.
    """
    _state["level"] = logging.CRITICAL + 1


def enable_logging(level: str = "WARNING") -> None:
    """Re-enable OpenLeaf logging after disable_logging().

    Args:
        level: Log level to restore. Defaults to WARNING.
    """
    configure_logging(level=level)
