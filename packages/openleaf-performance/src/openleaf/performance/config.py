"""
Configuration management for OpenLeaf using pydantic-settings.

This module provides configuration for BADA3 and BADA4 data paths and synonym files.
Configuration can be set via:
1. Environment variables (OPENLEAF_BADA3_PATH, OPENLEAF_BADA4_PATH, etc.)
2. Config file (~/.config/openleaf/settings.toml)
3. Programmatic API (set_bada3_path, set_bada4_path)

Usage:
    import openleaf.performance

    # View configuration
    openleaf.performance.config_path()      # Show config file location
    openleaf.performance.show_config()      # Print current configuration

    # Set BADA paths
    openleaf.performance.set_bada3_path("/path/to/bada3/data")
    openleaf.performance.set_bada4_path("/path/to/bada4/data")

    # Or use environment variables
    # export OPENLEAF_BADA3_PATH="/path/to/bada3/data"
    # export OPENLEAF_BADA4_PATH="/path/to/bada4/data"
"""

from __future__ import annotations


__all__ = [
    "config_path",
    "get_bada3_path",
    "get_bada3_synonyms_path",
    "get_bada4_path",
    "get_bada4_synonyms_path",
    "get_config",
    "set_bada3_path",
    "set_bada3_synonyms",
    "set_bada4_path",
    "set_bada4_synonyms",
    "show_config",
]

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from pydantic_settings import TomlConfigSettingsSource
from rich.console import Console
from rich.table import Table

from openleaf.performance.exceptions import ConfigurationError
from openleaf.performance.logging import get_logger


logger = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Platform-specific config directory
# ---------------------------------------------------------------------------


def _get_config_dir() -> Path:
    """Get platform-specific configuration directory.

    Returns:
        Path to configuration directory:
        - Linux/Mac: ~/.config/openleaf
        - Windows: %APPDATA%/openleaf
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    return base / "openleaf"


def _get_config_path() -> Path:
    """Get path to configuration file."""
    return _get_config_dir() / "settings.toml"


def _ensure_config_file() -> Path:
    """Ensure config file exists with template content."""
    config_file = _get_config_path()
    if not config_file.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            """\
# OpenLeaf Configuration
# https://github.com/eurocontrol-asu/openleaf
#
# Environment variables take precedence over this file:
#   OPENLEAF_BADA3_PATH, OPENLEAF_BADA3_SYNONYMS
#   OPENLEAF_BADA4_PATH, OPENLEAF_BADA4_SYNONYMS

[bada3]
# path = "/path/to/bada3/data"  # Directory containing *.OPF files
# synonyms = "/path/to/SYNONYM.NEW"  # Optional

[bada4]
# path = "/path/to/bada4/data"  # Directory containing aircraft folders
# synonyms = "/path/to/SYNONYM.BD4"  # Optional
"""
        )
    return config_file


# ---------------------------------------------------------------------------
# Pydantic-settings models
# ---------------------------------------------------------------------------


class Bada3Settings(BaseModel):
    """BADA3 data paths configuration."""

    path: str | None = None
    synonyms: str | None = None


class Bada4Settings(BaseModel):
    """BADA4 data paths configuration."""

    path: str | None = None
    synonyms: str | None = None


class OpenLeafSettings(BaseSettings):
    """OpenLeaf configuration.

    Settings are loaded from (highest to lowest priority):
    1. Environment variables (OPENLEAF_BADA3_PATH, OPENLEAF_BADA4_PATH, etc.)
    2. TOML config file (~/.config/openleaf/settings.toml)
    3. Field defaults (None)
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENLEAF_",
        env_nested_delimiter="_",
    )

    bada3: Bada3Settings = Field(default_factory=Bada3Settings)
    bada4: Bada4Settings = Field(default_factory=Bada4Settings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Priority: init > env vars > TOML file."""
        toml_path = _get_config_path()
        sources: list[Any] = [init_settings, env_settings]
        if toml_path.exists():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        return tuple(sources)


# ---------------------------------------------------------------------------
# Settings singleton with reload support
# ---------------------------------------------------------------------------

_settings: OpenLeafSettings | None = None


def _get_settings() -> OpenLeafSettings:
    """Get or create the settings singleton."""
    global _settings
    if _settings is None:
        _ensure_config_file()
        _settings = OpenLeafSettings()
    return _settings


def _reload_settings() -> None:
    """Force re-read of all configuration sources."""
    global _settings
    _settings = OpenLeafSettings()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def config_path() -> Path:
    """Get the path to the configuration file.

    Returns:
        Path to settings.toml file.
    """
    return _get_config_path()


def get_config() -> dict:
    """Get the current configuration as a dictionary.

    Returns:
        Dictionary with current settings.
    """
    s = _get_settings()
    return {
        "bada3": {
            "path": s.bada3.path,
            "synonyms": s.bada3.synonyms,
        },
        "bada4": {
            "path": s.bada4.path,
            "synonyms": s.bada4.synonyms,
        },
    }


def show_config() -> None:
    """Print the current configuration using Rich."""
    s = _get_settings()
    table = Table(title="OpenLeaf Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Config file", str(config_path()))
    table.add_row("", "")

    # BADA3
    table.add_row("[bold]BADA3[/bold]", "")
    table.add_row(
        "  path",
        s.bada3.path or "[dim](not set)[/dim]",
    )
    table.add_row(
        "  synonyms",
        s.bada3.synonyms or "[dim](auto-detect)[/dim]",
    )

    # BADA4
    table.add_row("[bold]BADA4[/bold]", "")
    table.add_row(
        "  path",
        s.bada4.path or "[dim](not set)[/dim]",
    )
    table.add_row(
        "  synonyms",
        s.bada4.synonyms or "[dim](auto-detect)[/dim]",
    )

    console.print(table)


def _toml_string(value: str) -> str:
    """Serialize *value* as a TOML basic string.

    Backslashes and double quotes must be escaped: interpolating a Windows
    path like ``C:\\Users\\x`` verbatim produces invalid TOML escape
    sequences and makes the settings file unreadable on the next load.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _insert_key_in_section(lines: list[str], section: str, key: str, value: str) -> list[str]:
    """Insert a key into the specified section of config lines."""
    result: list[str] = []
    section_found = False
    inserted = False

    for line in lines:
        if not inserted and section_found and line.strip().startswith("["):
            result.append(f"{key} = {_toml_string(value)}")
            inserted = True
        result.append(line)
        if not inserted and line.strip() == f"[{section}]":
            section_found = True

    if not inserted and section_found:
        result.append(f"{key} = {_toml_string(value)}")

    return result


def _update_config_file(section: str, key: str, value: str) -> None:
    """Update the config file with a new value."""
    config_file = _get_config_path()
    lines = config_file.read_text().splitlines()
    new_lines: list[str] = []
    in_section = False
    key_found = False

    for line in lines:
        if line.strip().startswith("["):
            in_section = line.strip() == f"[{section}]"

        if in_section and not key_found:
            stripped = line.lstrip("# ").strip()
            if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
                new_lines.append(f"{key} = {_toml_string(value)}")
                key_found = True
                continue

        new_lines.append(line)

    if not key_found:
        new_lines = _insert_key_in_section(new_lines, section, key, value)

    # Write atomically (temp file + rename) so a crash or a concurrent
    # reader never observes a truncated/partial settings file.
    tmp_file = config_file.with_name(config_file.name + ".tmp")
    tmp_file.write_text("\n".join(new_lines) + "\n")
    tmp_file.replace(config_file)

    # Reload settings to pick up the change
    _reload_settings()


def set_bada3_path(path: str | Path) -> None:
    """Set the BADA3 data directory path.

    Args:
        path: Directory containing BADA3 *.OPF files.
    """
    resolved_path = str(Path(path).resolve())
    _update_config_file("bada3", "path", resolved_path)
    logger.info("BADA3 path configured", path=resolved_path)
    console.print(f"[green]✓[/green] BADA3 path set to: {resolved_path}")


def set_bada3_synonyms(path: str | Path) -> None:
    """Set the BADA3 synonym file path.

    Args:
        path: Path to SYNONYM.NEW file.
    """
    resolved_path = str(Path(path).resolve())
    _update_config_file("bada3", "synonyms", resolved_path)
    logger.info("BADA3 synonyms configured", path=resolved_path)
    console.print(f"[green]✓[/green] BADA3 synonyms set to: {resolved_path}")


def set_bada4_path(path: str | Path) -> None:
    """Set the BADA4 data directory path.

    Args:
        path: Directory containing BADA4 aircraft folders.
    """
    resolved_path = str(Path(path).resolve())
    _update_config_file("bada4", "path", resolved_path)
    logger.info("BADA4 path configured", path=resolved_path)
    console.print(f"[green]✓[/green] BADA4 path set to: {resolved_path}")


def set_bada4_synonyms(path: str | Path) -> None:
    """Set the BADA4 synonym file path.

    Args:
        path: Path to SYNONYM.BD4 file.
    """
    resolved_path = str(Path(path).resolve())
    _update_config_file("bada4", "synonyms", resolved_path)
    logger.info("BADA4 synonyms configured", path=resolved_path)
    console.print(f"[green]✓[/green] BADA4 synonyms set to: {resolved_path}")


def get_bada3_path() -> Path:
    """Get the configured BADA3 data path.

    Returns:
        Path to BADA3 data directory.

    Raises:
        ConfigurationError: If BADA3 path is not configured.
    """
    path_str = _get_settings().bada3.path
    if not path_str:
        msg = (
            "BADA3 data path not configured.\n\n"
            "Please set it using:\n"
            "  >>> import openleaf.performance\n"
            "  >>> openleaf.performance.set_bada3_path('/path/to/bada3/data')\n\n"
            "Or set environment variable: OPENLEAF_BADA3_PATH\n\n"
            f"Config file location: {config_path()}"
        )
        raise ConfigurationError(msg)
    path = Path(path_str)
    if not path.exists():
        msg = f"BADA3 data path does not exist: {path}"
        raise ConfigurationError(msg)
    return path


def get_bada3_synonyms_path() -> Path | None:
    """Get the path to BADA3 synonyms file.

    Returns:
        Path to SYNONYM.NEW file, or None if not found.
    """
    s = _get_settings()

    # Check explicit config first
    if s.bada3.synonyms:
        path = Path(s.bada3.synonyms)
        if path.exists():
            return path

    # Auto-detect in data directory
    if s.bada3.path:
        path = Path(s.bada3.path) / "SYNONYM.NEW"
        if path.exists():
            return path

    return None


def get_bada4_path() -> Path:
    """Get the configured BADA4 data path.

    Returns:
        Path to BADA4 data directory.

    Raises:
        ConfigurationError: If BADA4 path is not configured.
    """
    path_str = _get_settings().bada4.path
    if not path_str:
        msg = (
            "BADA4 data path not configured.\n\n"
            "Please set it using:\n"
            "  >>> import openleaf.performance\n"
            "  >>> openleaf.performance.set_bada4_path('/path/to/bada4/data')\n\n"
            "Or set environment variable: OPENLEAF_BADA4_PATH\n\n"
            f"Config file location: {config_path()}"
        )
        raise ConfigurationError(msg)
    path = Path(path_str)
    if not path.exists():
        msg = f"BADA4 data path does not exist: {path}"
        raise ConfigurationError(msg)
    return path


def get_bada4_synonyms_path() -> Path | None:
    """Get the path to BADA4 synonyms file.

    Returns:
        Path to SYNONYM.BD4 file, or None if not found.
    """
    s = _get_settings()

    # Check explicit config first
    if s.bada4.synonyms:
        path = Path(s.bada4.synonyms)
        if path.exists():
            return path

    # Auto-detect in data directory
    if s.bada4.path:
        path = Path(s.bada4.path) / "SYNONYM.BD4"
        if path.exists():
            return path

    return None
