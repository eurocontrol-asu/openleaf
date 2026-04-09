"""Shared helpers for BADA synonym resolution."""

from __future__ import annotations


__all__ = ["build_not_found_error"]

from collections.abc import Callable
from pathlib import Path
from typing import Any


def build_not_found_error(
    icao: str,
    backend_name: str,
    data_path: Path,
    synonym_file: Path,
    get_synonyms_fn: Callable[[Path], dict[str, Any]],
    glob_pattern: str,
) -> FileNotFoundError:
    """Build a FileNotFoundError with available aircraft list.

    Args:
        icao: ICAO code that was not found.
        backend_name: Backend name for error message (e.g. 'BADA3', 'BADA4').
        data_path: Path to data directory.
        synonym_file: Path to synonym file.
        get_synonyms_fn: Callable to load synonyms from file.
        glob_pattern: Glob pattern for listing files (e.g. '*.OPF', '*.xml').

    Returns:
        FileNotFoundError with formatted message.
    """
    if synonym_file.exists():
        synonyms = get_synonyms_fn(synonym_file)
        available = sorted(synonyms.keys())[:20]
        available_str = ", ".join(available)
        if len(synonyms) > 20:
            available_str += f", ... ({len(synonyms)} total)"
    else:
        matched = list(data_path.glob(glob_pattern))
        folders = [d for d in data_path.iterdir() if d.is_dir()]
        names = {f.stem.rstrip("_") for f in matched} | {d.name for d in folders}
        available = sorted(names)[:20]
        available_str = ", ".join(available)

    return FileNotFoundError(
        f"Aircraft '{icao}' not found in {backend_name} data.\n"
        f"Data path: {data_path}\n"
        f"Available aircraft: {available_str}"
    )
