"""
BADA3 synonym file parser.

Parses SYNONYM.NEW files to map ICAO aircraft codes to BADA3 OPF file names.

The SYNONYM.NEW format (from BADA 3.0 User Manual):
- Lines starting with 'CC' are comments
- Lines starting with 'CD' are data lines
- Lines starting with 'FI' mark end of file

Data line format (Fortran notation: 'CD', 1X, A1, 1X, A4, 3X, A18, 1X, A25, 1X, A6, 2X, A4):
- Position 3: Support type ('-' = direct, '*' = equivalence)
- Position 5-8: ICAO code (4 characters)
- Position 10-27: Manufacturer (18 characters)
- Position 29-53: Name/Model (25 characters)
- Position 55-60: File name (6 characters, e.g., 'A320__')
- Position 63: ICAO indicator ('Y' or 'N')
"""

from __future__ import annotations


__all__ = [
    "BADA3SynonymEntry",
    "load_bada3_synonyms",
    "resolve_bada3_file",
]

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict


class BADA3SynonymEntry(BaseModel):
    """Entry from BADA3 synonym file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    icao: str
    """ICAO aircraft type code (e.g., 'A320')."""

    manufacturer: str
    """Aircraft manufacturer (e.g., 'AIRBUS')."""

    model: str
    """Aircraft model name (e.g., 'A320-231')."""

    file_prefix: str
    """OPF file prefix (e.g., 'A320__')."""

    is_direct: bool
    """True if directly supported, False if equivalence."""

    is_icao: bool
    """True if this is an official ICAO code."""


def _parse_synonym_line(line: str) -> BADA3SynonymEntry | None:
    """Parse a single synonym data line.

    Args:
        line: Line from SYNONYM.NEW file.

    Returns:
        BADA3SynonymEntry if valid data line, None otherwise.
    """
    # Skip non-data lines
    if not line.startswith("CD"):
        return None

    # Line must be long enough
    if len(line) < 65:
        return None

    try:
        support_type = line[3]  # '-' or '*'
        icao = line[5:9].strip()
        manufacturer = line[10:28].strip()
        model = line[29:54].strip()
        file_prefix = line[55:61].strip()
        is_icao_char = line[63] if len(line) > 63 else "N"

        if not icao or not file_prefix:
            return None

        return BADA3SynonymEntry(
            icao=icao,
            manufacturer=manufacturer,
            model=model,
            file_prefix=file_prefix,
            is_direct=(support_type == "-"),
            is_icao=(is_icao_char == "Y"),
        )
    except (IndexError, ValueError):
        return None


def load_bada3_synonyms(synonym_file: str | Path) -> dict[str, BADA3SynonymEntry]:
    """Load BADA3 synonym mappings from file.

    Args:
        synonym_file: Path to SYNONYM.NEW file.

    Returns:
        Dictionary mapping ICAO codes to synonym entries.
    """
    synonyms: dict[str, BADA3SynonymEntry] = {}

    with Path(synonym_file).open(encoding="latin-1") as f:
        for line in f:
            entry = _parse_synonym_line(line)
            if entry:
                synonyms[entry.icao] = entry

    return synonyms


# Cache for loaded synonyms
_synonym_cache: dict[str, dict[str, BADA3SynonymEntry]] = {}


def _get_synonyms(synonym_file: Path) -> dict[str, BADA3SynonymEntry]:
    """Get synonyms with caching."""
    key = str(synonym_file)
    if key not in _synonym_cache:
        _synonym_cache[key] = load_bada3_synonyms(synonym_file)
    return _synonym_cache[key]


def resolve_bada3_file(
    icao: str,
    data_path: Path,
    synonym_file: Path | None = None,
) -> Path:
    """Resolve ICAO code to BADA3 OPF file path.

    Args:
        icao: ICAO aircraft type code (e.g., 'A320').
        data_path: Path to BADA3 data directory.
        synonym_file: Path to SYNONYM.NEW file. If None, looks for it in data_path.

    Returns:
        Path to the OPF file for this aircraft.

    Raises:
        FileNotFoundError: If aircraft not found in synonyms or OPF file doesn't exist.
    """
    # Normalize ICAO code
    icao = icao.upper().strip()

    # Find synonym file
    if synonym_file is None:
        synonym_file = data_path / "SYNONYM.NEW"

    # Try to load synonyms
    if synonym_file.exists():
        synonyms = _get_synonyms(synonym_file)
        if icao in synonyms:
            entry = synonyms[icao]
            opf_file = data_path / f"{entry.file_prefix}.OPF"
            if opf_file.exists():
                return opf_file
            # Try with different underscore patterns
            for suffix in ["__", "___", "_"]:
                base = entry.file_prefix.rstrip("_")
                opf_file = data_path / f"{base}{suffix}.OPF"
                if opf_file.exists():
                    return opf_file

    # Fallback: try direct file match
    for suffix in ["__", "___", "_", ""]:
        opf_file = data_path / f"{icao}{suffix}.OPF"
        if opf_file.exists():
            return opf_file

    # Not found
    available = []
    if synonym_file.exists():
        synonyms = _get_synonyms(synonym_file)
        available = sorted(synonyms.keys())[:20]
        available_str = ", ".join(available)
        if len(synonyms) > 20:
            available_str += f", ... ({len(synonyms)} total)"
    else:
        # List OPF files in directory
        opf_files = list(data_path.glob("*.OPF"))
        available = sorted([f.stem.rstrip("_") for f in opf_files])[:20]
        available_str = ", ".join(available)

    raise FileNotFoundError(
        f"Aircraft '{icao}' not found in BADA3 data.\n"
        f"Data path: {data_path}\n"
        f"Available aircraft: {available_str}"
    )


def available_bada3_aircraft(
    data_path: Path,
    synonym_file: Path | None = None,
) -> list[str]:
    """List available BADA3 aircraft types.

    Args:
        data_path: Path to BADA3 data directory.
        synonym_file: Path to SYNONYM.NEW file. If None, looks for it in data_path.

    Returns:
        Sorted list of available ICAO codes.
    """
    if synonym_file is None:
        synonym_file = data_path / "SYNONYM.NEW"

    if synonym_file.exists():
        synonyms = _get_synonyms(synonym_file)
        # Only return ICAO codes for which an OPF file actually exists locally.
        # (A full BADA3 install has all OPF files; a partial install has fewer.)
        available = []
        for icao, entry in synonyms.items():
            opf_file = data_path / f"{entry.file_prefix}.OPF"
            if opf_file.exists():
                available.append(icao)
                continue
            # Try alternate underscore patterns
            base = entry.file_prefix.rstrip("_")
            if any((data_path / f"{base}{s}.OPF").exists() for s in ("__", "___", "_", "")):
                available.append(icao)
        return sorted(available)

    # Fallback: list OPF files
    opf_files = list(data_path.glob("*.OPF"))
    return sorted({f.stem.rstrip("_") for f in opf_files})
