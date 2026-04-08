"""
BADA4 synonym file parser.

Parses SYNONYM.BD4 files to map ICAO aircraft codes to BADA4 XML folder names.

The SYNONYM.BD4 format:
- Lines starting with 'CC' are comments
- Lines starting with 'CD' are data lines
- Lines starting with 'FI' mark end of file

Data line format:
- Position 3: Support type ('-' = direct, '*' = equivalence)
- Position 5-8: ICAO code (4 characters)
- Position 10-27: Manufacturer (18 characters)
- Position 29-53: Name/Model (25 characters)
- Position 55-73: Folder name (e.g., 'A320-214')
- Position 75+: ICAO indicator ('Y' or 'N')
"""

from __future__ import annotations


__all__ = [
    "BADA4SynonymEntry",
    "load_bada4_synonyms",
    "resolve_bada4_file",
]

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict


class BADA4SynonymEntry(BaseModel):
    """Entry from BADA4 synonym file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    icao: str
    """ICAO aircraft type code (e.g., 'A320')."""

    manufacturer: str
    """Aircraft manufacturer (e.g., 'AIRBUS')."""

    model: str
    """Aircraft model name (e.g., 'A320')."""

    folder_name: str
    """Folder/file name (e.g., 'A320-214')."""

    is_direct: bool
    """True if directly supported, False if equivalence."""

    is_icao: bool
    """True if this is an official ICAO code."""


def _parse_synonym_line(line: str) -> BADA4SynonymEntry | None:
    """Parse a single synonym data line.

    Args:
        line: Line from SYNONYM.BD4 file.

    Returns:
        BADA4SynonymEntry if valid data line, None otherwise.
    """
    # Skip non-data lines
    if not line.startswith("CD"):
        return None

    # Line must be long enough
    if len(line) < 60:
        return None

    try:
        support_type = line[3]  # '-' or '*'
        icao = line[5:9].strip()
        manufacturer = line[10:28].strip()
        model = line[29:54].strip()

        # BADA4 has longer folder names - find the field by looking for 'Y' marker
        # Format: folder_name followed by spaces and 'Y' or 'N'
        rest = line[55:].rstrip(" /\n")

        # Split by multiple spaces to separate folder name from ICAO marker
        parts = rest.split()
        if not parts:
            return None

        folder_name = parts[0]
        is_icao_char = parts[1] if len(parts) > 1 else "N"

        if not icao or not folder_name:
            return None

        return BADA4SynonymEntry(
            icao=icao,
            manufacturer=manufacturer,
            model=model,
            folder_name=folder_name,
            is_direct=(support_type == "-"),
            is_icao=(is_icao_char == "Y"),
        )
    except (IndexError, ValueError):
        return None


def load_bada4_synonyms(synonym_file: str | Path) -> dict[str, BADA4SynonymEntry]:
    """Load BADA4 synonym mappings from file.

    Args:
        synonym_file: Path to SYNONYM.BD4 file.

    Returns:
        Dictionary mapping ICAO codes to synonym entries.
    """
    synonyms: dict[str, BADA4SynonymEntry] = {}

    with Path(synonym_file).open(encoding="latin-1") as f:
        for line in f:
            entry = _parse_synonym_line(line)
            if entry:
                synonyms[entry.icao] = entry

    return synonyms


# Cache for loaded synonyms
_synonym_cache: dict[str, dict[str, BADA4SynonymEntry]] = {}


def _get_synonyms(synonym_file: Path) -> dict[str, BADA4SynonymEntry]:
    """Get synonyms with caching."""
    key = str(synonym_file)
    if key not in _synonym_cache:
        _synonym_cache[key] = load_bada4_synonyms(synonym_file)
    return _synonym_cache[key]


def resolve_bada4_file(
    icao: str,
    data_path: Path,
    synonym_file: Path | None = None,
) -> Path:
    """Resolve ICAO code to BADA4 XML file path.

    Args:
        icao: ICAO aircraft type code (e.g., 'A320').
        data_path: Path to BADA4 data directory.
        synonym_file: Path to SYNONYM.BD4 file. If None, looks for it in data_path.

    Returns:
        Path to the XML file for this aircraft.

    Raises:
        FileNotFoundError: If aircraft not found in synonyms or XML file doesn't exist.
    """
    # Normalize ICAO code
    icao = icao.upper().strip()

    # Find synonym file
    if synonym_file is None:
        synonym_file = data_path / "SYNONYM.BD4"

    # Try to load synonyms
    if synonym_file.exists():
        synonyms = _get_synonyms(synonym_file)
        if icao in synonyms:
            entry = synonyms[icao]

            # BADA4 structure: data_path/folder_name/folder_name.xml
            # or just: data_path/folder_name.xml
            folder_path = data_path / entry.folder_name
            if folder_path.is_dir():
                xml_file = folder_path / f"{entry.folder_name}.xml"
                if xml_file.exists():
                    return xml_file
                # Try any XML in folder
                xml_files = list(folder_path.glob("*.xml"))
                if xml_files:
                    return xml_files[0]

            # Try direct XML file
            xml_file = data_path / f"{entry.folder_name}.xml"
            if xml_file.exists():
                return xml_file

    # Fallback: try direct file match
    # Check for folder/file structure
    for folder in data_path.iterdir():
        if folder.is_dir() and folder.name.upper().startswith(icao):
            xml_file = folder / f"{folder.name}.xml"
            if xml_file.exists():
                return xml_file
            xml_files = list(folder.glob("*.xml"))
            if xml_files:
                return xml_files[0]

    # Try direct XML files
    for xml_file in data_path.glob("*.xml"):
        if xml_file.stem.upper().startswith(icao):
            return xml_file

    # Not found
    available = []
    if synonym_file.exists():
        synonyms = _get_synonyms(synonym_file)
        available = sorted(synonyms.keys())[:20]
        available_str = ", ".join(available)
        if len(synonyms) > 20:
            available_str += f", ... ({len(synonyms)} total)"
    else:
        # List folders/XML files in directory
        xml_files = list(data_path.glob("*.xml"))
        folders = [d for d in data_path.iterdir() if d.is_dir()]
        names = {f.stem for f in xml_files} | {d.name for d in folders}
        available = sorted(names)[:20]
        available_str = ", ".join(available)

    raise FileNotFoundError(
        f"Aircraft '{icao}' not found in BADA4 data.\n"
        f"Data path: {data_path}\n"
        f"Available aircraft: {available_str}"
    )


def available_bada4_aircraft(
    data_path: Path,
    synonym_file: Path | None = None,
) -> list[str]:
    """List available BADA4 aircraft types.

    Args:
        data_path: Path to BADA4 data directory.
        synonym_file: Path to SYNONYM.BD4 file. If None, looks for it in data_path.

    Returns:
        Sorted list of available ICAO codes.
    """
    if synonym_file is None:
        synonym_file = data_path / "SYNONYM.BD4"

    if synonym_file.exists():
        synonyms = _get_synonyms(synonym_file)
        # Only return ICAO codes for which the backing XML/folder actually exists.
        available = []
        for icao, entry in synonyms.items():
            folder_path = data_path / entry.folder_name
            if folder_path.is_dir():
                available.append(icao)
                continue
            if (data_path / f"{entry.folder_name}.xml").exists():
                available.append(icao)
        return sorted(available)

    # Fallback: list folders and XML files
    xml_files = list(data_path.glob("*.xml"))
    folders = [d for d in data_path.iterdir() if d.is_dir()]
    names = {f.stem for f in xml_files} | {d.name for d in folders}
    return sorted(names)
