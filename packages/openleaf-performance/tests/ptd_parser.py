"""
Parser for BADA Performance Table Data (PTD) files.

PTD files contain pre-computed performance data for validation of BADA models.
This parser extracts the tabular data for comparison against model outputs.
"""

__all__ = [
    "PTDData",
    "PTDRow",
    "PTDSection",
    "parse_ptd_file",
]

import re
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict


class PTDRow(BaseModel):
    """A single row of PTD performance data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fl: int  # Flight level
    temperature: float  # Temperature [K]
    pressure: float  # Pressure [Pa]
    density: float  # Density [kg/m³]
    speed_of_sound: float  # Speed of sound [m/s]
    tas: float  # True airspeed [kt]
    cas: float  # Calibrated airspeed [kt]
    mach: float  # Mach number
    mass: float  # Aircraft mass [kg]
    thrust: float  # Thrust [N]
    drag: float  # Drag [N]
    fuel_flow: float  # Fuel flow [kg/min]
    f: float | None = None  # Energy share factor [-]
    rocd: float | None = None  # Rate of climb/descent [fpm]
    gamma: float | None = None  # Flight path angle [deg]
    config: str | None = None  # Aerodynamic configuration (TO, IC, CR, AP, LD)


class PTDSection(BaseModel):
    """A section of PTD data (e.g., Low/Medium/High mass climb)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    mass_category: str  # "low", "medium", "high"
    phase: str  # "climb", "cruise", "descent"
    rows: list[PTDRow]


class PTDData(BaseModel):
    """Complete PTD file data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str
    sections: list[PTDSection]

    def get_section(self, mass_category: str, phase: str = "climb") -> PTDSection | None:
        """Get a specific section by mass category and phase."""
        for section in self.sections:
            if section.mass_category == mass_category and section.phase == phase:
                return section
        return None

    def get_row_at_fl(self, fl: int, mass_category: str, phase: str = "climb") -> PTDRow | None:
        """Get a specific row by flight level."""
        section = self.get_section(mass_category, phase)
        if section is None:
            return None
        for row in section.rows:
            if row.fl == fl:
                return row
        return None


def _parse_mass_category(section_name: str) -> str:
    """Extract mass category from section name."""
    name_lower = section_name.lower()
    if "low" in name_lower:
        return "low"
    elif "medium" in name_lower:
        return "medium"
    elif "high" in name_lower:
        return "high"
    return "unknown"


def _parse_phase(section_name: str) -> str:
    """Extract flight phase from section name."""
    name_lower = section_name.lower()
    if "climb" in name_lower:
        return "climb"
    elif "cruise" in name_lower:
        return "cruise"
    elif "descent" in name_lower or "desc" in name_lower:
        return "descent"
    return "unknown"


def _parse_data_row(line: str) -> PTDRow | None:
    """Parse a single data row from PTD file.

    PTD format columns:
    0: FL, 1: T, 2: p, 3: rho, 4: a, 5: TAS, 6: CAS, 7: M, 8: mass,
    9: Thrust, 10: Drag, 11: Fuel, 12: ESF, 13: ROCD, 14: gamma, 15: Conf

    Args:
        line: A line of text from the PTD file.

    Returns:
        PTDRow if parsing succeeds, None otherwise.
    """
    # Split on whitespace and filter empty strings
    parts = line.split()

    # Skip lines that don't look like data rows
    if len(parts) < 12:
        return None

    # First element should be numeric (FL)
    try:
        fl = int(parts[0])
    except ValueError:
        return None

    try:
        # Parse optional extended columns
        f_val = float(parts[12]) if len(parts) > 12 else None
        rocd = float(parts[13]) if len(parts) > 13 else None
        gamma = float(parts[14]) if len(parts) > 14 else None

        # Config is a string (TO, IC, CR, AP, LD)
        config = None
        if len(parts) > 15:
            conf_str = parts[15].upper()
            if conf_str in ("TO", "IC", "CR", "AP", "LD"):
                config = conf_str

        return PTDRow(
            fl=fl,
            temperature=float(parts[1]),
            pressure=float(parts[2]),
            density=float(parts[3]),
            speed_of_sound=float(parts[4]),
            tas=float(parts[5]),
            cas=float(parts[6]),
            mach=float(parts[7]),
            mass=float(parts[8]),
            thrust=float(parts[9]),
            drag=float(parts[10]),
            fuel_flow=float(parts[11]),
            f=f_val,
            rocd=rocd,
            gamma=gamma,
            config=config,
        )
    except (ValueError, IndexError):
        return None


def parse_ptd_file(filepath: str | Path) -> PTDData:
    """Parse a BADA PTD file.

    Args:
        filepath: Path to the PTD file.

    Returns:
        PTDData containing all parsed sections and rows.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    filepath = Path(filepath)

    with filepath.open(encoding="utf-8") as f:
        content = f.read()

    sections = []

    # Find section headers (e.g., "Low mass CLIMBS", "Medium mass CLIMB")
    # Pattern matches lines like "Low mass CLIMBS" or "Medium mass CLIMB"
    section_pattern = re.compile(
        r"^((?:Low|Medium|High)\s+mass\s+(?:CLIMB|CRUISE|DESCENT|CLIMBS|DESCENTS?))\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    matches = list(section_pattern.finditer(content))

    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start_pos = match.end()

        # Find end of section (next section or end of file)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)

        section_content = content[start_pos:end_pos]

        # Parse rows from section
        rows = []
        for line in section_content.split("\n"):
            line = line.strip()
            if not line:
                continue

            row = _parse_data_row(line)
            if row is not None:
                rows.append(row)

        if rows:
            sections.append(
                PTDSection(
                    name=section_name,
                    mass_category=_parse_mass_category(section_name),
                    phase=_parse_phase(section_name),
                    rows=rows,
                )
            )

    return PTDData(filename=filepath.name, sections=sections)
