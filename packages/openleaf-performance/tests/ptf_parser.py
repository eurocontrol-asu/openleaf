"""
Parser for BADA Performance Table Files (PTF).

PTF files contain pre-computed performance data including cruise fuel flow,
climb/descent TAS, ROCD (Rate of Climb/Descent), and fuel flow at various
flight levels. This data is used to validate the Total Energy Model (TEM).

PTF Format (BADA3 and BADA4):
- Cruise: TAS [kts], fuel lo/nom/hi [kg/min]
- Climb: TAS [kts], ROCD lo/nom/hi [fpm], fuel nom [kg/min]
- Descent: TAS [kts], ROCD [fpm], fuel nom [kg/min]
"""

__all__ = [
    "PTFClimbRow",
    "PTFCruiseRow",
    "PTFData",
    "PTFDescentRow",
    "PTFMassLevels",
    "parse_ptf_file",
]

import re
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict


class PTFCruiseRow(BaseModel):
    """Cruise performance data at a flight level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fl: int
    tas: float | None  # TAS [kts], None if not available at this FL
    fuel_lo: float | None  # Fuel flow at low mass [kg/min]
    fuel_nom: float | None  # Fuel flow at nominal mass [kg/min]
    fuel_hi: float | None  # Fuel flow at high mass [kg/min]


class PTFClimbRow(BaseModel):
    """Climb performance data at a flight level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fl: int
    tas: float  # TAS [kts]
    rocd_lo: float  # ROCD at low mass [fpm]
    rocd_nom: float  # ROCD at nominal mass [fpm]
    rocd_hi: float  # ROCD at high mass [fpm]
    fuel_nom: float  # Fuel flow at nominal mass [kg/min]


class PTFDescentRow(BaseModel):
    """Descent performance data at a flight level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fl: int
    tas: float  # TAS [kts]
    rocd_nom: float  # ROCD at nominal mass [fpm] (always available)
    rocd_lo: float | None  # ROCD at low mass [fpm] (BADA4 only)
    rocd_hi: float | None  # ROCD at high mass [fpm] (BADA4 only)
    fuel_nom: float  # Fuel flow at nominal mass [kg/min]


class PTFMassLevels(BaseModel):
    """Mass levels from PTF header."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float  # Low mass [kg]
    nominal: float  # Nominal mass [kg]
    high: float  # High mass [kg]


class PTFData(BaseModel):
    """Complete PTF file data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str
    aircraft_type: str
    mass_levels: PTFMassLevels
    max_altitude_ft: int
    cruise_rows: list[PTFCruiseRow]
    climb_rows: list[PTFClimbRow]
    descent_rows: list[PTFDescentRow]

    def get_climb_row(self, fl: int) -> PTFClimbRow | None:
        """Get climb data at a specific flight level."""
        for row in self.climb_rows:
            if row.fl == fl:
                return row
        return None

    def get_descent_row(self, fl: int) -> PTFDescentRow | None:
        """Get descent data at a specific flight level."""
        for row in self.descent_rows:
            if row.fl == fl:
                return row
        return None

    def get_cruise_row(self, fl: int) -> PTFCruiseRow | None:
        """Get cruise data at a specific flight level."""
        for row in self.cruise_rows:
            if row.fl == fl:
                return row
        return None


def _parse_float_or_none(s: str) -> float | None:
    """Parse a float value, returning None for empty or invalid strings."""
    s = s.strip()
    if not s or s == "(T)":  # (T) indicates above ceiling
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_mass_levels(content: str) -> PTFMassLevels:
    """Parse mass levels from PTF header."""
    low = nominal = high = 0.0

    # Pattern for "low - 46800"
    low_match = re.search(r"low\s*-\s*(\d+)", content, re.IGNORECASE)
    if low_match:
        low = float(low_match.group(1))

    nom_match = re.search(r"nominal\s*-\s*(\d+)", content, re.IGNORECASE)
    if nom_match:
        nominal = float(nom_match.group(1))

    high_match = re.search(r"high\s*-\s*(\d+)", content, re.IGNORECASE)
    if high_match:
        high = float(high_match.group(1))

    return PTFMassLevels(low=low, nominal=nominal, high=high)


def _parse_max_altitude(content: str) -> int:
    """Parse maximum altitude from PTF header."""
    match = re.search(r"Max Alt\.\s*\[ft\]:\s*(\d+)", content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def _parse_aircraft_type(content: str) -> str:
    """Parse aircraft type from PTF header."""
    match = re.search(r"AC/Type:\s*(\S+)", content)
    if match:
        return match.group(1)
    return "Unknown"


def _is_data_row(line: str) -> bool:
    """Check if a line is a data row (starts with FL number)."""
    parts = line.split("|")
    if len(parts) < 2:
        return False
    first_part = parts[0].strip()
    try:
        int(first_part)
        return True
    except ValueError:
        return False


def _parse_data_row(
    line: str, is_bada4: bool = False
) -> tuple[PTFCruiseRow | None, PTFClimbRow | None, PTFDescentRow | None]:
    """Parse a single data row from PTF file.

    Returns tuple of (cruise_row, climb_row, descent_row).
    """
    # Split by pipe character
    parts = line.split("|")
    if len(parts) < 4:
        return None, None, None

    # Extract FL from first part
    try:
        fl = int(parts[0].strip())
    except ValueError:
        return None, None, None

    cruise_row = None
    climb_row = None
    descent_row = None

    # Parse CRUISE section (part 1)
    cruise_part = parts[1].strip()
    if cruise_part:
        cruise_values = cruise_part.split()
        if len(cruise_values) >= 4:
            tas = _parse_float_or_none(cruise_values[0])
            fuel_lo = _parse_float_or_none(cruise_values[1])
            fuel_nom = _parse_float_or_none(cruise_values[2])
            fuel_hi = _parse_float_or_none(cruise_values[3])
            cruise_row = PTFCruiseRow(
                fl=fl, tas=tas, fuel_lo=fuel_lo, fuel_nom=fuel_nom, fuel_hi=fuel_hi
            )

    # Parse CLIMB section (part 2)
    climb_part = parts[2].strip()
    if climb_part:
        climb_values = climb_part.split()
        if len(climb_values) >= 5:
            tas = _parse_float_or_none(climb_values[0])
            rocd_lo = _parse_float_or_none(climb_values[1])
            rocd_nom = _parse_float_or_none(climb_values[2])
            rocd_hi = _parse_float_or_none(climb_values[3])
            fuel_nom = _parse_float_or_none(climb_values[4])
            if all(v is not None for v in [tas, rocd_lo, rocd_nom, rocd_hi, fuel_nom]):
                climb_row = PTFClimbRow(
                    fl=fl,
                    tas=tas,
                    rocd_lo=rocd_lo,
                    rocd_nom=rocd_nom,
                    rocd_hi=rocd_hi,
                    fuel_nom=fuel_nom,
                )

    # Parse DESCENT section (part 3)
    descent_part = parts[3].strip()
    if descent_part:
        descent_values = descent_part.split()
        if is_bada4 and len(descent_values) >= 5:
            # BADA4: TAS, ROCD_lo, ROCD_nom, ROCD_hi, fuel
            tas = _parse_float_or_none(descent_values[0])
            rocd_lo = _parse_float_or_none(descent_values[1])
            rocd_nom = _parse_float_or_none(descent_values[2])
            rocd_hi = _parse_float_or_none(descent_values[3])
            fuel_nom = _parse_float_or_none(descent_values[4])
            if all(v is not None for v in [tas, rocd_nom, fuel_nom]):
                descent_row = PTFDescentRow(
                    fl=fl,
                    tas=tas,
                    rocd_nom=rocd_nom,
                    rocd_lo=rocd_lo,
                    rocd_hi=rocd_hi,
                    fuel_nom=fuel_nom,
                )
        elif len(descent_values) >= 3:
            # BADA3: TAS, ROCD_nom, fuel
            tas = _parse_float_or_none(descent_values[0])
            rocd_nom = _parse_float_or_none(descent_values[1])
            fuel_nom = _parse_float_or_none(descent_values[2])
            if all(v is not None for v in [tas, rocd_nom, fuel_nom]):
                descent_row = PTFDescentRow(
                    fl=fl,
                    tas=tas,
                    rocd_nom=rocd_nom,
                    rocd_lo=None,
                    rocd_hi=None,
                    fuel_nom=fuel_nom,
                )

    return cruise_row, climb_row, descent_row


def parse_ptf_file(filepath: str | Path) -> PTFData:
    """Parse a BADA PTF file.

    Args:
        filepath: Path to the PTF file.

    Returns:
        PTFData containing all parsed performance data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    filepath = Path(filepath)

    with filepath.open(encoding="utf-8") as f:
        content = f.read()

    # Detect if BADA4 format (has lo/nom/hi in descent header)
    is_bada4 = "lo    nom    hi    nom" in content.lower().replace(" ", "")
    # More reliable check: BADA4 descent header has 5 columns
    if re.search(r"lo\s+nom\s+hi\s+nom\s*$", content, re.MULTILINE | re.IGNORECASE):
        is_bada4 = True

    aircraft_type = _parse_aircraft_type(content)
    mass_levels = _parse_mass_levels(content)
    max_altitude = _parse_max_altitude(content)

    cruise_rows = []
    climb_rows = []
    descent_rows = []

    for line in content.split("\n"):
        if not _is_data_row(line):
            continue

        cruise, climb, descent = _parse_data_row(line, is_bada4)
        if cruise is not None:
            cruise_rows.append(cruise)
        if climb is not None:
            climb_rows.append(climb)
        if descent is not None:
            descent_rows.append(descent)

    return PTFData(
        filename=filepath.name,
        aircraft_type=aircraft_type,
        mass_levels=mass_levels,
        max_altitude_ft=max_altitude,
        cruise_rows=cruise_rows,
        climb_rows=climb_rows,
        descent_rows=descent_rows,
    )
