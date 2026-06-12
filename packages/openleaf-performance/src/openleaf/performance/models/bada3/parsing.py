"""
BADA3 OPF file parsing.

Provides robust parsing for BADA3 Operations Performance Files (OPF)
per EUROCONTROL BADA 3.x User Manual.

Reference: EUROCONTROL Base of Aircraft Data (BADA) Family 3 User Manual
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Final

from openleaf.performance.core.perfos import EngineType
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.exceptions import BADA3ParseError

# Import data classes from schemas module
from openleaf.performance.models.bada3.schemas import BADA3AerodynamicsData
from openleaf.performance.models.bada3.schemas import BADA3AircraftData
from openleaf.performance.models.bada3.schemas import BADA3Data
from openleaf.performance.models.bada3.schemas import BADA3EngineData
from openleaf.performance.models.bada3.schemas import BADA3LimitsData


__all__ = [
    # Re-export data classes for backwards compatibility
    "BADA3AerodynamicsData",
    "BADA3AircraftData",
    "BADA3Data",
    "BADA3EngineData",
    "BADA3LimitsData",
    "BADA3ParseError",
    # Parsing functions
    "parse_bada3_file",
]


# =============================================================================
# Regex Patterns (compiled for performance)
# =============================================================================

# Scientific notation float: matches .123E+04, -.456E-02, +.789E00
_FLOAT_PATTERN: Final[str] = r"([-+]?\.\d+E[+-]?\d+)"
_FLOAT_RE: Final[re.Pattern] = re.compile(_FLOAT_PATTERN)

# Aircraft header: "CD A320 2 engines Jet M"
_AIRCRAFT_HEADER_RE: Final[re.Pattern] = re.compile(
    r"CD\s+(\w+)\s+(\d+)\s+engines\s+(\w+).*?(\b[HML]\b)"
)

# Configuration line: "CD 1 CR Clean .123E+02 .456E-02 .789E-01"
_CONFIG_LINE_RE: Final[re.Pattern] = re.compile(
    r"^\s*CD\s*\d+\s+"
    r"(CR|IC|TO|AP|LD)\s+"  # phase
    r"\S+\s+"  # config name (ignored)
    + _FLOAT_PATTERN
    + r"\s+"  # Vstall
    + _FLOAT_PATTERN
    + r"\s+"  # CD0
    + _FLOAT_PATTERN,  # CD2
    re.MULTILINE,
)

# Section extraction: captures content between title and next "CC" or EOF
_SECTION_TEMPLATE: Final[str] = r"{title}.*?\n(.*?)(?=\nCC|\Z)"


# =============================================================================
# Section Titles (per BADA3 OPF format)
# =============================================================================


class _Sections:
    """OPF section title constants."""

    WING_AREA: Final[str] = "Wing Area"
    CONFIG: Final[str] = "Configuration characteristics"
    GEAR: Final[str] = "Gear"
    MAX_THRUST: Final[str] = "Max climb thrust coefficients"
    IDLE_THRUST: Final[str] = "Desc(low)"
    TSFC: Final[str] = "Thrust Specific Fuel Consumption Coefficients"
    IDLE_FUEL: Final[str] = "Descent Fuel Flow Coefficients"
    CRUISE_CORR: Final[str] = "Cruise Corr"
    MASS: Final[str] = "Mass (t)"
    ENVELOPE: Final[str] = "Flight envelope"
    BUFFET: Final[str] = "Buffet"


# =============================================================================
# Parser Implementation
# =============================================================================


def _read_file(path: str | Path) -> str:
    """Read OPF file with encoding fallback.

    Args:
        path: Path to BADA3 OPF file.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If file does not exist.
        BADA3ParseError: If file cannot be read.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"BADA3 file not found: {path}")

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        warnings.warn(
            f"File '{path}' contains non-UTF-8 characters, using Latin-1.",
            UnicodeWarning,
            stacklevel=3,
        )
        return path.read_text(encoding="latin-1")


def _extract_section(title: str, text: str) -> str:
    """Extract named section content from OPF file.

    Args:
        title: Section title to find.
        text: Full OPF file content.

    Returns:
        Section content (text between title and next section).

    Raises:
        BADA3ParseError: If section not found.
    """
    pattern = _SECTION_TEMPLATE.format(title=re.escape(title))
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise BADA3ParseError(f"Section not found: '{title}'", section=title)
    return match.group(1)


def _parse_floats(pattern: str, text: str, section: str, count: int) -> list[float]:
    """Extract float values from text using regex.

    Args:
        pattern: Regex pattern with capture groups.
        text: Text to search.
        section: Section name for error messages.
        count: Expected number of values.

    Returns:
        List of parsed float values.

    Raises:
        BADA3ParseError: If pattern not found or wrong count.
    """
    match = re.search(pattern, text)
    if not match:
        raise BADA3ParseError(f"Expected {count} values, found none", section=section)

    values = [float(v) for v in match.groups()]
    if len(values) != count:
        raise BADA3ParseError(f"Expected {count} values, found {len(values)}", section=section)
    return values


def _parse_aircraft(text: str) -> BADA3AircraftData:
    """Parse aircraft identification data.

    Args:
        text: Full OPF file content.

    Returns:
        Aircraft data with model, WTC, engine type, and engine count.

    Raises:
        BADA3ParseError: If aircraft header not found or invalid.
    """
    match = _AIRCRAFT_HEADER_RE.search(text)
    if not match:
        raise BADA3ParseError("Aircraft header not found", section="Header")

    model, engines_str, engine_type_str, wtc = match.groups()

    try:
        engine_type = EngineType.from_string(engine_type_str.upper())
    except (ValueError, KeyError) as e:
        raise BADA3ParseError(f"Invalid engine type: {engine_type_str}", section="Header") from e

    return BADA3AircraftData(
        model=model,
        engines=int(engines_str),
        engine_type=engine_type,
        wtc=wtc,
    )


def _parse_aerodynamics(text: str) -> tuple[BADA3AerodynamicsData, dict]:
    """Parse aerodynamic data and stall speeds.

    Args:
        text: Full OPF file content.

    Returns:
        Tuple of (aerodynamics data, vstall dict).

    Raises:
        BADA3ParseError: If required sections not found.
    """
    # Wing area
    wing_section = _extract_section(_Sections.WING_AREA, text)
    match = _FLOAT_RE.search(wing_section)
    if not match:
        raise BADA3ParseError("Wing area value not found", section=_Sections.WING_AREA)
    surface_area = float(match.group(1))

    # Configuration data (drag polars and stall speeds)
    config_section = _extract_section(_Sections.CONFIG, text)

    cd_0: dict[AerodynamicConfig, float] = {}
    cd_2: dict[AerodynamicConfig, float] = {}
    vstall: dict[AerodynamicConfig, float] = {}

    for match in _CONFIG_LINE_RE.finditer(config_section):
        phase_str, vs_str, c0_str, c2_str = match.groups()
        config = AerodynamicConfig.from_string(phase_str)
        vstall[config] = float(vs_str)
        cd_0[config] = float(c0_str)
        cd_2[config] = float(c2_str)

    if not cd_0:
        raise BADA3ParseError("No configuration data found", section=_Sections.CONFIG)

    # Gear drag increment (added to landing configuration)
    gear_section = _extract_section(_Sections.GEAR, text)
    match = _FLOAT_RE.search(gear_section)
    if not match:
        raise BADA3ParseError("Gear drag value not found", section=_Sections.GEAR)
    cd_0_gear = float(match.group(1))

    if AerodynamicConfig.LD in cd_0:
        cd_0[AerodynamicConfig.LD] += cd_0_gear

    aero = BADA3AerodynamicsData(
        surface_area=surface_area,
        cd_0=cd_0,
        cd_2=cd_2,
    )
    return aero, vstall


def _parse_engine(text: str) -> BADA3EngineData:
    """Parse engine thrust and fuel flow coefficients.

    Args:
        text: Full OPF file content.

    Returns:
        Engine data with all coefficients.

    Raises:
        BADA3ParseError: If required sections not found.
    """
    # Max thrust coefficients (5 values)
    pattern_5 = r"\s+".join([_FLOAT_PATTERN] * 5)
    max_thrust_section = _extract_section(_Sections.MAX_THRUST, text)
    ct_1, ct_2, ct_3, ct_4, ct_5 = _parse_floats(
        pattern_5, max_thrust_section, _Sections.MAX_THRUST, 5
    )

    # Idle thrust coefficients (5 values).
    # OPF column order (per the section header itself):
    #   Desc(low)  Desc(high)  Desc level  Desc(app)  Desc(ld)
    idle_section = _extract_section(_Sections.IDLE_THRUST, text)
    ct_des_low, ct_des_high, h_des, ct_des_app, ct_des_ld = _parse_floats(
        pattern_5, idle_section, _Sections.IDLE_THRUST, 5
    )

    # Fuel flow coefficients
    pattern_2 = r"\s+".join([_FLOAT_PATTERN] * 2)
    tsfc_section = _extract_section(_Sections.TSFC, text)
    cf_1, cf_2 = _parse_floats(pattern_2, tsfc_section, _Sections.TSFC, 2)

    idle_fuel_section = _extract_section(_Sections.IDLE_FUEL, text)
    cf_3, cf_4 = _parse_floats(pattern_2, idle_fuel_section, _Sections.IDLE_FUEL, 2)

    cruise_section = _extract_section(_Sections.CRUISE_CORR, text)
    match = _FLOAT_RE.search(cruise_section)
    if not match:
        raise BADA3ParseError("Cruise correction not found", section=_Sections.CRUISE_CORR)
    cf_crz = float(match.group(1))

    return BADA3EngineData(
        cf_1=cf_1,
        cf_2=cf_2,
        cf_3=cf_3,
        cf_4=cf_4,
        cf_crz=cf_crz,
        ct_1=ct_1,
        ct_2=ct_2,
        ct_3=ct_3,
        ct_4=ct_4,
        ct_5=ct_5,
        ct_des_high=ct_des_high,
        ct_des_low=ct_des_low,
        h_des=h_des,
        ct_des_app=ct_des_app,
        ct_des_ld=ct_des_ld,
    )


def _parse_limits(
    text: str,
    ct_4: float,
    vstall: dict[AerodynamicConfig, float],
) -> BADA3LimitsData:
    """Parse flight envelope and buffet limits.

    Args:
        text: Full OPF file content.
        ct_4: Thrust temperature coefficient from engine data.
        vstall: Stall speeds by configuration from aerodynamics parsing.

    Returns:
        Limits data with envelope parameters.

    Raises:
        BADA3ParseError: If required sections not found.
    """
    # Mass data (5 values): first 4 are in tonnes (convert to kg), 5th is mass
    # gradient Gw in [ft/kg] (NOT in tonnes — it is a gradient, not a mass).
    # OPF comment: "reference  minimum  maximum  max payload  mass grad"
    pattern_5 = r"\s+".join([_FLOAT_PATTERN] * 5)
    mass_section = _extract_section(_Sections.MASS, text)
    mass_vals = _parse_floats(pattern_5, mass_section, _Sections.MASS, 5)
    mref_t, oem_t, mtom_t, mpl_t, gw_mass = mass_vals
    mref = mref_t * 1000.0
    oem = oem_t * 1000.0
    mtom = mtom_t * 1000.0
    mpl = mpl_t * 1000.0
    # hmax is defined at MTOM per BADA3 Section 3.5, so mmax = mtom
    mmax = mtom

    # Flight envelope (6 values preferred, 5 acceptable)
    envelope_section = _extract_section(_Sections.ENVELOPE, text)
    pattern_6 = r"\s+".join([_FLOAT_PATTERN] * 6)

    try:
        vmo, mmo, hmo, hmax, gt, gw = _parse_floats(
            pattern_6, envelope_section, _Sections.ENVELOPE, 6
        )
    except BADA3ParseError:
        # Fall back to 5 values; use mass-section gradient as Gw [ft/kg]
        vmo, mmo, hmo, hmax, gt = _parse_floats(pattern_5, envelope_section, _Sections.ENVELOPE, 5)
        gw = gw_mass

    # Buffet coefficients. The "Buffet" title sits on the shared aero header
    # ("Wing Area and Buffet coefficients"), whose data line reads:
    #   ndrst  S  Clbo  k  CM16
    # ndrst is a plain integer (not matched by the sci-float pattern), so the
    # scientific floats on the line are S, Clbo, k, CM16 — skip S, take 2 & 3.
    pattern_3 = r"\s+".join([_FLOAT_PATTERN] * 3)
    buffet_section = _extract_section(_Sections.BUFFET, text)
    _surface_area, clbo, k = _parse_floats(pattern_3, buffet_section, _Sections.BUFFET, 3)

    return BADA3LimitsData(
        vmo=vmo,
        mmo=mmo,
        hmo=hmo,
        oem=oem,
        mtom=mtom,
        mpl=mpl,
        mref=mref,
        mmax=mmax,
        hmax=hmax,
        gt=gt,
        gw=gw,
        clbo=clbo,
        k=k,
        ct_4=ct_4,
        vstall=vstall,
    )


def parse_bada3_file(file: str | Path) -> BADA3Data:
    """Parse complete BADA3 aircraft data from OPF file.

    Args:
        file: Path to BADA3 OPF file.

    Returns:
        Complete BADA3 data with aircraft, aerodynamics, engine, and limits.

    Raises:
        FileNotFoundError: If file does not exist.
        BADA3ParseError: If parsing fails.

    Example:
        >>> data = parse_bada3_file("/path/to/A320__.OPF")
        >>> print(data.aircraft.model)
        'A320'
        >>> print(data.limits.mtom)
        77000.0
    """
    text = _read_file(file)

    aircraft = _parse_aircraft(text)
    aerodynamics, vstall = _parse_aerodynamics(text)
    engine = _parse_engine(text)
    limits = _parse_limits(text, ct_4=engine.ct_4, vstall=vstall)

    return BADA3Data(
        aircraft=aircraft,
        aerodynamics=aerodynamics,
        engine=engine,
        limits=limits,
    )
