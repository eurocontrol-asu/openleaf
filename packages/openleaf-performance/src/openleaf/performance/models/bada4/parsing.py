"""
BADA4 (Base of Aircraft Data 4) XML parsing.

Provides parsing functions for BADA4 XML parameter files.

Reference: EUROCONTROL BADA Family 4 User Manual
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy.typing as npt

import openleaf.performance.numpy as np
import openleaf.performance.physics.isa as isa
from openleaf.performance.core.perfos import EngineSetting
from openleaf.performance.core.perfos import EngineType
from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import LandingGearPosition

# Import limits model from limits module
from openleaf.performance.models.bada4.limits import BADA4Limits

# Import data classes from schemas module
from openleaf.performance.models.bada4.schemas import CVMIN_DEFAULT
from openleaf.performance.models.bada4.schemas import CVMIN_TO_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_AP_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_IC_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_LD_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_TO_DEFAULT
from openleaf.performance.models.bada4.schemas import BADA4AerodynamicsData
from openleaf.performance.models.bada4.schemas import BADA4AircraftData
from openleaf.performance.models.bada4.schemas import BADA4ARPMConfig
from openleaf.performance.models.bada4.schemas import BADA4BuffetLimitsData
from openleaf.performance.models.bada4.schemas import BADA4Data
from openleaf.performance.models.bada4.schemas import BADA4EngineData
from openleaf.performance.models.bada4.schemas import BADA4GeometricLimitsData
from openleaf.performance.models.bada4.schemas import BADA4JetEngineData
from openleaf.performance.models.bada4.schemas import BADA4JetOrTurbopropEngineData
from openleaf.performance.models.bada4.schemas import BADA4KinematicLimitsData
from openleaf.performance.models.bada4.schemas import BADA4LimitsData
from openleaf.performance.models.bada4.schemas import BADA4PistonEngineData
from openleaf.performance.models.bada4.schemas import BADA4TurbopropEngineData
from openleaf.performance.models.bada4.schemas import BADA4TurbopropOrPistonEngineData


__all__ = [
    # Re-export constants for backwards compatibility
    "CVMIN_DEFAULT",
    "CVMIN_TO_DEFAULT",
    "HMAX_AP_DEFAULT",
    "HMAX_IC_DEFAULT",
    "HMAX_LD_DEFAULT",
    "HMAX_TO_DEFAULT",
    # Re-export data classes for backwards compatibility
    "BADA4ARPMConfig",
    "BADA4AerodynamicsData",
    "BADA4AircraftData",
    "BADA4BuffetLimitsData",
    "BADA4Data",
    "BADA4EngineData",
    "BADA4GeometricLimitsData",
    "BADA4JetEngineData",
    "BADA4JetOrTurbopropEngineData",
    "BADA4KinematicLimitsData",
    "BADA4Limits",
    "BADA4LimitsData",
    "BADA4PistonEngineData",
    "BADA4TurbopropEngineData",
    "BADA4TurbopropOrPistonEngineData",
    # Parsing functions
    "parse_aero_data",
    "parse_aircraft_data",
    "parse_arpm",
    "parse_bada4_file",
    "parse_base_engine_data",
    "parse_jet_engine_data",
    "parse_limits",
    "parse_piston_engine_data",
    "parse_propeller_engine_data",
    "parse_turbine_engine_data",
    "parse_turboprop_engine_data",
]


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================


def parse_aircraft_data(root: ET.Element) -> BADA4AircraftData:
    """Parse aircraft data from BADA4 XML file.

    Args:
        root: XML root element.

    Returns:
        Aircraft data with specifications and performance limits.
    """
    model = root.find("model").text
    engine_type_str = root.find("type").text
    engine_type = EngineType.from_string(engine_type_str.upper())

    icao = root.find("ICAO")
    designator = icao.find("designator").text
    wtc = icao.find("WTC").text

    alm = root.find("ALM")
    dlm = alm.find("DLM")
    klm = alm.find("KLM")
    glm = alm.find("GLM")

    oew = float(dlm.find("OEW").text)
    mtow = float(dlm.find("MTOW").text)

    mpl_elem = dlm.find("MPL")
    mpl = float(mpl_elem.text) if mpl_elem is not None else 0.0

    vmo = float(klm.find("vmo").text)

    mmo_elem = klm.find("mmo")
    mmo = float(mmo_elem.text) if mmo_elem is not None else None

    hmo = float(glm.find("hmo").text)

    return BADA4AircraftData(
        model=model,
        engine_type=engine_type,
        designator=designator,
        wtc=wtc,
        vmo=vmo,
        mmo=mmo,
        hmo=hmo,
        oem=oew,
        mtom=mtow,
        mpl=mpl,
    )


def parse_base_engine_data(root: ET.Element) -> BADA4EngineData:
    """Parse base engine data common to all engine types.

    Args:
        root: XML performance data element.

    Returns:
        Base engine data with reference weight, LHV, and engine count.
    """
    wref = float(root.find("MREF").text) * isa.G
    lhv = float(root.find("LHV").text)
    engines = int(root.find("n_eng").text)

    return BADA4EngineData(
        wref=wref,
        engines=engines,
        lhv=lhv,
    )


def parse_turbine_engine_data(
    root: ET.Element,
) -> BADA4JetOrTurbopropEngineData:
    """Parse turbine engine coefficients for thrust and fuel flow.

    Args:
        root: XML turbine performance element.

    Returns:
        Turbine engine coefficients (a, f, fi, ti arrays).
    """
    cx = root.find("CT")

    if cx is None:
        cx = root.find("CP")

    a = np.array([float(x.text) for x in cx.findall("a")])
    f = np.array([float(x.text) for x in root.find("CF").findall("f")])

    lidl = root.find("LIDL")
    ti = np.array([float(x.text) for x in lidl.find("CT").findall("ti")])
    fi = np.array([float(x.text) for x in lidl.find("CF").findall("fi")])

    return BADA4JetOrTurbopropEngineData(
        a=a,
        f=f,
        fi=fi,
        ti=ti,
    )


def parse_propeller_engine_data(
    root: ET.Element,
) -> BADA4TurbopropOrPistonEngineData:
    """Parse propeller engine data for turboprop and piston engines.

    Args:
        root: XML propeller performance element.

    Returns:
        Propeller data including diameter and maximum efficiency.
    """
    prop_dia = float(root.find("prop_dia").text)
    max_eff = float(root.find("max_eff").text)

    return BADA4TurbopropOrPistonEngineData(
        prop_dia=prop_dia,
        max_eff=max_eff,
    )


def parse_jet_engine_data(root: ET.Element) -> BADA4JetEngineData:
    """Parse jet engine data from BADA4 XML.

    Args:
        root: XML performance model element for jet engines.

    Returns:
        Complete jet engine data with thrust and fuel flow coefficients.
    """
    # Base engine data
    base = parse_base_engine_data(root)

    # Turbine data
    tfm = root.find("TFM")
    jet_turboprop = parse_turbine_engine_data(tfm)

    # Jet-specific data
    c = {}
    b = {}
    kink = {}

    for s in EngineSetting:
        name = s.name
        eng = tfm.find(name)

        if eng is not None:
            if eng.find("kink") is not None:
                kink[s] = float(eng.find("kink").text)

            if eng.find("flat_rating") is not None:
                b[s] = np.array([float(x.text) for x in eng.find("flat_rating").findall("b")])

            if eng.find("temp_rating") is not None:
                c[s] = np.array([float(x.text) for x in eng.find("temp_rating").findall("c")])

    return BADA4JetEngineData(
        **base.model_dump(),  # unpack all fields from the base parser
        **jet_turboprop.model_dump(),
        b=b,
        c=c,
        kink=kink,
    )


def parse_turboprop_engine_data(
    root: ET.Element,
) -> BADA4TurbopropEngineData:
    """Parse turboprop engine data from BADA4 XML.

    Args:
        root: XML performance model element for turboprop engines.

    Returns:
        Complete turboprop engine data with thrust and fuel flow coefficients.
    """
    # Base engine data
    base = parse_base_engine_data(root)

    # Turbine data
    tpm = root.find("TPM")
    jet_turboprop = parse_turbine_engine_data(tpm)

    # Propeller data
    turborpop_piston = parse_propeller_engine_data(tpm)

    # Turboprop-spcieific data
    p = {}
    max_power = {}

    for s in EngineSetting:
        name = s.name
        eng = tpm.find(name)

        if eng is not None:
            if eng.find("max_power") is not None:
                max_power[s] = float(eng.find("max_power").text)

            if eng.find("rating") is not None:
                p[s] = np.array([float(x.text) for x in eng.find("rating").findall("p")])

    return BADA4TurbopropEngineData(
        **base.model_dump(),  # unpack all fields from the base parser
        **jet_turboprop.model_dump(),
        **turborpop_piston.model_dump(),
        p=p,
        max_power=max_power,
    )


def parse_piston_engine_data(
    root: ET.Element,
) -> BADA4PistonEngineData:
    """Parse piston engine data from BADA4 XML.

    Args:
        root: XML performance model element for piston engines.

    Returns:
        Complete piston engine data with power and fuel flow coefficients.
    """
    # Base engine data
    base = parse_base_engine_data(root)

    # Propeller common data
    pem = root.find("PEM")
    turborpop_piston = parse_propeller_engine_data(pem)

    # Piston-specific data
    hd_turbo = float(pem.find("Hd_turbo").text)
    cpsfc = float(pem.find("CPSFC").text)
    p = float(pem.find("P").text)

    return BADA4PistonEngineData(
        **base.model_dump(),  # unpack all fields from the base parser,
        **turborpop_piston.model_dump(),
        hd_turbo=hd_turbo,
        cpsfc=cpsfc,
        p=p,
    )


def parse_aero_data(
    root: ET.Element,
    phases: ET.Element | None = None,
) -> BADA4AerodynamicsData:
    """Parse aerodynamic data from BADA4 XML.

    Args:
        root: XML aerodynamic coefficient model element.
        phases: Optional XML element (unused, kept for API compatibility).

    Returns:
        Aerodynamic data including drag coefficients and wing area.
        Non-clean coefficients are stored by (hlid, lg) tuple.
    """
    surface_area = float(root.find("S").text)

    d_clean = None
    d_nonclean: dict[tuple[str, LandingGearPosition], npt.NDArray] = {}
    mach_max = None
    scalar = None

    # Map XML element names to LandingGearPosition enum
    lg_xml_map = {
        "UP": LandingGearPosition.UP,
        "DN": LandingGearPosition.DOWN,
    }

    for c in root.findall("Configuration"):
        hlid = c.get("HLid")

        for lg_xml, lg_enum in lg_xml_map.items():
            lg = c.find(f"LG{lg_xml}")

            if lg is not None:
                dpm = lg.find("DPM_clean")

                if dpm is not None and d_clean is None:
                    d_clean = np.array([float(x.text) for x in dpm.find("CD_clean").findall("d")])

                    mach_max = float(dpm.find("M_max").text)
                    scalar = float(dpm.find("scalar").text)
                else:
                    dpm = lg.find("DPM_nonclean")
                    if dpm is not None:
                        d_nonclean[(hlid, lg_enum)] = np.array(
                            [float(x.text) for x in dpm.find("CD_nonclean").findall("d")]
                        )

    return BADA4AerodynamicsData(
        surface_area=surface_area,
        mach_max=mach_max,
        scalar=scalar,
        d_clean=d_clean,
        d_nonclean=d_nonclean,
    )


def _find_float(elem: ET.Element | None, tag: str, default: float | None = None) -> float | None:
    """Extract a float from a child element, returning default if missing."""
    if elem is None:
        return default
    child = elem.find(tag)
    if child is None:
        return default
    return float(child.text)


def _map_aero_conf_schedule(
    arpm: ET.Element | None,
    value_lookup: dict[str, float],
    *,
    exclude_cr: bool = False,
) -> dict[AerodynamicConfig, float]:
    """Map AerodynamicConfig to values using AeroConfSchedule.

    Iterates AeroPhase elements and maps each config to a value
    looked up by its HLid (for VFE) or (HLid, LGpos) key (for CL_max).

    Args:
        arpm: ARPM XML element containing AeroConfSchedule.
        value_lookup: Dict mapping key to float value.
        exclude_cr: If True, skip AerodynamicConfig.CR entries.
    """
    result: dict[AerodynamicConfig, float] = {}
    if arpm is None:
        return result
    conf_schedule = arpm.find("AeroConfSchedule")
    if conf_schedule is None:
        return result
    for phase in conf_schedule.findall("AeroPhase"):
        name_elem = phase.find("name")
        hl_elem = phase.find("HLid")
        if name_elem is None or hl_elem is None:
            continue
        conf = AerodynamicConfig.from_string(name_elem.text)
        if exclude_cr and conf == AerodynamicConfig.CR:
            continue
        hl = hl_elem.text
        if hl in value_lookup:
            result[conf] = value_lookup[hl]
    return result


def _map_aero_conf_schedule_lg(
    arpm: ET.Element | None,
    value_lookup: dict[tuple[str, LandingGearPosition], float],
) -> dict[AerodynamicConfig, float]:
    """Map AerodynamicConfig to CL_max using AeroConfSchedule with LG position."""
    result: dict[AerodynamicConfig, float] = {}
    if arpm is None:
        return result
    conf_schedule = arpm.find("AeroConfSchedule")
    if conf_schedule is None:
        return result
    for phase in conf_schedule.findall("AeroPhase"):
        hl = phase.find("HLid").text
        lg_str = phase.find("LG").text
        lg_pos = LandingGearPosition.from_string(lg_str)
        conf = AerodynamicConfig.from_string(phase.find("name").text)
        if (hl, lg_pos) in value_lookup:
            result[conf] = value_lookup[(hl, lg_pos)]
    return result


def _parse_geometric_limits(alm: ET.Element) -> BADA4GeometricLimitsData:
    """Parse Geometric Limits (GLM) - Section 6.1."""
    glm = alm.find("GLM")
    hmo = _find_float(glm, "hmo", default=0.0)
    mfa = _find_float(glm, "mfa")
    return BADA4GeometricLimitsData(hmo=hmo, mfa=mfa)


def _parse_kinematic_limits(
    alm: ET.Element,
    afcm: ET.Element | None,
    arpm: ET.Element | None,
) -> BADA4KinematicLimitsData:
    """Parse Kinematic Limits (KLM) - Section 6.2."""
    klm = alm.find("KLM")
    vmo = _find_float(klm, "vmo", default=0.0)
    mmo = _find_float(klm, "mmo")
    vle = _find_float(klm, "vle")
    mle = _find_float(klm, "mle")

    # Parse VFE speeds from AFCM configurations (tag is lowercase <vfe>)
    vfe: dict[str, float] = {}
    if afcm is not None:
        for config in afcm.findall("Configuration"):
            hl_id = config.get("HLid")
            if hl_id and hl_id != "0":
                vfe_val = _find_float(config, "vfe")
                if vfe_val is not None:
                    vfe[hl_id] = vfe_val

    vfe_by_config = _map_aero_conf_schedule(arpm, vfe, exclude_cr=True)

    return BADA4KinematicLimitsData(
        vmo=vmo, mmo=mmo, vle=vle, mle=mle, vfe=vfe, vfe_by_config=vfe_by_config
    )


def _parse_buffet_limits(
    alm: ET.Element,
    afcm: ET.Element | None,
    arpm: ET.Element | None,
) -> BADA4BuffetLimitsData:
    """Parse Buffet Limits (BLM) - Section 6.3."""
    blm = alm.find("BLM")
    bf = None
    mach_min = None
    mach_max = None
    cl_mach0 = None

    if blm is not None:
        bf_elems = blm.findall("bf")
        if bf_elems:
            bf = np.array([float(b.text) for b in bf_elems])
        mach_min = _find_float(blm, "Mmin")
        mach_max = _find_float(blm, "Mmax")
        cl_mach0 = _find_float(blm, "CL_Mach0")

    # Parse CL_max from AFCM configurations for non-clean
    lg_afcm_map = {
        "LGUP": LandingGearPosition.UP,
        "LGDN": LandingGearPosition.DOWN,
    }
    cl_max_nonclean: dict[tuple[str, LandingGearPosition], float] = {}
    if afcm is not None:
        for config in afcm.findall("Configuration"):
            hl_id = config.get("HLid", "0")
            for lg_tag, lg_enum in lg_afcm_map.items():
                cl_max_val = _find_float(config.find(lg_tag), "BLM/CL_max")
                if cl_max_val is not None:
                    cl_max_nonclean[(hl_id, lg_enum)] = cl_max_val

    cl_max_config = _map_aero_conf_schedule_lg(arpm, cl_max_nonclean)

    return BADA4BuffetLimitsData(
        bf=bf,
        mach_min=mach_min,
        mach_max=mach_max,
        cl_mach0=cl_mach0,
        cl_max_nonclean=cl_max_nonclean,
        cl_max_config=cl_max_config,
    )


def parse_limits(
    root: ET.Element,
    afcm: ET.Element | None = None,
    arpm: ET.Element | None = None,
    aircraft: BADA4AircraftData | None = None,
    surface_area: float = 0.0,
    cvmin: float = CVMIN_DEFAULT,
) -> BADA4Limits | None:
    """Parse aircraft limitations from BADA4 XML.

    Args:
        root: XML root element containing ALM.
        afcm: Optional AFCM element for buffet CL_max values.
        arpm: Optional ARPM element for AeroConfSchedule mapping.
        aircraft: Optional aircraft data for mass fields.
        surface_area: Wing reference area [m²] for stall speed calculation.
        cvmin: Minimum speed coefficient (V_min = CVmin * V_stall).

    Returns:
        BADA4Limits or None if ALM not found.
    """
    alm = root.find("ALM")
    if alm is None:
        return None

    geometric = _parse_geometric_limits(alm)
    kinematic = _parse_kinematic_limits(alm, afcm, arpm)
    buffet = _parse_buffet_limits(alm, afcm, arpm)

    limits_data = BADA4LimitsData(
        geometric=geometric,
        kinematic=kinematic,
        buffet=buffet,
    )

    oem = aircraft.oem if aircraft is not None else 0.0
    mtom = aircraft.mtom if aircraft is not None else 0.0
    mpl = aircraft.mpl if aircraft is not None else 0.0

    return BADA4Limits(
        limits_data,
        oem=oem,
        mtom=mtom,
        mpl=mpl,
        surface_area=surface_area,
        cvmin=cvmin,
    )


def parse_arpm(arpm: ET.Element | None) -> BADA4ARPMConfig:
    """Parse Airline Procedure Model configuration from BADA4 XML.

    Extracts altitude thresholds, minimum speed coefficients, and
    HLid/LG mappings for configuration selection logic (Section 8 / Appendix B).

    Args:
        arpm: XML ARPM element, or None to use defaults.

    Returns:
        ARPM configuration with altitude thresholds, speed coefficients,
        and HLid/LG mappings per aerodynamic configuration.
    """
    if arpm is None:
        return BADA4ARPMConfig()

    # Altitude thresholds
    hmax_to = HMAX_TO_DEFAULT
    hmax_ic = HMAX_IC_DEFAULT
    hmax_ap = HMAX_AP_DEFAULT
    hmax_ld = HMAX_LD_DEFAULT
    cvmin_to = CVMIN_TO_DEFAULT
    cvmin = CVMIN_DEFAULT

    # HLid and LG mappings per aerodynamic configuration
    hlid_map: dict[AerodynamicConfig, str] = {}
    lg_map: dict[AerodynamicConfig, LandingGearPosition] = {}

    # Parse altitude thresholds and HLid/LG mappings from AeroConfSchedule
    aero_sched = arpm.find("AeroConfSchedule")
    if aero_sched is not None:
        for phase in aero_sched.findall("AeroPhase"):
            name = phase.find("name")
            if name is None:
                continue
            phase_name = name.text

            # Parse HLid mapping
            hlid_elem = phase.find("HLid")
            if hlid_elem is not None:
                try:
                    conf = AerodynamicConfig.from_string(phase_name)
                    hlid_map[conf] = hlid_elem.text
                except ValueError:
                    pass  # Skip unknown phase names

            # Parse LG mapping
            lg_elem = phase.find("LG")
            if lg_elem is not None:
                try:
                    conf = AerodynamicConfig.from_string(phase_name)
                    lg_map[conf] = LandingGearPosition.from_string(lg_elem.text)
                except ValueError:
                    pass  # Skip unknown phase names

            # Parse altitude thresholds
            hmax_elem = phase.find("Hmax")
            if hmax_elem is not None:
                hmax_val = float(hmax_elem.text)
                if phase_name == "TO":
                    hmax_to = hmax_val
                elif phase_name == "IC":
                    hmax_ic = hmax_val
                elif phase_name == "AP":
                    hmax_ap = hmax_val
                elif phase_name == "LD":
                    hmax_ld = hmax_val

    # Parse minimum speed coefficients if available
    speed_sched = arpm.find("SpeedSchedule")
    if speed_sched is not None:
        for sched in speed_sched.findall("SpeedPhase"):
            name = sched.find("name")
            if name is None:
                continue
            phase_name = name.text

            cvmin_elem = sched.find("CVmin")
            if cvmin_elem is not None:
                cvmin_val = float(cvmin_elem.text)
                if phase_name == "TO":
                    cvmin_to = cvmin_val
                elif phase_name in ("IC", "CR", "AP", "LD"):
                    cvmin = cvmin_val

    return BADA4ARPMConfig(
        hmax_to=hmax_to,
        hmax_ic=hmax_ic,
        hmax_ap=hmax_ap,
        hmax_ld=hmax_ld,
        cvmin_to=cvmin_to,
        cvmin=cvmin,
        hlid_map=hlid_map,
        lg_map=lg_map,
    )


def parse_bada4_file(xml_path: str) -> BADA4Data:
    """Parse complete BADA4 aircraft data from XML file.

    Args:
        xml_path: Path to BADA4 XML file.

    Returns:
        Complete BADA4 data including aircraft, engine, aero, and limits.

    Raises:
        ValueError: If engine type is not recognized.
    """
    tree = ET.parse(xml_path)  # noqa: S314 — trusted local BADA4 data files
    root = tree.getroot()

    # Parse aircraft data
    aircraft = parse_aircraft_data(root)

    # Parse engine data
    if aircraft.engine_type == EngineType.JET:
        engine = parse_jet_engine_data(root.find("PFM"))
    elif aircraft.engine_type == EngineType.TURBOPROP:
        engine = parse_turboprop_engine_data(root.find("PFM"))
    elif aircraft.engine_type == EngineType.PISTON:
        engine = parse_piston_engine_data(root.find("PFM"))
    else:
        raise ValueError(f"Unknown engine type: {aircraft.engine_type}")

    # Parse aerodynamic data
    afcm = root.find("AFCM")
    arpm_elem = root.find("ARPM")
    aero_data = parse_aero_data(afcm, arpm_elem)

    # Parse ARPM configuration (needed for cvmin before limits)
    arpm = parse_arpm(arpm_elem)

    # Parse limitations (pass surface_area and cvmin for stall speed calculation)
    limits = parse_limits(
        root,
        afcm,
        arpm_elem,
        aircraft,
        surface_area=aero_data.surface_area,
        cvmin=arpm.cvmin,
    )

    return BADA4Data(
        aircraft=aircraft,
        engine=engine,
        aero=aero_data,
        limits=limits,
        arpm=arpm,
    )
