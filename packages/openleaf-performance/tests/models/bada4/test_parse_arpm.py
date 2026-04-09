from __future__ import annotations

import xml.etree.ElementTree as ET

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.core.types import LandingGearPosition
from openleaf.performance.models.bada4.parsing import parse_arpm
from openleaf.performance.models.bada4.schemas import CVMIN_DEFAULT
from openleaf.performance.models.bada4.schemas import CVMIN_TO_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_AP_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_IC_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_LD_DEFAULT
from openleaf.performance.models.bada4.schemas import HMAX_TO_DEFAULT
from openleaf.performance.models.bada4.schemas import BADA4ARPMConfig


def _xml(text: str) -> ET.Element:
    """Parse an XML string into an Element."""
    return ET.fromstring(text)  # noqa: S314


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_parse_arpm_none() -> None:
    """parse_arpm(None) returns default BADA4ARPMConfig."""
    result = parse_arpm(None)
    assert result == BADA4ARPMConfig()


def test_parse_arpm_altitude_thresholds() -> None:
    """XML with AeroConfSchedule containing Hmax values parses correctly."""
    xml = _xml(
        "<ARPM>"
        "  <AeroConfSchedule>"
        "    <AeroPhase><name>TO</name><Hmax>1500.0</Hmax></AeroPhase>"
        "    <AeroPhase><name>IC</name><Hmax>2000.0</Hmax></AeroPhase>"
        "    <AeroPhase><name>AP</name><Hmax>8000.0</Hmax></AeroPhase>"
        "    <AeroPhase><name>LD</name><Hmax>3000.0</Hmax></AeroPhase>"
        "  </AeroConfSchedule>"
        "</ARPM>"
    )
    result = parse_arpm(xml)
    assert result.hmax_to == 1500.0
    assert result.hmax_ic == 2000.0
    assert result.hmax_ap == 8000.0
    assert result.hmax_ld == 3000.0


def test_parse_arpm_speed_schedule() -> None:
    """XML with SpeedSchedule containing CVmin parses correctly."""
    xml = _xml(
        "<ARPM>"
        "  <SpeedSchedule>"
        "    <SpeedPhase><name>TO</name><CVmin>1.13</CVmin></SpeedPhase>"
        "    <SpeedPhase><name>IC</name><CVmin>1.23</CVmin></SpeedPhase>"
        "  </SpeedSchedule>"
        "</ARPM>"
    )
    result = parse_arpm(xml)
    assert result.cvmin_to == 1.13
    assert result.cvmin == 1.23


def test_parse_arpm_hlid_lg_mapping() -> None:
    """XML with AeroPhase entries having HLid and LG parses correctly."""
    xml = _xml(
        "<ARPM>"
        "  <AeroConfSchedule>"
        "    <AeroPhase>"
        "      <name>TO</name><HLid>CONF1+F</HLid><LG>UP</LG>"
        "    </AeroPhase>"
        "    <AeroPhase>"
        "      <name>LD</name><HLid>FULL</HLid><LG>DN</LG>"
        "    </AeroPhase>"
        "  </AeroConfSchedule>"
        "</ARPM>"
    )
    result = parse_arpm(xml)
    assert result.hlid_map[AerodynamicConfig.TO] == "CONF1+F"
    assert result.hlid_map[AerodynamicConfig.LD] == "FULL"
    assert result.lg_map[AerodynamicConfig.TO] == LandingGearPosition.UP
    assert result.lg_map[AerodynamicConfig.LD] == LandingGearPosition.DOWN


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_parse_arpm_missing_aero_conf_schedule() -> None:
    """ARPM without AeroConfSchedule yields default thresholds and empty maps."""
    xml = _xml("<ARPM></ARPM>")
    result = parse_arpm(xml)
    assert result.hmax_to == HMAX_TO_DEFAULT
    assert result.hmax_ic == HMAX_IC_DEFAULT
    assert result.hmax_ap == HMAX_AP_DEFAULT
    assert result.hmax_ld == HMAX_LD_DEFAULT
    assert result.hlid_map == {}
    assert result.lg_map == {}


def test_parse_arpm_missing_speed_schedule() -> None:
    """ARPM without SpeedSchedule yields default CVmin values."""
    xml = _xml("<ARPM></ARPM>")
    result = parse_arpm(xml)
    assert result.cvmin_to == CVMIN_TO_DEFAULT
    assert result.cvmin == CVMIN_DEFAULT


def test_parse_arpm_unknown_phase_name() -> None:
    """Unknown phase name in AeroPhase is skipped without error."""
    xml = _xml(
        "<ARPM>"
        "  <AeroConfSchedule>"
        "    <AeroPhase>"
        "      <name>XX</name><HLid>CONF1</HLid><LG>UP</LG><Hmax>999.0</Hmax>"
        "    </AeroPhase>"
        "  </AeroConfSchedule>"
        "</ARPM>"
    )
    result = parse_arpm(xml)
    assert result.hlid_map == {}
    assert result.lg_map == {}
    # Hmax for unknown phase is ignored
    assert result.hmax_to == HMAX_TO_DEFAULT
    assert result.hmax_ic == HMAX_IC_DEFAULT
    assert result.hmax_ap == HMAX_AP_DEFAULT
    assert result.hmax_ld == HMAX_LD_DEFAULT


def test_parse_arpm_missing_name_element() -> None:
    """AeroPhase without <name> is skipped."""
    xml = _xml(
        "<ARPM>"
        "  <AeroConfSchedule>"
        "    <AeroPhase><HLid>CONF1</HLid><LG>UP</LG></AeroPhase>"
        "  </AeroConfSchedule>"
        "</ARPM>"
    )
    result = parse_arpm(xml)
    assert result.hlid_map == {}
    assert result.lg_map == {}


def test_parse_arpm_partial_aero_phase() -> None:
    """AeroPhase with HLid but missing LG maps HLid only."""
    xml = _xml(
        "<ARPM>"
        "  <AeroConfSchedule>"
        "    <AeroPhase>"
        "      <name>IC</name><HLid>CONF2</HLid>"
        "    </AeroPhase>"
        "  </AeroConfSchedule>"
        "</ARPM>"
    )
    result = parse_arpm(xml)
    assert result.hlid_map[AerodynamicConfig.IC] == "CONF2"
    assert AerodynamicConfig.IC not in result.lg_map
