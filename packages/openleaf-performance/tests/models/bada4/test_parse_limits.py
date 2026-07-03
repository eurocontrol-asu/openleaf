from __future__ import annotations

import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DefusedET
import numpy as np

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.models.bada4.parsing import parse_limits
from openleaf.performance.models.bada4.schemas import BADA4BuffetLimitsData
from openleaf.performance.models.bada4.schemas import BADA4GeometricLimitsData
from openleaf.performance.models.bada4.schemas import BADA4KinematicLimitsData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root(xml_str: str) -> ET.Element:
    """Wrap XML string in a root element and parse."""
    return DefusedET.fromstring(f"<root>{xml_str}</root>")


def _afcm(*configs: tuple[str, str | None, dict | None]) -> ET.Element:
    """Build a minimal AFCM element.

    Each config is (HLid, vfe_value_or_None, lg_clmax_dict_or_None).
    lg_clmax_dict maps 'LGUP'/'LGDN' -> CL_max value.
    """
    afcm = ET.Element("AFCM")
    for hl_id, vfe_val, lg_clmax in configs:
        cfg = ET.SubElement(afcm, "Configuration", HLid=hl_id)
        if vfe_val is not None:
            vfe_elem = ET.SubElement(cfg, "vfe")
            vfe_elem.text = vfe_val
        if lg_clmax is not None:
            for lg_tag, cl_val in lg_clmax.items():
                lg = ET.SubElement(cfg, lg_tag)
                blm = ET.SubElement(lg, "BLM")
                cl_max = ET.SubElement(blm, "CL_max")
                cl_max.text = str(cl_val)
    return afcm


def _arpm(*phases: tuple[str, str, str]) -> ET.Element:
    """Build a minimal ARPM element with AeroConfSchedule.

    Each phase is (name, HLid, LG).
    """
    arpm = ET.Element("ARPM")
    schedule = ET.SubElement(arpm, "AeroConfSchedule")
    for name, hl_id, lg in phases:
        phase = ET.SubElement(schedule, "AeroPhase")
        n = ET.SubElement(phase, "name")
        n.text = name
        h = ET.SubElement(phase, "HLid")
        h.text = hl_id
        lg_elem = ET.SubElement(phase, "LG")
        lg_elem.text = lg
    return arpm


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestParseGeometricLimits:
    """Test geometric limits (GLM) parsing."""

    def test_parse_geometric_limits(self) -> None:
        """Minimal XML with GLM returns correct BADA4GeometricLimitsData."""
        root = _root("<ALM><GLM><hmo>41000</hmo></GLM></ALM>")
        result = parse_limits(root)
        assert result is not None
        geo = result.data.geometric
        assert isinstance(geo, BADA4GeometricLimitsData)
        assert geo.hmo == 41000.0
        assert geo.mfa is not None  # defaults to hmo via validator


class TestParseKinematicLimits:
    """Test kinematic limits (KLM) parsing with VFE."""

    def test_parse_kinematic_limits_with_vfe(self) -> None:
        """XML with KLM + AFCM configs + AeroConfSchedule returns correct vfe_by_config."""
        root = _root(
            "<ALM><GLM><hmo>41000</hmo></GLM><KLM><vmo>350</vmo><mmo>0.89</mmo></KLM></ALM>"
        )
        afcm = _afcm(
            ("0", None, None),
            ("1", "220", None),
            ("2", "200", None),
        )
        arpm = _arpm(
            ("IC", "1", "UP"),
            ("AP", "2", "UP"),
        )
        result = parse_limits(root, afcm=afcm, arpm=arpm)
        assert result is not None
        kin = result.data.kinematic
        assert isinstance(kin, BADA4KinematicLimitsData)
        assert kin.vmo == 350.0
        assert kin.mmo == 0.89
        assert kin.vfe == {"1": 220.0, "2": 200.0}
        assert AerodynamicConfig.IC in kin.vfe_by_config
        assert kin.vfe_by_config[AerodynamicConfig.IC] == 220.0
        assert AerodynamicConfig.AP in kin.vfe_by_config
        assert kin.vfe_by_config[AerodynamicConfig.AP] == 200.0


class TestParseBuffetLimits:
    """Test buffet limits (BLM) parsing with CL_max."""

    def test_parse_buffet_limits_with_cl_max(self) -> None:
        """XML with BLM + AFCM CL_max + AeroConfSchedule returns correct buffet data."""
        root = _root(
            "<ALM>"
            "<GLM><hmo>41000</hmo></GLM>"
            "<KLM><vmo>350</vmo></KLM>"
            "<BLM>"
            "<bf>0.1</bf><bf>0.2</bf><bf>0.3</bf><bf>0.4</bf><bf>0.5</bf>"
            "<Mmin>0.15</Mmin><Mmax>0.90</Mmax><CL_Mach0>1.5</CL_Mach0>"
            "</BLM>"
            "</ALM>"
        )
        afcm = _afcm(
            ("1", "220", {"LGUP": 1.8, "LGDN": 1.6}),
        )
        arpm = _arpm(
            ("IC", "1", "UP"),
            ("LD", "1", "DN"),
        )
        result = parse_limits(root, afcm=afcm, arpm=arpm)
        assert result is not None
        buf = result.data.buffet
        assert isinstance(buf, BADA4BuffetLimitsData)
        np.testing.assert_array_almost_equal(buf.bf, [0.1, 0.2, 0.3, 0.4, 0.5])
        assert buf.mach_min == 0.15
        assert buf.mach_max == 0.90
        assert buf.cl_mach0 == 1.5
        assert AerodynamicConfig.IC in buf.cl_max_config
        assert buf.cl_max_config[AerodynamicConfig.IC] == 1.8
        assert AerodynamicConfig.LD in buf.cl_max_config
        assert buf.cl_max_config[AerodynamicConfig.LD] == 1.6


class TestParseLimitsNoALM:
    """Test parse_limits returns None when ALM is absent."""

    def test_parse_limits_no_alm(self) -> None:
        """XML without ALM element returns None."""
        root = _root("<SomeOtherElement/>")
        result = parse_limits(root)
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case scenarios for parse_limits."""

    def test_missing_optional_glm_mfa(self) -> None:
        """GLM without mfa -> mfa is None (before validator default)."""
        root = _root("<ALM><GLM><hmo>35000</hmo></GLM><KLM><vmo>300</vmo></KLM></ALM>")
        result = parse_limits(root)
        assert result is not None
        # mfa defaults to hmo via model validator, but the raw parsed value was None
        geo = result.data.geometric
        assert geo.hmo == 35000.0
        # After validator, mfa == hmo
        assert geo.mfa == 35000.0

    def test_missing_optional_klm_fields(self) -> None:
        """KLM without mle and vle -> fields are None (before validator default)."""
        root = _root(
            "<ALM><GLM><hmo>35000</hmo></GLM><KLM><vmo>300</vmo><mmo>0.82</mmo></KLM></ALM>"
        )
        result = parse_limits(root)
        assert result is not None
        kin = result.data.kinematic
        # vle defaults to vmo, mle defaults to mmo via validator
        assert kin.vle == 300.0
        assert kin.mle == 0.82

    def test_missing_optional_blm_fields(self) -> None:
        """BLM without Mmin, Mmax, CL_Mach0 -> fields are None."""
        root = _root(
            "<ALM><GLM><hmo>35000</hmo></GLM><KLM><vmo>300</vmo></KLM>"
            "<BLM><bf>0.1</bf><bf>0.2</bf><bf>0.3</bf><bf>0.4</bf><bf>0.5</bf></BLM>"
            "</ALM>"
        )
        result = parse_limits(root)
        assert result is not None
        buf = result.data.buffet
        assert buf.mach_min is None
        assert buf.mach_max is None
        assert buf.cl_mach0 is None

    def test_no_afcm_provided(self) -> None:
        """afcm=None -> VFE dict and CL_max dict are empty."""
        root = _root("<ALM><GLM><hmo>35000</hmo></GLM><KLM><vmo>300</vmo></KLM></ALM>")
        result = parse_limits(root, afcm=None)
        assert result is not None
        assert result.data.kinematic.vfe == {}
        assert result.data.buffet.cl_max_nonclean == {}

    def test_no_arpm_provided(self) -> None:
        """arpm=None -> vfe_by_config and cl_max_config are empty."""
        root = _root("<ALM><GLM><hmo>35000</hmo></GLM><KLM><vmo>300</vmo></KLM></ALM>")
        afcm = _afcm(("1", "220", {"LGUP": 1.8}))
        result = parse_limits(root, afcm=afcm, arpm=None)
        assert result is not None
        assert result.data.kinematic.vfe_by_config == {}
        assert result.data.buffet.cl_max_config == {}

    def test_hlid_zero_skipped_in_vfe(self) -> None:
        """Configuration with HLid='0' is skipped for VFE (clean config)."""
        root = _root("<ALM><GLM><hmo>35000</hmo></GLM><KLM><vmo>300</vmo></KLM></ALM>")
        afcm = _afcm(
            ("0", "999", None),  # Should be skipped
            ("1", "220", None),
        )
        result = parse_limits(root, afcm=afcm)
        assert result is not None
        assert "0" not in result.data.kinematic.vfe
        assert "1" in result.data.kinematic.vfe
        assert result.data.kinematic.vfe["1"] == 220.0

    def test_unknown_phase_name_skipped(self) -> None:
        """AeroPhase with unrecognized name is skipped via AerodynamicConfig.from_string."""
        root = _root("<ALM><GLM><hmo>35000</hmo></GLM><KLM><vmo>300</vmo></KLM></ALM>")
        afcm = _afcm(("1", "220", None))
        # CR maps to AerodynamicConfig.CR which is filtered out in vfe_by_config
        arpm = _arpm(("CR", "1", "UP"))
        result = parse_limits(root, afcm=afcm, arpm=arpm)
        assert result is not None
        # CR is excluded from vfe_by_config
        assert AerodynamicConfig.CR not in result.data.kinematic.vfe_by_config
