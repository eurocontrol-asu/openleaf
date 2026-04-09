from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock


if TYPE_CHECKING:
    from pathlib import Path

import pytest

from openleaf.performance.models.bada4.synonyms import available_bada4_aircraft


# ---------------------------------------------------------------------------
# Unit tests — _available_from_synonyms
# ---------------------------------------------------------------------------


def test_available_from_synonyms_dir_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ICAO included when backing folder exists."""
    # Setup: create a directory matching the synonym entry
    (tmp_path / "A320_data").mkdir()

    synonyms = {"A320": MagicMock(folder_name="A320_data")}
    monkeypatch.setattr(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        lambda _: synonyms,
    )

    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    result = available_bada4_aircraft(tmp_path, synonym_file=synonym_file)
    assert "A320" in result


def test_available_from_synonyms_xml_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ICAO included when .xml file exists (no folder)."""
    (tmp_path / "B738_data.xml").touch()

    synonyms = {"B738": MagicMock(folder_name="B738_data")}
    monkeypatch.setattr(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        lambda _: synonyms,
    )

    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    result = available_bada4_aircraft(tmp_path, synonym_file=synonym_file)
    assert "B738" in result


def test_available_from_synonyms_neither(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ICAO excluded when neither folder nor .xml exists."""
    synonyms = {"C172": MagicMock(folder_name="C172_data")}
    monkeypatch.setattr(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        lambda _: synonyms,
    )

    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    result = available_bada4_aircraft(tmp_path, synonym_file=synonym_file)
    assert "C172" not in result


def test_available_from_filesystem(tmp_path: Path) -> None:
    """Fallback returns stems from .xml files and directory names."""
    (tmp_path / "A320.xml").touch()
    (tmp_path / "B738").mkdir()

    # No synonym file → filesystem fallback
    result = available_bada4_aircraft(tmp_path, synonym_file=tmp_path / "nonexistent.bd4")
    assert "A320" in result
    assert "B738" in result


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def test_available_bada4_aircraft_with_synonym(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full path through synonym file: result is sorted and matches expected set."""
    synonyms = {
        "B738": MagicMock(folder_name="B738_data"),
        "A320": MagicMock(folder_name="A320_data"),
        "C172": MagicMock(folder_name="C172_data"),
    }
    # Create backing data for A320 (dir) and B738 (xml)
    (tmp_path / "A320_data").mkdir()
    (tmp_path / "B738_data.xml").touch()
    # C172 has no backing data

    monkeypatch.setattr(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        lambda _: synonyms,
    )

    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    result = available_bada4_aircraft(tmp_path, synonym_file=synonym_file)
    assert result == sorted(result)
    assert result == ["A320", "B738"]


def test_available_bada4_aircraft_fallback(tmp_path: Path) -> None:
    """Without SYNONYM.BD4, falls back to filesystem scan."""
    (tmp_path / "E170.xml").touch()
    (tmp_path / "A380").mkdir()

    # synonym_file=None → default path, which won't exist in tmp_path
    result = available_bada4_aircraft(tmp_path)
    assert "E170" in result
    assert "A380" in result
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_synonym_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SYNONYM.BD4 exists but parses to empty dict → empty list."""
    monkeypatch.setattr(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        lambda _: {},
    )

    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    result = available_bada4_aircraft(tmp_path, synonym_file=synonym_file)
    assert result == []


def test_no_matching_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All synonym entries point to nonexistent folders/files → empty list."""
    synonyms = {
        "A320": MagicMock(folder_name="missing_A320"),
        "B738": MagicMock(folder_name="missing_B738"),
    }
    monkeypatch.setattr(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        lambda _: synonyms,
    )

    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    result = available_bada4_aircraft(tmp_path, synonym_file=synonym_file)
    assert result == []


def test_mixed_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Some entries have folders, some XML, some neither → only matching ICAOs, sorted."""
    synonyms = {
        "C172": MagicMock(folder_name="C172_data"),  # will have dir
        "B738": MagicMock(folder_name="B738_data"),  # will have xml
        "A320": MagicMock(folder_name="A320_data"),  # neither
    }
    (tmp_path / "C172_data").mkdir()
    (tmp_path / "B738_data.xml").touch()

    monkeypatch.setattr(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        lambda _: synonyms,
    )

    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    result = available_bada4_aircraft(tmp_path, synonym_file=synonym_file)
    assert result == ["B738", "C172"]
    assert "A320" not in result
