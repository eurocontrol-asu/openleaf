from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch


if TYPE_CHECKING:
    from pathlib import Path

import pytest

from openleaf.performance.models.bada4.synonyms import BADA4SynonymEntry
from openleaf.performance.models.bada4.synonyms import resolve_bada4_file


def _make_entry(icao: str, folder_name: str) -> BADA4SynonymEntry:
    return BADA4SynonymEntry(
        icao=icao,
        manufacturer="TEST",
        model="TEST",
        folder_name=folder_name,
        is_direct=True,
        is_icao=True,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_resolve_bada4_synonym_folder(tmp_path: Path) -> None:
    """Synonym file + folder/name.xml -> returns correct path."""
    folder = tmp_path / "A320_model"
    folder.mkdir()
    xml = folder / "A320_model.xml"
    xml.touch()

    synonyms = {"A320": _make_entry("A320", "A320_model")}
    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    with patch(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        return_value=synonyms,
    ):
        result = resolve_bada4_file("A320", tmp_path, synonym_file)

    assert result == xml


def test_resolve_bada4_synonym_direct_xml(tmp_path: Path) -> None:
    """Synonym file + data_path/name.xml (no folder) -> returns correct path."""
    xml = tmp_path / "A320_model.xml"
    xml.touch()

    synonyms = {"A320": _make_entry("A320", "A320_model")}
    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    with patch(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        return_value=synonyms,
    ):
        result = resolve_bada4_file("A320", tmp_path, synonym_file)

    assert result == xml


def test_resolve_bada4_fallback_folder(tmp_path: Path) -> None:
    """No synonym file, folder matching ICAO -> returns XML from folder."""
    folder = tmp_path / "A320"
    folder.mkdir()
    xml = folder / "A320.xml"
    xml.touch()

    # No synonym file exists
    result = resolve_bada4_file("A320", tmp_path)

    assert result == xml


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_resolve_bada4_icao_lowercase(tmp_path: Path) -> None:
    """Lowercase ICAO is normalized to uppercase."""
    folder = tmp_path / "A320"
    folder.mkdir()
    xml = folder / "A320.xml"
    xml.touch()

    result = resolve_bada4_file("a320", tmp_path)

    assert result == xml


def test_resolve_bada4_icao_whitespace(tmp_path: Path) -> None:
    """ICAO with whitespace is stripped and resolved."""
    folder = tmp_path / "A320"
    folder.mkdir()
    xml = folder / "A320.xml"
    xml.touch()

    result = resolve_bada4_file(" A320 ", tmp_path)

    assert result == xml


def test_resolve_bada4_no_synonym_file(tmp_path: Path) -> None:
    """No synonym file -> falls through to filesystem scan."""
    xml = tmp_path / "A320.xml"
    xml.touch()

    result = resolve_bada4_file("A320", tmp_path)

    assert result == xml


def test_resolve_bada4_icao_not_in_synonyms(tmp_path: Path) -> None:
    """Valid synonym file but ICAO not listed -> falls through to filesystem scan."""
    folder = tmp_path / "B738"
    folder.mkdir()
    xml = folder / "B738.xml"
    xml.touch()

    synonyms = {"A320": _make_entry("A320", "A320_model")}
    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    with patch(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        return_value=synonyms,
    ):
        result = resolve_bada4_file("B738", tmp_path, synonym_file)

    assert result == xml


def test_resolve_bada4_no_match_anywhere(tmp_path: Path) -> None:
    """ICAO not found by any strategy -> FileNotFoundError with available list."""
    with pytest.raises(FileNotFoundError, match="Aircraft 'ZZZZ' not found in BADA4"):
        resolve_bada4_file("ZZZZ", tmp_path)


def test_resolve_bada4_empty_data_directory(tmp_path: Path) -> None:
    """No XML/OPF files -> FileNotFoundError with empty available list."""
    with pytest.raises(FileNotFoundError, match="Aircraft 'A320' not found in BADA4"):
        resolve_bada4_file("A320", tmp_path)


def test_resolve_bada4_multiple_xmls_in_folder(tmp_path: Path) -> None:
    """Folder has multiple .xml files -> returns first (sorted by glob)."""
    folder = tmp_path / "A320"
    folder.mkdir()
    # Named xml does NOT match folder name, so fallback to glob
    (folder / "alpha.xml").touch()
    (folder / "beta.xml").touch()
    (folder / "gamma.xml").touch()

    result = resolve_bada4_file("A320", tmp_path)

    # glob returns files; the first one found is returned
    assert result.suffix == ".xml"
    assert result.parent == folder


def test_resolve_bada4_synonym_folder_any_xml(tmp_path: Path) -> None:
    """Synonym folder exists but named xml missing -> returns any xml in folder."""
    folder = tmp_path / "A320_model"
    folder.mkdir()
    xml = folder / "other.xml"
    xml.touch()

    synonyms = {"A320": _make_entry("A320", "A320_model")}
    synonym_file = tmp_path / "SYNONYM.BD4"
    synonym_file.touch()

    with patch(
        "openleaf.performance.models.bada4.synonyms._get_synonyms",
        return_value=synonyms,
    ):
        result = resolve_bada4_file("A320", tmp_path, synonym_file)

    assert result == xml
