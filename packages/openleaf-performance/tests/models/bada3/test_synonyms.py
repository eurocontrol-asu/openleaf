from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch


if TYPE_CHECKING:
    from pathlib import Path

import pytest

from openleaf.performance.models.bada3.synonyms import BADA3SynonymEntry
from openleaf.performance.models.bada3.synonyms import resolve_bada3_file


def _make_entry(icao: str, file_prefix: str) -> BADA3SynonymEntry:
    return BADA3SynonymEntry(
        icao=icao,
        manufacturer="TEST",
        model="TEST",
        file_prefix=file_prefix,
        is_direct=True,
        is_icao=True,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_resolve_bada3_synonym_exact(tmp_path: Path) -> None:
    """Synonym file + prefix.OPF -> returns correct path."""
    opf = tmp_path / "A320__.OPF"
    opf.touch()

    synonyms = {"A320": _make_entry("A320", "A320__")}
    synonym_file = tmp_path / "SYNONYM.NEW"
    synonym_file.touch()

    with patch(
        "openleaf.performance.models.bada3.synonyms._get_synonyms",
        return_value=synonyms,
    ):
        result = resolve_bada3_file("A320", tmp_path, synonym_file)

    assert result == opf


def test_resolve_bada3_synonym_underscore(tmp_path: Path) -> None:
    """Synonym file + prefix with underscore variant -> returns correct path."""
    # The exact prefix doesn't exist, but the underscore variant does
    opf = tmp_path / "A320___.OPF"
    opf.touch()

    synonyms = {"A320": _make_entry("A320", "A320__")}
    synonym_file = tmp_path / "SYNONYM.NEW"
    synonym_file.touch()

    with patch(
        "openleaf.performance.models.bada3.synonyms._get_synonyms",
        return_value=synonyms,
    ):
        result = resolve_bada3_file("A320", tmp_path, synonym_file)

    assert result == opf


def test_resolve_bada3_fallback(tmp_path: Path) -> None:
    """No synonym file, ICAO__.OPF exists -> returns correct path."""
    opf = tmp_path / "A320__.OPF"
    opf.touch()

    result = resolve_bada3_file("A320", tmp_path)

    assert result == opf


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_resolve_bada3_icao_lowercase(tmp_path: Path) -> None:
    """Lowercase ICAO is normalized to uppercase."""
    opf = tmp_path / "A320__.OPF"
    opf.touch()

    result = resolve_bada3_file("a320", tmp_path)

    assert result == opf


def test_resolve_bada3_icao_whitespace(tmp_path: Path) -> None:
    """ICAO with whitespace is stripped and resolved."""
    opf = tmp_path / "A320__.OPF"
    opf.touch()

    result = resolve_bada3_file(" A320 ", tmp_path)

    assert result == opf


def test_resolve_bada3_no_synonym_file(tmp_path: Path) -> None:
    """No synonym file -> falls through to filesystem scan."""
    opf = tmp_path / "A320_.OPF"
    opf.touch()

    result = resolve_bada3_file("A320", tmp_path)

    assert result == opf


def test_resolve_bada3_icao_not_in_synonyms(tmp_path: Path) -> None:
    """Valid synonym file but ICAO not listed -> falls through to filesystem scan."""
    opf = tmp_path / "B738__.OPF"
    opf.touch()

    synonyms = {"A320": _make_entry("A320", "A320__")}
    synonym_file = tmp_path / "SYNONYM.NEW"
    synonym_file.touch()

    with patch(
        "openleaf.performance.models.bada3.synonyms._get_synonyms",
        return_value=synonyms,
    ):
        result = resolve_bada3_file("B738", tmp_path, synonym_file)

    assert result == opf


def test_resolve_bada3_no_match_anywhere(tmp_path: Path) -> None:
    """ICAO not found by any strategy -> FileNotFoundError with available list."""
    with pytest.raises(FileNotFoundError, match="Aircraft 'ZZZZ' not found in BADA3"):
        resolve_bada3_file("ZZZZ", tmp_path)


def test_resolve_bada3_empty_data_directory(tmp_path: Path) -> None:
    """No OPF files -> FileNotFoundError with empty available list."""
    with pytest.raises(FileNotFoundError, match="Aircraft 'A320' not found in BADA3"):
        resolve_bada3_file("A320", tmp_path)


def test_resolve_bada3_synonym_fallback_underscore_order(tmp_path: Path) -> None:
    """Underscore suffix patterns are tried in order: __, ___, _, empty."""
    # Only the triple-underscore variant exists
    opf = tmp_path / "A320.OPF"
    opf.touch()

    # Fallback tries: A320__.OPF, A320___.OPF, A320_.OPF, A320.OPF
    result = resolve_bada3_file("A320", tmp_path)

    assert result == opf
