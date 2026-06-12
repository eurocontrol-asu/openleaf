from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path

from openleaf.performance.config import _update_config_file


@pytest.fixture()
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temporary config file and patch helpers."""
    path = tmp_path / "settings.toml"
    monkeypatch.setattr("openleaf.performance.config._get_config_path", lambda: path)
    monkeypatch.setattr("openleaf.performance.config._reload_settings", lambda: None)
    return path


# --- Unit tests ---


def test_update_existing_key(config_file: Path) -> None:
    """Key already present in section is updated in-place."""
    config_file.write_text('[section]\nkey = "old"\n')
    _update_config_file("section", "key", "new")
    lines = config_file.read_text().splitlines()
    assert 'key = "new"' in lines
    assert 'key = "old"' not in lines


def test_uncomment_key(config_file: Path) -> None:
    """Commented-out key is uncommented and updated."""
    config_file.write_text('[section]\n# key = "old"\n')
    _update_config_file("section", "key", "new")
    lines = config_file.read_text().splitlines()
    assert 'key = "new"' in lines
    assert '# key = "old"' not in lines


def test_insert_missing_key(config_file: Path) -> None:
    """Missing key is inserted after the section header."""
    config_file.write_text('[section]\nother = "val"\n')
    _update_config_file("section", "key", "new")
    text = config_file.read_text()
    assert 'key = "new"' in text


# --- Edge cases ---


def test_insert_key_before_next_section(config_file: Path) -> None:
    """Key inserted between target section and the next section header."""
    config_file.write_text('[section]\n[other]\nfoo = "bar"\n')
    _update_config_file("section", "key", "new")
    lines = config_file.read_text().splitlines()
    section_idx = lines.index("[section]")
    other_idx = lines.index("[other]")
    key_idx = lines.index('key = "new"')
    assert section_idx < key_idx < other_idx


def test_insert_key_at_end_of_file(config_file: Path) -> None:
    """Key appended when target section is the last section."""
    config_file.write_text("[section]\n")
    _update_config_file("section", "key", "new")
    lines = config_file.read_text().splitlines()
    assert 'key = "new"' in lines
    assert lines.index('key = "new"') > lines.index("[section]")


def test_multiple_sections_only_target_modified(config_file: Path) -> None:
    """Only the target section is modified when multiple sections exist."""
    config_file.write_text('[a]\nkey = "a_val"\n[b]\nkey = "b_val"\n[c]\nkey = "c_val"\n')
    _update_config_file("b", "key", "updated")
    text = config_file.read_text()
    lines = text.splitlines()
    # Section [a] and [c] keep their original values
    a_idx = lines.index("[a]")
    b_idx = lines.index("[b]")
    c_idx = lines.index("[c]")
    assert lines[a_idx + 1] == 'key = "a_val"'
    assert lines[b_idx + 1] == 'key = "updated"'
    assert lines[c_idx + 1] == 'key = "c_val"'


def test_backslash_value_stays_valid_toml(config_file: Path) -> None:
    """Windows-style paths must be escaped; the file must reload cleanly."""
    import tomllib

    config_file.write_text("[section]\n")
    _update_config_file("section", "path", "C:\\Users\\test\\bada3")
    loaded = tomllib.loads(config_file.read_text())
    assert loaded["section"]["path"] == "C:\\Users\\test\\bada3"


def test_quote_in_value_stays_valid_toml(config_file: Path) -> None:
    import tomllib

    config_file.write_text("[section]\n")
    _update_config_file("section", "path", 'dir with "quotes"')
    loaded = tomllib.loads(config_file.read_text())
    assert loaded["section"]["path"] == 'dir with "quotes"'


def test_write_is_atomic_no_temp_left_behind(config_file: Path) -> None:
    config_file.write_text('[section]\nkey = "old"\n')
    _update_config_file("section", "key", "new")
    assert not config_file.with_name(config_file.name + ".tmp").exists()
    assert 'key = "new"' in config_file.read_text()
