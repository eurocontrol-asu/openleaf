"""Tests for aetherx.config module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from openleaf.performance import config
from openleaf.performance.config import Bada3Settings
from openleaf.performance.config import Bada4Settings
from openleaf.performance.config import OpenLeafSettings
from openleaf.performance.exceptions import ConfigurationError


class TestConfigDir:
    """Tests for configuration directory resolution."""

    def test_config_dir_linux(self):
        """Linux/Mac uses XDG_CONFIG_HOME or ~/.config."""
        with patch("openleaf.performance.config.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = config._get_config_dir()
            assert "openleaf" in str(result)

    def test_config_dir_windows(self):
        """Windows uses APPDATA."""
        with (
            patch("openleaf.performance.config.sys") as mock_sys,
            patch.dict("os.environ", {"APPDATA": "/fake/appdata"}),
        ):
            mock_sys.platform = "win32"
            result = config._get_config_dir()
            assert result == Path("/fake/appdata/openleaf")

    def test_config_path_returns_toml(self):
        """Config file path ends with settings.toml."""
        result = config._get_config_path()
        assert result.name == "settings.toml"


class TestGetConfig:
    """Tests for get_config()."""

    def test_returns_dict(self):
        """get_config returns a dict with bada3/bada4 keys."""
        result = config.get_config()
        assert "bada3" in result
        assert "bada4" in result
        assert "path" in result["bada3"]
        assert "synonyms" in result["bada3"]
        assert "path" in result["bada4"]
        assert "synonyms" in result["bada4"]


class TestShowConfig:
    """Tests for show_config()."""

    def test_show_config_runs(self):
        """show_config completes without error."""
        config.show_config()


class TestConfigPath:
    """Tests for config_path()."""

    def test_config_path_returns_path(self):
        """config_path returns a Path object."""
        result = config.config_path()
        assert isinstance(result, Path)


class TestUpdateConfigFile:
    """Tests for _update_config_file()."""

    def test_update_existing_key(self, tmp_path):
        """Updating an existing key rewrites the value."""
        toml = tmp_path / "settings.toml"
        toml.write_text('[bada3]\n# path = "/old/path"\n\n[bada4]\n')

        with (
            patch("openleaf.performance.config._get_config_path", return_value=toml),
            patch("openleaf.performance.config._reload_settings"),
        ):
            config._update_config_file("bada3", "path", "/new/path")

        content = toml.read_text()
        assert 'path = "/new/path"' in content

    def test_add_missing_key(self, tmp_path):
        """Adding a key that doesn't exist appends to section."""
        toml = tmp_path / "settings.toml"
        toml.write_text("[bada3]\n\n[bada4]\n")

        with (
            patch("openleaf.performance.config._get_config_path", return_value=toml),
            patch("openleaf.performance.config._reload_settings"),
        ):
            config._update_config_file("bada3", "path", "/some/path")

        content = toml.read_text()
        assert 'path = "/some/path"' in content


class TestSetBadaPaths:
    """Tests for set_bada3_path, set_bada4_path, etc."""

    def test_set_bada3_path(self, tmp_path):
        """set_bada3_path updates config file."""
        toml = tmp_path / "settings.toml"
        toml.write_text("[bada3]\n\n[bada4]\n")

        with (
            patch("openleaf.performance.config._get_config_path", return_value=toml),
            patch("openleaf.performance.config._reload_settings"),
        ):
            config.set_bada3_path(tmp_path)

    def test_set_bada3_synonyms(self, tmp_path):
        """set_bada3_synonyms updates config file."""
        toml = tmp_path / "settings.toml"
        toml.write_text("[bada3]\n\n[bada4]\n")

        with (
            patch("openleaf.performance.config._get_config_path", return_value=toml),
            patch("openleaf.performance.config._reload_settings"),
        ):
            config.set_bada3_synonyms(tmp_path / "SYNONYM.NEW")

    def test_set_bada4_path(self, tmp_path):
        """set_bada4_path updates config file."""
        toml = tmp_path / "settings.toml"
        toml.write_text("[bada3]\n\n[bada4]\n")

        with (
            patch("openleaf.performance.config._get_config_path", return_value=toml),
            patch("openleaf.performance.config._reload_settings"),
        ):
            config.set_bada4_path(tmp_path)

    def test_set_bada4_synonyms(self, tmp_path):
        """set_bada4_synonyms updates config file."""
        toml = tmp_path / "settings.toml"
        toml.write_text("[bada3]\n\n[bada4]\n")

        with (
            patch("openleaf.performance.config._get_config_path", return_value=toml),
            patch("openleaf.performance.config._reload_settings"),
        ):
            config.set_bada4_synonyms(tmp_path / "SYNONYM.BD4")


class TestGetBadaPaths:
    """Tests for get_bada3_path, get_bada4_path error paths."""

    def test_get_bada3_path_not_configured(self):
        """get_bada3_path raises when path is not configured."""
        test_settings = OpenLeafSettings(bada3=Bada3Settings(), bada4=Bada4Settings())
        with (
            patch("openleaf.performance.config._get_settings", return_value=test_settings),
            pytest.raises(ConfigurationError, match="not configured"),
        ):
            config.get_bada3_path()

    def test_get_bada3_path_not_exists(self, tmp_path):
        """get_bada3_path raises when path doesn't exist."""
        fake_path = str(tmp_path / "nonexistent")
        test_settings = OpenLeafSettings(
            bada3=Bada3Settings(path=fake_path), bada4=Bada4Settings()
        )
        with (
            patch("openleaf.performance.config._get_settings", return_value=test_settings),
            pytest.raises(ConfigurationError, match="does not exist"),
        ):
            config.get_bada3_path()

    def test_get_bada3_path_valid(self, tmp_path):
        """get_bada3_path returns path when valid."""
        test_settings = OpenLeafSettings(
            bada3=Bada3Settings(path=str(tmp_path)), bada4=Bada4Settings()
        )
        with patch("openleaf.performance.config._get_settings", return_value=test_settings):
            result = config.get_bada3_path()
            assert result == tmp_path

    def test_get_bada4_path_not_configured(self):
        """get_bada4_path raises when path is not configured."""
        test_settings = OpenLeafSettings(bada3=Bada3Settings(), bada4=Bada4Settings())
        with (
            patch("openleaf.performance.config._get_settings", return_value=test_settings),
            pytest.raises(ConfigurationError, match="not configured"),
        ):
            config.get_bada4_path()

    def test_get_bada4_path_not_exists(self, tmp_path):
        """get_bada4_path raises when path doesn't exist."""
        fake_path = str(tmp_path / "nonexistent")
        test_settings = OpenLeafSettings(
            bada3=Bada3Settings(), bada4=Bada4Settings(path=fake_path)
        )
        with (
            patch("openleaf.performance.config._get_settings", return_value=test_settings),
            pytest.raises(ConfigurationError, match="does not exist"),
        ):
            config.get_bada4_path()

    def test_get_bada4_path_valid(self, tmp_path):
        """get_bada4_path returns path when valid."""
        test_settings = OpenLeafSettings(
            bada3=Bada3Settings(), bada4=Bada4Settings(path=str(tmp_path))
        )
        with patch("openleaf.performance.config._get_settings", return_value=test_settings):
            result = config.get_bada4_path()
            assert result == tmp_path


class TestGetBadaSynonyms:
    """Tests for get_bada3_synonyms_path, get_bada4_synonyms_path."""

    def test_get_bada3_synonyms_autodetect(self, tmp_path):
        """Auto-detect synonyms in data directory."""
        syn = tmp_path / "SYNONYM.NEW"
        syn.touch()

        test_settings = OpenLeafSettings(
            bada3=Bada3Settings(path=str(tmp_path)), bada4=Bada4Settings()
        )
        with patch("openleaf.performance.config._get_settings", return_value=test_settings):
            result = config.get_bada3_synonyms_path()
            assert result == syn

    def test_get_bada3_synonyms_explicit(self, tmp_path):
        """Explicit synonyms path takes priority."""
        syn = tmp_path / "my_synonyms.txt"
        syn.touch()

        test_settings = OpenLeafSettings(
            bada3=Bada3Settings(synonyms=str(syn)), bada4=Bada4Settings()
        )
        with patch("openleaf.performance.config._get_settings", return_value=test_settings):
            result = config.get_bada3_synonyms_path()
            assert result == syn

    def test_get_bada3_synonyms_none(self):
        """Returns None when no path configured."""
        test_settings = OpenLeafSettings(bada3=Bada3Settings(), bada4=Bada4Settings())
        with patch("openleaf.performance.config._get_settings", return_value=test_settings):
            result = config.get_bada3_synonyms_path()
            assert result is None

    def test_get_bada4_synonyms_autodetect(self, tmp_path):
        """Auto-detect BADA4 synonyms in data directory."""
        syn = tmp_path / "SYNONYM.BD4"
        syn.touch()

        test_settings = OpenLeafSettings(
            bada3=Bada3Settings(), bada4=Bada4Settings(path=str(tmp_path))
        )
        with patch("openleaf.performance.config._get_settings", return_value=test_settings):
            result = config.get_bada4_synonyms_path()
            assert result == syn

    def test_get_bada4_synonyms_none(self):
        """Returns None when no path configured."""
        test_settings = OpenLeafSettings(bada3=Bada3Settings(), bada4=Bada4Settings())
        with patch("openleaf.performance.config._get_settings", return_value=test_settings):
            result = config.get_bada4_synonyms_path()
            assert result is None
