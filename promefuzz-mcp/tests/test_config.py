"""
Tests for configuration module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock
import tomllib

from promefuzz_mcp.config import Config, get_config


class TestConfig:
    """Test cases for Config class."""

    def test_config_init_default(self):
        """Test Config initialization with default path."""
        config = Config()
        assert config._config_path == config._get_default_config_path()
        assert not config._loaded

    def test_config_init_custom_path(self, temp_dir):
        """Test Config initialization with custom path."""
        config_path = temp_dir / "custom_config.toml"
        config = Config(config_path)
        assert config._config_path == config_path

    def test_config_load_template_only(self, temp_dir):
        """Test loading config with only template file."""
        # Create template file
        template_path = temp_dir / "config.template.toml"
        template_content = """
[preprocessor]
run_dummydriver_test = false

[llm]
default_llm = "test_llm"
"""
        template_path.write_text(template_content)

        # Create config with template path
        config = Config(config_path=temp_dir / "config.toml")
        config._template_path = template_path
        config.load()

        assert config._loaded
        assert config.get("preprocessor.run_dummydriver_test") == False

    def test_config_load_user_config(self, temp_dir):
        """Test loading user config file."""
        # Create template file
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("[preprocessor]\nrun_dummydriver_test = false\n")

        # Create user config file
        config_path = temp_dir / "config.toml"
        config_path.write_text("""
[preprocessor]
run_dummydriver_test = true

[llm]
default_llm = "my_llm"
""")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert config.get("preprocessor.run_dummydriver_test") == True
        assert config.get("llm.default_llm") == "my_llm"

    def test_config_get_with_dot_notation(self, temp_dir):
        """Test getting config values with dot notation."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("[llm.test_llm]\nmodel = 'gpt-4'\n")

        config_path = temp_dir / "config.toml"
        config_path.write_text("[llm.test_llm]\nmodel = 'gpt-5'\n")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert config.get("llm.test_llm.model") == "gpt-5"

    def test_config_default_value(self, temp_dir):
        """Test getting default value when key doesn't exist."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert config.get("nonexistent.key", "default") == "default"

    def test_config_fallback_to_template(self, temp_dir):
        """Test fallback to template when key not in user config."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("""
[llm]
default_llm = "template_llm"

[llm.template_llm]
model = "gpt-4"
""")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert config.get("llm.default_llm") == "template_llm"
        assert config.get("llm.template_llm.model") == "gpt-4"

    def test_config_preprocessor_config_property(self, temp_dir):
        """Test preprocessor_config property."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("[preprocessor]\nrun_dummydriver_test = false\n")

        config_path = temp_dir / "config.toml"
        config_path.write_text("[preprocessor]\nrun_dummydriver_test = true\n")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert config.preprocessor_config == {"run_dummydriver_test": True}

    def test_config_bin_config_property(self, temp_dir):
        """Test bin_config property."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert "preprocessor" in config.bin_config
        assert "cgprocessor" in config.bin_config

    def test_config_llm_config_property(self, temp_dir):
        """Test llm_config property."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert "default_llm" in config.llm_config

    def test_config_get_preprocessor_bin_path(self, temp_dir):
        """Test getting preprocessor binary path."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        bin_path = config.get_preprocessor_bin_path()
        assert "processor" in str(bin_path)
        assert "preprocessor" in str(bin_path)

    def test_config_get_cgprocessor_bin_path(self, temp_dir):
        """Test getting cgprocessor binary path."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        bin_path = config.get_cgprocessor_bin_path()
        assert "processor" in str(bin_path)
        assert "cgprocessor" in str(bin_path)

    def test_config_resolve_bin_path_relative(self, temp_dir):
        """Test resolving relative binary path."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        resolved = config._resolve_bin_path("processor/build/bin/preprocessor")
        assert resolved.is_absolute()
        assert "processor" in str(resolved)

    def test_config_resolve_bin_path_absolute(self, temp_dir):
        """Test resolving absolute binary path."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        resolved = config._resolve_bin_path("/usr/bin/test")
        assert str(resolved) == "/usr/bin/test"

    def test_config_get_llm_config(self, temp_dir):
        """Test getting LLM config by name."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("""
[llm.test_llm]
llm_type = "openai"
model = "gpt-4"
""")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        llm_cfg = config.get_llm_config("test_llm")
        assert llm_cfg["llm_type"] == "openai"
        assert llm_cfg["model"] == "gpt-4"

    def test_config_get_default_llm_name(self, temp_dir):
        """Test getting default LLM name."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert config.get_default_llm_name() == "cloud_llm"

    def test_config_get_embedding_llm_name(self, temp_dir):
        """Test getting embedding LLM name."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        config = Config(config_path=config_path)
        config._template_path = template_path
        config.load()

        assert config.get_embedding_llm_name() == "embedding_llm"


class TestConfigSingleton:
    """Test cases for Config singleton pattern."""

    def test_config_singleton(self, temp_dir):
        """Test Config singleton pattern."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        # Reset singleton
        Config._instance = None

        # Get first instance
        config1 = Config.get_instance(config_path)
        config1._template_path = template_path
        config1.load()

        # Get second instance
        config2 = Config.get_instance()

        assert config1 is config2

    def test_config_reset(self, temp_dir):
        """Test Config reset method."""
        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        # Create first instance
        Config._instance = Config(config_path)
        Config._instance._template_path = template_path

        # Reset
        Config.reset()

        # Create new instance
        config = Config(config_path)
        config._template_path = template_path

        assert Config._instance is None or Config._instance is not config


class TestGetConfig:
    """Test cases for get_config function."""

    def test_get_config_creates_instance(self, temp_dir):
        """Test get_config creates an instance."""
        from promefuzz_mcp import config as config_module

        template_path = temp_dir / "config.template.toml"
        template_path.write_text("")

        config_path = temp_dir / "config.toml"
        config_path.write_text("")

        # Reset global config
        config_module._config = None

        config = get_config(config_path)
        config._template_path = template_path
        config.load()

        assert config is not None
