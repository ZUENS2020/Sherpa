"""
Configuration management for PromeFuzz MCP.
"""

from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from loguru import logger


class Config:
    """Configuration manager for PromeFuzz MCP."""

    _instance: Optional["Config"] = None

    def __init__(self, config_path: Optional[Path] = None):
        self._config: dict = {}
        self._template: dict = {}
        self._config_path = config_path or self._get_default_config_path()
        self._template_path = self._get_default_template_path()
        self._loaded = False

    @staticmethod
    def _get_default_config_path() -> Path:
        """Get default config path."""
        return Path(__file__).parent.parent / "config.toml"

    @staticmethod
    def _get_default_template_path() -> Path:
        """Get template config path."""
        return Path(__file__).parent.parent / "config.template.toml"

    def load(self) -> "Config":
        """Load configuration from file."""
        if self._loaded:
            return self

        # Load template
        if self._template_path.exists():
            with open(self._template_path, "rb") as f:
                self._template = tomllib.load(f)
        else:
            logger.warning(f"Template config not found: {self._template_path}")

        # Load user config
        if self._config_path.exists():
            with open(self._config_path, "rb") as f:
                self._config = tomllib.load(f)
            logger.info(f"Loaded config from {self._config_path}")
        else:
            logger.warning(f"Config file not found: {self._config_path}")
            logger.info(f"Copy {self._template_path} to {self._config_path} and modify it")

        self._loaded = True
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key (supports dot notation)."""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    # Try template
                    value = self._template
                    for tk in keys:
                        if isinstance(value, dict):
                            value = value.get(tk)
                        else:
                            return default
                    return value if value is not None else default
            else:
                return default

        return value if value is not None else default

    @property
    def preprocessor_config(self) -> dict:
        """Get preprocessor configuration."""
        return self.get("preprocessor", {})

    @property
    def comprehender_config(self) -> dict:
        """Get comprehender configuration."""
        return self.get("comprehender", {})

    @property
    def llm_config(self) -> dict:
        """Get LLM configuration."""
        return self.get("llm", {})

    @property
    def bin_config(self) -> dict:
        """Get binary tools configuration."""
        return self.get("bin", {})

    def get_preprocessor_bin_path(self) -> Path:
        """Get preprocessor binary path."""
        bin_path = self.get("bin.preprocessor", "processor/build/bin/preprocessor")
        return self._resolve_bin_path(bin_path)

    def get_cgprocessor_bin_path(self) -> Path:
        """Get cgprocessor binary path."""
        bin_path = self.get("bin.cgprocessor", "processor/build/bin/cgprocessor")
        return self._resolve_bin_path(bin_path)

    def _resolve_bin_path(self, bin_path: str) -> Path:
        """Resolve binary path (relative to project root or absolute)."""
        path = Path(bin_path)
        if path.is_absolute():
            return path
        # Relative to project root
        return Path(__file__).parent.parent / path

    def get_llm_config(self, name: str) -> dict:
        """Get LLM configuration by name."""
        return self.get(f"llm.{name}", {})

    def get_default_llm_name(self) -> str:
        """Get default LLM name."""
        return self.get("llm.default_llm", "cloud_llm")

    def get_embedding_llm_name(self) -> str:
        """Get embedding LLM name."""
        return self.get("comprehender.embedding_llm", "embedding_llm")

    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> "Config":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config_path).load()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton instance."""
        cls._instance = None


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config(config_path).load()
    return _config
