"""
PromeFuzz MCP Tools.

A collection of MCP tools for code analysis and comprehension.
"""

from .config import get_config
from .build import build_binaries, check_binaries
from .preprocessor import ASTPreprocessor, APIExtractor
from .comprehender import KnowledgeBase

__version__ = "0.1.0"

__all__ = [
    "get_config",
    "build_binaries",
    "check_binaries",
    "ASTPreprocessor",
    "APIExtractor",
    "KnowledgeBase",
]
