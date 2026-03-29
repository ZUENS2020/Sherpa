"""
Preprocessor module - Relevance calculation.
"""

from loguru import logger


class TypeRelevance:
    """Calculate type-based relevance between API functions."""

    def __init__(self, api_collection, meta):
        self.api_collection = api_collection
        self.meta = meta

    def calculate(self) -> dict:
        """Calculate type relevance."""
        # Placeholder implementation
        return {}


class ClassRelevance:
    """Calculate class-based relevance between API functions."""

    def __init__(self, api_collection, info_repo):
        self.api_collection = api_collection
        self.info_repo = info_repo

    def calculate(self) -> dict:
        """Calculate class relevance."""
        # Placeholder implementation
        return {}


class CallRelevance:
    """Calculate call-based relevance between API functions."""

    def __init__(self, api_collection, call_graph):
        self.api_collection = api_collection
        self.call_graph = call_graph

    def calculate(self) -> dict:
        """Calculate call relevance."""
        # Placeholder implementation
        return {}
