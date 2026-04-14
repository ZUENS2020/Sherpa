"""
Preprocessor module - Incidental relations.
"""

from loguru import logger


class IncidentalExtractor:
    """Extract incidental relations between API functions."""

    def __init__(self, call_graph):
        self.call_graph = call_graph

    def extract(self) -> dict:
        """Extract incidental relations."""
        raise NotImplementedError("IncidentalExtractor.extract() not yet implemented")
