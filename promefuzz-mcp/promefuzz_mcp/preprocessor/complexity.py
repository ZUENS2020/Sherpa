"""
Preprocessor module - Complexity calculation.
"""

from loguru import logger


class ComplexityCalculator:
    """Calculate complexity of API functions."""

    def __init__(self, api_collection=None, info_repo=None, call_graph=None):
        self.api_collection = api_collection
        self.info_repo = info_repo
        self.call_graph = call_graph

    def calculate(self) -> dict:
        """Calculate complexity."""
        raise NotImplementedError("ComplexityCalculator.calculate() not yet implemented")
