"""
Comprehender module - Function usage comprehension.
"""

from loguru import logger


class FunctionUsageComprehender:
    """Comprehend function usage using LLM."""

    def __init__(self, llm_client, knowledge_base):
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base

    def comprehend(self, function_name: str) -> str:
        """Comprehend function usage."""
        raise NotImplementedError("func_usage.comprehend() not yet implemented")

    def comprehend_all(self, function_names: list[str]) -> dict:
        """Comprehend usage of all functions."""
        raise NotImplementedError("func_usage.comprehend_all() not yet implemented")
