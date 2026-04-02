"""
LLM client module.
"""

from loguru import logger
from typing import Optional


class LLMClient:
    """LLM client for various backends (OpenAI, Ollama)."""

    def __init__(self, config: dict):
        self.config = config
        self.client = None

    def initialize(self):
        """Initialize the LLM client."""
        llm_type = self.config.get("llm_type", "openai")

        if llm_type in ("openai", "openai-reasoning"):
            self._init_openai()
        elif llm_type in ("ollama", "ollama-reasoning"):
            self._init_ollama()
        elif llm_type == "minimax":
            self._init_minimax()
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        from openai import OpenAI
        import os

        api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
        )

    def _init_ollama(self):
        """Initialize Ollama client."""
        from ollama import Client

        self.client = Client(
            host=f"http://{self.config.get('host', 'localhost')}:{self.config.get('port', 11434)}"
        )

    def _init_minimax(self):
        """Initialize Minimax client (OpenAI-compatible API)."""
        from openai import OpenAI
        import os

        api_key = self.config.get("api_key") or os.environ.get("MINIMAX_API_KEY", "")

        # Minimax requires Bearer token in Authorization header
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.get("base_url", "https://api.minimax.chat/v1"),
            default_headers={"Authorization": f"Bearer {api_key}"}
        )

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send chat request."""
        if self.client is None:
            self.initialize()

        model = self.config.get("model", "gpt-4o")
        temperature = self.config.get("temperature", 0.5)
        max_tokens = self.config.get("max_tokens", -1)

        if max_tokens <= 0:
            max_tokens = None

        try:
            if hasattr(self.client, "chat"):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )
                return response["message"]["content"]
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def embed(self, text: str) -> list[float]:
        """Get embeddings for text."""
        raise NotImplementedError("LLMClient.embed() not yet implemented")


def create_llm_client(config: dict) -> LLMClient:
    """Factory function to create LLM client."""
    client = LLMClient(config)
    return client
