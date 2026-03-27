"""
Tests for LLM client module.
"""

import pytest
from unittest.mock import MagicMock, patch

from promefuzz_mcp.llm.client import LLMClient, create_llm_client


class TestLLMClient:
    """Test cases for LLMClient class."""

    def test_init_with_openai_config(self):
        """Test LLMClient initialization with OpenAI config."""
        config = {
            "llm_type": "openai",
            "base_url": "https://api.test.com/v1",
            "api_key": "test_key",
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000,
        }

        client = LLMClient(config)
        assert client.config == config
        assert client.client is None

    def test_init_with_ollama_config(self):
        """Test LLMClient initialization with Ollama config."""
        config = {
            "llm_type": "ollama",
            "host": "localhost",
            "port": 11434,
            "model": "llama2",
            "temperature": 0.7,
        }

        client = LLMClient(config)
        assert client.config == config
        assert client.client is None

    def test_init_with_reasoning_model(self):
        """Test LLMClient initialization with reasoning model."""
        config = {
            "llm_type": "openai-reasoning",
            "model": "o1",
            "temperature": 0.5,
        }

        client = LLMClient(config)
        assert client.config == config

    @patch("promefuzz_mcp.llm.client.OpenAI")
    def test_init_openai_client(self, mock_openai):
        """Test initializing OpenAI client."""
        config = {
            "llm_type": "openai",
            "base_url": "https://api.test.com/v1",
            "api_key": "test_key",
            "model": "gpt-4",
        }

        client = LLMClient(config)
        client.initialize()

        mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.test.com/v1",
        )
        assert client.client is not None

    @patch("promefuzz_mcp.llm.client.Client")
    def test_init_ollama_client(self, mock_ollama):
        """Test initializing Ollama client."""
        config = {
            "llm_type": "ollama",
            "host": "localhost",
            "port": 11434,
            "model": "llama2",
        }

        client = LLMClient(config)
        client.initialize()

        mock_ollama.assert_called_once_with(
            host="http://localhost:11434",
        )
        assert client.client is not None

    def test_init_invalid_llm_type(self):
        """Test initialization with invalid LLM type."""
        config = {
            "llm_type": "invalid_type",
        }

        client = LLMClient(config)

        with pytest.raises(ValueError, match="Unknown LLM type"):
            client.initialize()

    @patch("promefuzz_mcp.llm.client.OpenAI")
    def test_chat_with_openai(self, mock_openai):
        """Test chat method with OpenAI."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Test response"))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        config = {
            "llm_type": "openai",
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 100,
        }

        client = LLMClient(config)
        client.initialize()

        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("promefuzz_mcp.llm.client.Client")
    def test_chat_with_ollama(self, mock_ollama):
        """Test chat method with Ollama."""
        # Setup mock
        mock_response = {
            "message": {
                "content": "Ollama response"
            }
        }
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response
        mock_ollama.return_value = mock_client

        config = {
            "llm_type": "ollama",
            "model": "llama2",
            "temperature": 0.5,
            "max_tokens": 100,
        }

        client = LLMClient(config)
        client.initialize()

        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response == "Ollama response"

    @patch("promefuzz_mcp.llm.client.OpenAI")
    def test_chat_not_initialized(self, mock_openai):
        """Test chat without initializing client."""
        config = {
            "llm_type": "openai",
            "model": "gpt-4",
        }

        client = LLMClient(config)
        # Don't call initialize

        messages = [{"role": "user", "content": "Hello"}]

        # Should auto-initialize
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Response"))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        response = client.chat(messages)

        assert response == "Response"

    def test_embed_placeholder(self):
        """Test embed method placeholder."""
        config = {
            "llm_type": "openai",
            "model": "gpt-4",
        }

        client = LLMClient(config)

        embedding = client.embed("test text")

        # Placeholder returns empty list
        assert embedding == []


class TestCreateLLMClient:
    """Test cases for create_llm_client factory function."""

    def test_create_llm_client(self):
        """Test creating LLM client via factory."""
        config = {
            "llm_type": "openai",
            "model": "gpt-4",
        }

        client = create_llm_client(config)

        assert isinstance(client, LLMClient)
        assert client.config == config
