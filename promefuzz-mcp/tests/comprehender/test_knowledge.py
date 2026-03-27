"""
Tests for knowledge base module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from promefuzz_mcp.comprehender.knowledge import KnowledgeBase, RAGRetriever


class TestKnowledgeBase:
    """Test cases for KnowledgeBase class."""

    def test_init(self):
        """Test KnowledgeBase initialization."""
        kb = KnowledgeBase(
            document_paths=["/path/to/docs"],
            embedding_model="test-model",
        )

        assert kb.document_paths == ["/path/to/docs"]
        assert kb.embedding_model == "test-model"
        assert kb.initialized == False

    def test_initialize(self):
        """Test KnowledgeBase initialization."""
        kb = KnowledgeBase(
            document_paths=["/path/to/docs"],
            embedding_model="test-model",
        )

        result = kb.initialize()

        assert result == True
        assert kb.initialized == True

    def test_initialize_already_initialized(self):
        """Test initializing an already initialized knowledge base."""
        kb = KnowledgeBase(
            document_paths=["/path/to/docs"],
            embedding_model="test-model",
        )
        kb.initialized = True

        result = kb.initialize()

        # Should still return True but not re-initialize
        assert result == True

    def test_retrieve(self):
        """Test document retrieval."""
        kb = KnowledgeBase(
            document_paths=["/path/to/docs"],
            embedding_model="test-model",
        )

        results = kb.retrieve("test query", top_k=3)

        # Placeholder implementation returns empty list
        assert results == []

    def test_add_document(self):
        """Test adding a document."""
        kb = KnowledgeBase(
            document_paths=["/path/to/docs"],
            embedding_model="test-model",
        )

        result = kb.add_document("/path/to/new_doc.txt")

        # Placeholder implementation returns True
        assert result == True


class TestRAGRetriever:
    """Test cases for RAGRetriever class."""

    def test_init(self):
        """Test RAGRetriever initialization."""
        kb = KnowledgeBase(
            document_paths=["/path/to/docs"],
            embedding_model="test-model",
        )

        retriever = RAGRetriever(knowledge_base=kb)

        assert retriever.knowledge_base == kb

    def test_retrieve(self):
        """Test retrieval through RAGRetriever."""
        kb = KnowledgeBase(
            document_paths=["/path/to/docs"],
            embedding_model="test-model",
        )
        kb.initialize()

        retriever = RAGRetriever(knowledge_base=kb)

        results = retriever.retrieve("test query", k=5)

        # Should delegate to knowledge base
        assert isinstance(results, list)
