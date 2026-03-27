"""
Type definitions for knowledge and comprehension.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGExcerpt:
    """RAG document excerpt."""

    content: str
    """Excerpt content."""

    source: str
    """Source file or URL."""

    score: float = 0.0
    """Relevance score."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RAGExcerpt":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LibraryComprehension:
    """Library comprehension result."""

    purpose: str = ""
    """Library purpose description."""

    functions: dict[str, str] = field(default_factory=dict)
    """Function name to usage mapping."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "purpose": self.purpose,
            "functions": self.functions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LibraryComprehension":
        """Create from dictionary."""
        return cls(
            purpose=data.get("purpose", ""),
            functions=data.get("functions", {}),
        )


@dataclass
class SemanticRelevanceResult:
    """Semantic relevance result."""

    func_loc_a: str
    """Function A location."""

    func_loc_b: str
    """Function B location."""

    relevance: float
    """Relevance score."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "func_loc_a": self.func_loc_a,
            "func_loc_b": self.func_loc_b,
            "relevance": self.relevance,
        }


@dataclass
class KnowledgeBaseInfo:
    """Knowledge base information."""

    id: str
    """Knowledge base ID."""

    document_count: int = 0
    """Number of documents."""

    chunk_count: int = 0
    """Number of chunks."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
        }
