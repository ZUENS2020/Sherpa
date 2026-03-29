"""
Comprehender module - Knowledge base management.
"""

from __future__ import annotations

import json
import math
import re
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List, Tuple
try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal test env
    import logging
    logger = logging.getLogger("promefuzz.knowledge")

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_:+./-]*")
_DEFAULT_SUFFIXES = [".md", ".txt", ".rst", ".html", ".json", ".xml", ".yaml", ".yml", ".c", ".h", ".cc", ".cpp"]
_OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
_VECTOR_DIM_LIMIT = 8192


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    token_count: int


class KnowledgeBase:
    """RAG knowledge base for document retrieval."""

    def __init__(
        self,
        document_paths: List[str],
        output_path: str = "knowledge_db",
        embedding_model: str = "nomic-embed-text",
    ):
        self.document_paths = [Path(p) for p in document_paths]
        self.output_path = Path(output_path)
        self.embedding_model = embedding_model
        self.initialized = False
        self.collection = None
        self.documents: List[dict] = []
        self.chunks: List[_Chunk] = []
        self.inverted_index: dict[str, dict[str, int]] = {}
        self.idf: dict[str, float] = {}
        self._metadata_file = self.output_path / "metadata.json"
        self._index_file = self.output_path / "index.json"
        self._chunks_file = self.output_path / "chunks.json"
        self._vectors_file = self.output_path / "vectors.json"
        self.chunk_vectors: dict[str, list[float]] = {}
        self.embedding_ok: bool = False
        self.rag_degraded: bool = False
        self.rag_degraded_reason: str = ""
        self.embedding_provider: str = "openrouter"
        self.embedding_model_used: str = ""
        self.cache_loaded: bool = False

    @staticmethod
    def _normalize_vector(raw: Any) -> list[float]:
        if not isinstance(raw, list):
            return []
        out: list[float] = []
        for x in raw[:_VECTOR_DIM_LIMIT]:
            try:
                out.append(float(x))
            except Exception:
                return []
        return out

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]

    @staticmethod
    def _chunk_text(text: str, *, max_chars: int = 1400, overlap_chars: int = 220) -> list[str]:
        body = str(text or "")
        if not body:
            return []
        if len(body) <= max_chars:
            return [body]
        chunks: list[str] = []
        i = 0
        step = max(1, max_chars - overlap_chars)
        while i < len(body):
            part = body[i:i + max_chars]
            if not part:
                break
            chunks.append(part)
            if i + max_chars >= len(body):
                break
            i += step
        return chunks

    def _load_if_exists(self) -> bool:
        if not (self._metadata_file.is_file() and self._index_file.is_file() and self._chunks_file.is_file()):
            return False
        try:
            metadata = json.loads(self._metadata_file.read_text(encoding="utf-8", errors="replace"))
            index_doc = json.loads(self._index_file.read_text(encoding="utf-8", errors="replace"))
            chunks_doc = json.loads(self._chunks_file.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            logger.warning(f"failed to load knowledge cache: {e}")
            return False
        if not isinstance(metadata, dict) or not isinstance(index_doc, dict) or not isinstance(chunks_doc, list):
            return False
        vectors_doc: dict[str, Any] = {}
        if self._vectors_file.is_file():
            try:
                parsed_vectors = json.loads(self._vectors_file.read_text(encoding="utf-8", errors="replace"))
                if isinstance(parsed_vectors, dict):
                    vectors_doc = parsed_vectors
            except Exception:
                vectors_doc = {}
        docs = metadata.get("documents")
        if not isinstance(docs, list):
            docs = []
        self.documents = docs
        self.inverted_index = index_doc.get("inverted_index") if isinstance(index_doc.get("inverted_index"), dict) else {}
        self.idf = index_doc.get("idf") if isinstance(index_doc.get("idf"), dict) else {}
        self.chunks = []
        for item in chunks_doc:
            if not isinstance(item, dict):
                continue
            self.chunks.append(
                _Chunk(
                    chunk_id=str(item.get("chunk_id") or ""),
                    doc_id=str(item.get("doc_id") or ""),
                    source_path=str(item.get("source_path") or ""),
                    text=str(item.get("text") or ""),
                    token_count=int(item.get("token_count") or 0),
                )
            )
        self.chunk_vectors = {}
        raw_vectors = vectors_doc.get("chunk_vectors")
        if isinstance(raw_vectors, dict):
            for chunk_id, raw_vec in raw_vectors.items():
                vec = self._normalize_vector(raw_vec)
                if vec:
                    self.chunk_vectors[str(chunk_id)] = vec
        self.embedding_model_used = str(metadata.get("embedding_model_used") or metadata.get("embedding_model") or "")
        self.embedding_ok = bool(metadata.get("embedding_ok")) and bool(self.chunk_vectors)
        self.rag_degraded = bool(metadata.get("rag_degraded") or False)
        self.rag_degraded_reason = str(metadata.get("rag_degraded_reason") or "")
        self.initialized = True
        self.cache_loaded = True
        return True

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def _embedding_credentials(self) -> tuple[str, str]:
        key = str(os.environ.get("OPENROUTER_EMBEDDING_API_KEY") or "").strip()
        model = (
            str(os.environ.get("OPENROUTER_EMBEDDING_MODEL") or "").strip()
            or str(self.embedding_model or "").strip()
            or "text-embedding-3-small"
        )
        return key, model

    def _embed_texts_openrouter(self, texts: list[str], *, model: str) -> list[list[float]]:
        key, _ = self._embedding_credentials()
        if not key:
            raise RuntimeError("openrouter_embedding_key_missing")
        payload = {"model": model, "input": texts}
        req = urllib.request.Request(
            _OPENROUTER_EMBEDDING_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(e)
            raise RuntimeError(f"openrouter_embedding_http_error:{e.code}:{body[:240]}") from e
        except Exception as e:
            raise RuntimeError(f"openrouter_embedding_request_failed:{e}") from e
        try:
            parsed = json.loads(body)
        except Exception as e:
            raise RuntimeError(f"openrouter_embedding_json_invalid:{e}") from e
        data_rows = parsed.get("data")
        if not isinstance(data_rows, list):
            raise RuntimeError("openrouter_embedding_missing_data")
        # Prefer index-aware mapping to stay compatible with providers that may
        # return duplicated/expanded rows for a single input.
        indexed_vectors: dict[int, list[float]] = {}
        sequential_vectors: list[list[float]] = []
        for row in data_rows:
            if not isinstance(row, dict):
                continue
            vec = self._normalize_vector(row.get("embedding"))
            if not vec:
                continue
            idx_raw = row.get("index")
            try:
                idx = int(idx_raw)
            except Exception:
                idx = -1
            if 0 <= idx < len(texts):
                indexed_vectors.setdefault(idx, vec)
            else:
                sequential_vectors.append(vec)
        vectors: list[list[float]] = []
        if indexed_vectors:
            for i in range(len(texts)):
                vec = indexed_vectors.get(i)
                if vec:
                    vectors.append(vec)
            if len(vectors) < len(texts):
                for vec in sequential_vectors:
                    if len(vectors) >= len(texts):
                        break
                    vectors.append(vec)
        else:
            vectors = list(sequential_vectors)
        if len(vectors) < len(texts):
            raise RuntimeError(f"openrouter_embedding_size_mismatch:{len(vectors)}!={len(texts)}")
        if len(vectors) > len(texts):
            logger.warning(
                "openrouter returned extra embeddings; truncating extras "
                f"({len(vectors)} -> {len(texts)})"
            )
            vectors = vectors[: len(texts)]
        return vectors

    def _build_chunk_embeddings(self) -> None:
        key, model = self._embedding_credentials()
        self.embedding_model_used = model
        if not key:
            self.embedding_ok = False
            self.rag_degraded = True
            self.rag_degraded_reason = "embedding_key_or_model_missing"
            return
        if not self.chunks:
            self.embedding_ok = False
            self.rag_degraded = True
            self.rag_degraded_reason = "no_chunks_for_embedding"
            return
        batch_size = 16
        vectors: dict[str, list[float]] = {}
        try:
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i + batch_size]
                texts = [c.text for c in batch]
                embs = self._embed_texts_openrouter(texts, model=model)
                for c, emb in zip(batch, embs):
                    vectors[c.chunk_id] = emb
            self.chunk_vectors = vectors
            self.embedding_ok = bool(self.chunk_vectors)
            self.rag_degraded = not self.embedding_ok
            self.rag_degraded_reason = "" if self.embedding_ok else "embedding_empty_result"
        except Exception as e:
            self.embedding_ok = False
            self.chunk_vectors = {}
            self.rag_degraded = True
            self.rag_degraded_reason = str(e)
            logger.warning(f"embedding degraded; fallback to lexical retrieval: {e}")

    def initialize(self) -> Tuple[bool, Path]:
        """
        Initialize the knowledge base.

        Returns:
            Tuple of (success, output_path)
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self._load_if_exists():
            logger.info(f"Loaded knowledge base from cache at {self.output_path}")
            return True, self.output_path

        self.documents = self._collect_documents()
        self.chunks = []
        for doc in self.documents:
            source_path = str(doc.get("path") or "")
            doc_id = str(doc.get("id") or "")
            text = str(doc.get("text") or "")
            for idx, part in enumerate(self._chunk_text(text)):
                tokens = self._tokenize(part)
                if not tokens:
                    continue
                self.chunks.append(
                    _Chunk(
                        chunk_id=f"{doc_id}::{idx}",
                        doc_id=doc_id,
                        source_path=source_path,
                        text=part,
                        token_count=len(tokens),
                    )
                )

        chunk_count = max(1, len(self.chunks))
        doc_freq: dict[str, int] = {}
        self.inverted_index = {}
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.text)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, count in tf.items():
                postings = self.inverted_index.setdefault(tok, {})
                postings[chunk.chunk_id] = count
            for tok in tf.keys():
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        self.idf = {tok: math.log((chunk_count + 1) / (df + 1)) + 1.0 for tok, df in doc_freq.items()}
        self._build_chunk_embeddings()
        self.initialized = True
        self.cache_loaded = False
        logger.info(
            f"Initialized knowledge base with {len(self.documents)} docs and {len(self.chunks)} chunks at {self.output_path}"
        )

        self._metadata_file.write_text(
            json.dumps(
                {
                    "document_count": len(self.documents),
                    "chunk_count": len(self.chunks),
                    "embedding_model": self.embedding_model,
                    "embedding_provider": self.embedding_provider,
                    "embedding_model_used": self.embedding_model_used,
                    "embedding_ok": self.embedding_ok,
                    "rag_degraded": self.rag_degraded,
                    "rag_degraded_reason": self.rag_degraded_reason,
                    "documents": [
                        {k: v for k, v in doc.items() if k != "text"}
                        for doc in self.documents
                        if isinstance(doc, dict)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._index_file.write_text(
            json.dumps(
                {
                    "idf": self.idf,
                    "inverted_index": self.inverted_index,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._chunks_file.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "source_path": c.source_path,
                        "token_count": c.token_count,
                        "text": c.text,
                    }
                    for c in self.chunks
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._vectors_file.write_text(
            json.dumps(
                {
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": self.embedding_model_used,
                    "chunk_vectors": self.chunk_vectors,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return True, self.output_path

    def _collect_documents(self) -> List[dict]:
        """Collect documents from specified paths."""
        documents = []
        suffixes = _DEFAULT_SUFFIXES

        doc_idx = 0
        for doc_path in self.document_paths:
            if doc_path.is_file():
                text = doc_path.read_text(encoding="utf-8", errors="replace")
                documents.append({
                    "id": f"doc-{doc_idx}",
                    "path": str(doc_path),
                    "name": doc_path.name,
                    "type": doc_path.suffix,
                    "text": text,
                })
                doc_idx += 1
            elif doc_path.is_dir():
                for suffix in suffixes:
                    for file in doc_path.rglob(f"*{suffix}"):
                        try:
                            text = file.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            continue
                        documents.append({
                            "id": f"doc-{doc_idx}",
                            "path": str(file),
                            "name": file.name,
                            "type": file.suffix,
                            "text": text,
                        })
                        doc_idx += 1

        return documents

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant document excerpts
        """
        q = str(query or "").strip()
        if not q:
            return []
        if not self.initialized:
            self.initialize()
        chunk_map = {c.chunk_id: c for c in self.chunks}
        scores: dict[str, float] = {}

        if self.embedding_ok and self.chunk_vectors:
            try:
                query_vec = self._embed_texts_openrouter([q], model=self.embedding_model_used or self.embedding_model)[0]
            except Exception as e:
                self.rag_degraded = True
                self.rag_degraded_reason = str(e)
                query_vec = []
            if query_vec:
                for chunk_id, vec in self.chunk_vectors.items():
                    sim = self._cosine_similarity(query_vec, vec)
                    if sim > 0.0:
                        scores[chunk_id] = sim

        if not scores:
            query_terms = self._tokenize(q)
            if not query_terms:
                return []
            for tok in query_terms:
                postings = self.inverted_index.get(tok) or {}
                idf = float(self.idf.get(tok) or 0.0)
                for chunk_id, tf in postings.items():
                    scores[chunk_id] = float(scores.get(chunk_id) or 0.0) + float(tf) * idf
            if not scores:
                return []

        out: list[dict[str, Any]] = []
        for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            text = chunk.text.strip()
            snippet = text[:280]
            out.append(
                {
                    "chunk_id": chunk_id,
                    "source_path": chunk.source_path,
                    "score": round(float(score), 6),
                    "snippet": snippet,
                    "token_count": chunk.token_count,
                    "embedding_enabled": bool(self.embedding_ok),
                    "rag_degraded": bool(self.rag_degraded),
                }
            )
        return out

    def add_document(self, path: str) -> bool:
        """Add a document to the knowledge base."""
        self.documents.append({
            "path": path,
            "name": Path(path).name,
            "type": Path(path).suffix
        })
        return True

    def save(self) -> Path:
        """Save knowledge base to disk."""
        return self.output_path


class RAGRetriever:
    """RAG retriever for document search."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def retrieve(self, query: str, k: int = 3) -> List[dict]:
        """Retrieve documents by query."""
        return self.knowledge_base.retrieve(query, k)
