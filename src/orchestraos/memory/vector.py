"""Semantic memory: ChromaDB (persistent, local) with embeddings from Ollama.

Stores two kinds of documents, told apart by `kind` metadata:
  * "episode": a past goal and the start of its answer
  * "fact":    something the user said about themselves ("user prefers bullet points")
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

Kind = Literal["episode", "fact"]


class Embedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """nomic-embed-text expects task prefixes; using them noticeably improves retrieval."""

    def __init__(self, model: str, base_url: str):
        from langchain_ollama import OllamaEmbeddings

        self._embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        self._prefixes = "nomic" in model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = "search_document: " if self._prefixes else ""
        return await self._embeddings.aembed_documents([prefix + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        prefix = "search_query: " if self._prefixes else ""
        return await self._embeddings.aembed_query(prefix + text)


@dataclass(frozen=True)
class MemoryHit:
    text: str
    kind: str
    distance: float  # cosine distance: 0 = identical, 1 = unrelated
    created: str


class VectorMemory:
    def __init__(self, path: Path, embedder: Embedder, collection: str = "memories"):
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # We always pass our own embeddings, so Chroma's default embedding model is never used.
        self._collection = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        self._embedder = embedder

    async def add(
        self,
        ids: list[str],
        texts: list[str],
        kind: Kind,
        created: str,
        subjects: list[str] | None = None,
    ) -> None:
        if not texts:
            return
        embeddings = await self._embedder.embed_documents(texts)
        subjects = subjects or [""] * len(texts)
        await asyncio.to_thread(
            self._collection.upsert,  # upsert: storing the same fact twice keeps one copy
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"kind": kind, "created": created, "subject": subject} for subject in subjects
            ],
        )

    async def facts_about(self, subject: str, limit: int = 8) -> list[str]:
        """Facts with this subject, newest first, regardless of what's being asked.

        Preferences like "user prefers bullet points" apply to every question, but are
        never semantically close to one ("What is Kubernetes?"), so search alone misses them.
        """
        result = await asyncio.to_thread(
            self._collection.get,
            where={"$and": [{"kind": "fact"}, {"subject": subject}]},
            include=["documents", "metadatas"],
        )
        pairs = sorted(
            zip(result["documents"], result["metadatas"], strict=True),
            key=lambda pair: pair[1]["created"],
            reverse=True,
        )
        return [doc for doc, _ in pairs[:limit]]

    async def search(self, query: str, kind: Kind, k: int = 5) -> list[MemoryHit]:
        count = await asyncio.to_thread(self._collection.count)
        if count == 0:
            return []
        embedding = await self._embedder.embed_query(query)
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=min(k, count),
            where={"kind": kind},
        )
        return [
            MemoryHit(text=doc, kind=meta["kind"], distance=dist, created=meta["created"])
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
