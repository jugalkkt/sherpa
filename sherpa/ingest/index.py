"""Step 3: store chunk embeddings in Chroma and search them.

Embedding = turning text into a vector (768 numbers for nomic-embed-text) such that
texts with similar meaning get vectors pointing in similar directions. Search embeds
the query the same way and returns the chunks whose vectors are closest by cosine
distance (similarity = 1 - distance; 1.0 means same direction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sherpa.config import Settings, get_settings
from sherpa.ingest.chunking import Chunk
from sherpa.llm import get_embeddings, with_retry

COLLECTION = "notes"
EMBED_BATCH = 32

# nomic-embed-text was trained with task prefixes; without them retrieval quality drops.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingModelMismatch(RuntimeError):
    """The index was built with a different embedding model than the one configured.
    Vectors from different models are not comparable, so querying would return junk."""


@dataclass
class Hit:
    id: str
    text: str
    source: str
    title: str
    similarity: float
    page: int | None = None
    section: str | None = None

    @property
    def location(self) -> str:
        """Human-readable pointer used in citations: `notes/x.pdf p.3` or `notes/x.md › A > B`."""
        if self.page is not None:
            return f"{self.source} p.{self.page}"
        if self.section:
            return f"{self.source} › {self.section}"
        return self.source


class NotesIndex:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None
        self._collection = None

    # -- plumbing ---------------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            import chromadb  # slow import; only paid by commands that need the index
            from chromadb.config import Settings as ChromaSettings

            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                COLLECTION,
                embedding_function=None,  # we embed ourselves via Ollama
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"embed_model": self.settings.embed_model},
            )
            self.check_model()
        return self._collection

    def check_model(self) -> None:
        built_with = (self._collection.metadata or {}).get("embed_model")
        if built_with and built_with != self.settings.embed_model:
            raise EmbeddingModelMismatch(
                f"index was built with {built_with!r} but EMBED_MODEL is {self.settings.embed_model!r}; "
                "run `python -m sherpa ingest --reset` to rebuild it"
            )

    def reset(self) -> None:
        if COLLECTION in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(COLLECTION)
        self._collection = None

    def count(self) -> int:
        return self.collection.count()

    def has_ids(self, ids: list[str]) -> bool:
        if not ids:
            return False
        return len(self.collection.get(ids=ids, include=[])["ids"]) == len(ids)

    # -- write ------------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], on_progress: Callable[[int], None] | None = None) -> None:
        """Embed chunks in batches and upsert them. `on_progress(n)` is called after each batch."""
        embed = with_retry(get_embeddings().embed_documents)
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors = embed([DOC_PREFIX + c.text for c in batch])
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=vectors,
                documents=[c.text for c in batch],
                # Chroma metadata values must be str/int/float/bool (no None)
                metadatas=[{k: v for k, v in c.metadata.items() if v is not None} for c in batch],
            )
            if on_progress:
                on_progress(len(batch))

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    # -- read -------------------------------------------------------------------------

    def query(self, text: str, k: int | None = None) -> list[Hit]:
        k = k or self.settings.retrieval_top_k
        total = self.count()
        if total == 0:
            return []
        vector = with_retry(get_embeddings().embed_query)(QUERY_PREFIX + text)
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(
                Hit(
                    id=id_,
                    text=doc,
                    source=meta.get("source", "?"),
                    title=meta.get("title", "?"),
                    similarity=round(1.0 - dist, 4),
                    page=meta.get("page"),
                    section=meta.get("section"),
                )
            )
        return hits
