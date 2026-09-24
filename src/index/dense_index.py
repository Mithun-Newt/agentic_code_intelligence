"""Exact dense vector index supporting cosine similarity via FAISS IndexFlatIP.

Normalizes all vectors to unit L2 norm so inner product search corresponds
strictly to cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("FAISS not installed; falling back to NumPy matrix multiplication.")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D numpy array to unit length."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class DenseIndex:
    """Exact nearest-neighbor index using inner product on normalized embeddings."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.doc_ids: list[str] = []
        self._vectors: list[np.ndarray] = []  # Used for numpy fallback or state tracking

        if _FAISS_AVAILABLE:
            self._faiss_index = faiss.IndexFlatIP(dim)
        else:
            self._faiss_index = None

    def __len__(self) -> int:
        return len(self.doc_ids)

    def add(self, corpus_ids: Sequence[str], embeddings: np.ndarray) -> None:
        """Add corpus embeddings and their corresponding IDs to the index.

        Args:
            corpus_ids: List of unique document or snippet identifiers.
            embeddings: 2D NumPy array of shape (N, dim).
        """
        if len(corpus_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(corpus_ids)} ids but {len(embeddings)} embeddings."
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings of shape (N, {self.dim}), got {embeddings.shape}."
            )

        norm_embeddings = normalize_vectors(embeddings)
        self.doc_ids.extend(corpus_ids)

        if self._faiss_index is not None:
            self._faiss_index.add(norm_embeddings)
        else:
            self._vectors.append(norm_embeddings)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Search the top-k most similar documents for given queries.

        Args:
            query_embeddings: 2D NumPy array of shape (Q, dim) or 1D array of shape (dim,).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            scores: NumPy array of shape (Q, top_k) containing cosine similarity scores.
            retrieved_ids: Nested list of retrieved document IDs of shape (Q, top_k).
        """
        if len(self.doc_ids) == 0:
            raise RuntimeError("Index is empty. Add documents before searching.")

        if query_embeddings.ndim == 1:
            query_embeddings = np.expand_dims(query_embeddings, axis=0)

        if query_embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension {query_embeddings.shape[1]} does not match index dim {self.dim}."
            )

        norm_queries = normalize_vectors(query_embeddings)
        k = min(top_k, len(self.doc_ids))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(norm_queries, k)
        else:
            # Exact cosine similarity via matrix multiplication fallback
            corpus_matrix = np.vstack(self._vectors)
            similarity_matrix = np.dot(norm_queries, corpus_matrix.T)
            # Get top-k indices per query
            indices = np.argsort(-similarity_matrix, axis=1)[:, :k]
            scores = np.take_along_axis(similarity_matrix, indices, axis=1)

        retrieved_ids: list[list[str]] = [
            [self.doc_ids[idx] for idx in row_indices if idx >= 0]
            for row_indices in indices
        ]
        return scores, retrieved_ids

    def reset(self) -> None:
        """Clear all stored vectors and document IDs."""
        self.doc_ids.clear()
        self._vectors.clear()
        if self._faiss_index is not None:
            self._faiss_index.reset()
