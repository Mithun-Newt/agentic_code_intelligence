"""Smoke tests for Experiment 0 baseline pipeline components.

Covers:
- PrePostPipelineEncoder conformity with AbsEncoder
- Unit normalization of embeddings
- DenseIndex exact search
- End-to-end mini-retrieval
"""

import numpy as np
import pytest
from mteb.models.abs_encoder import AbsEncoder
from src.encoder import PrePostPipelineEncoder
from src.index.dense_index import DenseIndex, normalize_vectors


@pytest.fixture(scope="module")
def encoder():
    """Shared encoder fixture across tests."""
    return PrePostPipelineEncoder(model_name="intfloat/e5-base-v2", device="cpu")


def test_encoder_is_abs_encoder(encoder):
    """Verify that PrePostPipelineEncoder conforms to MTEB's AbsEncoder."""
    assert issubclass(PrePostPipelineEncoder, AbsEncoder)
    assert isinstance(encoder, AbsEncoder)
    assert encoder.mteb_model_meta is not None
    assert encoder.mteb_model_meta.embed_dim == 768


def test_encoder_output_shape_and_norm(encoder):
    """Verify encoder produces unit-normalized 768-dim vectors."""
    queries = ["Find recursive depth first search on graphs"]
    corpus = [
        "def dfs(graph, node, visited):\n    visited.add(node)\n    for neighbor in graph[node]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)"
    ]

    q_emb = encoder.encode_queries(queries)
    c_emb = encoder.encode_corpus(corpus)

    assert q_emb.shape == (1, 768)
    assert c_emb.shape == (1, 768)

    # Check L2 normalization (cosine similarity property)
    q_norm = float(np.linalg.norm(q_emb, axis=-1)[0])
    c_norm = float(np.linalg.norm(c_emb, axis=-1)[0])
    assert pytest.approx(q_norm, abs=1e-5) == 1.0
    assert pytest.approx(c_norm, abs=1e-5) == 1.0


def test_dense_index_orthogonal_exact_search():
    """Verify that DenseIndex retrieves exact orthogonal vector matches."""
    dim = 4
    index = DenseIndex(dim=dim)

    # 4 orthogonal standard basis vectors
    vectors = np.eye(dim, dtype=np.float32)
    ids = [f"doc_{i}" for i in range(dim)]

    index.add(ids, vectors)
    assert len(index) == dim

    # Query with the 3rd basis vector
    query = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    scores, retrieved_ids = index.search(query, top_k=2)

    assert retrieved_ids[0][0] == "doc_2"
    assert pytest.approx(scores[0][0], abs=1e-5) == 1.0


def test_end_to_end_mini_retrieval(encoder):
    """Verify end-to-end dense retrieval on a small set of Python code snippets."""
    code_snippets = {
        "doc_sort": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",
        "doc_search": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
        "doc_hello": "def print_greeting():\n    print('Hello, Hackathon!')",
    }

    corpus_ids = list(code_snippets.keys())
    corpus_texts = list(code_snippets.values())

    corpus_embeddings = encoder.encode_corpus(corpus_texts)

    index = DenseIndex(dim=768)
    index.add(corpus_ids, corpus_embeddings)

    query = "Find function to perform binary search on sorted array"
    query_embedding = encoder.encode_queries([query])

    scores, retrieved_ids = index.search(query_embedding, top_k=3)

    # Top retrieved document should be binary_search
    assert retrieved_ids[0][0] == "doc_search"
    assert scores[0][0] > scores[0][1]
