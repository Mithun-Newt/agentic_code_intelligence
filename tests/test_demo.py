"""Unit and integration tests for the judge-ready demo prototype.

Covers:
- Demo corpus format and integrity
- PrePostPipelineEncoder model loading with google/embeddinggemma-300m
- DenseIndex indexing and search
- Deterministic retrieval ranking and result structure
- CLI execution via subprocess
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from demo import CodeSearchDemo


@pytest.fixture(scope="module")
def demo_engine():
    """Shared demo engine fixture for test suite."""
    return CodeSearchDemo(
        model_name="google/embeddinggemma-300m",
        corpus_path="data/demo_corpus.json",
        device="cpu",
    )


def test_demo_corpus_integrity():
    """Verify demo corpus exists, contains valid JSON, and has required fields."""
    corpus_file = Path("data/demo_corpus.json")
    assert corpus_file.exists(), "Demo corpus JSON file must exist"

    with open(corpus_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 10, "Corpus should have at least 10 sample code snippets"

    seen_ids = set()
    for doc in data:
        assert "id" in doc and doc["id"]
        assert "title" in doc and doc["title"]
        assert "text" in doc and doc["text"]
        assert doc["id"] not in seen_ids, f"Duplicate ID: {doc['id']}"
        seen_ids.add(doc["id"])


def test_demo_indexing_and_dimension(demo_engine):
    """Verify model embedding dimension and index capacity."""
    assert demo_engine.dim == 768
    assert len(demo_engine.index) == len(demo_engine.corpus_data)
    assert len(demo_engine.doc_ids) == len(demo_engine.corpus_data)


def test_demo_retrieval_ranking(demo_engine):
    """Verify natural language query accurately retrieves target algorithm."""
    response = demo_engine.search("Find function to perform binary search on sorted array", top_k=3)
    results = response["results"]

    assert len(results) == 3
    # Top retrieved document must be binary search
    assert results[0]["id"] == "doc_binary_search"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["rank"] == 1


def test_demo_result_structure(demo_engine):
    """Verify deterministic structure and fields of search response."""
    query = "Dijkstra shortest path algorithm"
    response = demo_engine.search(query, top_k=5)

    assert response["query"] == query
    assert response["top_k"] == 5
    assert isinstance(response["latency_ms"], float)
    assert response["latency_ms"] > 0

    results = response["results"]
    assert len(results) == 5

    # Check ascending ranks and descending scores
    prev_score = float("inf")
    for idx, item in enumerate(results, start=1):
        assert item["rank"] == idx
        assert "id" in item
        assert "title" in item
        assert "category" in item
        assert "text" in item
        assert item["score"] <= prev_score
        prev_score = item["score"]

    # Top result should be Dijkstra
    assert results[0]["id"] == "doc_dijkstra"


def test_demo_cli_execution():
    """Verify CLI demo runs successfully as an independent subprocess."""
    cmd = [
        sys.executable,
        "demo.py",
        "--query",
        "least recently used cache data structure",
        "--top-k",
        "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    assert proc.returncode == 0, f"CLI exited with error: {proc.stderr}"
    assert "SAMSUNG PRISM GENAI HACKATHON" in proc.stdout
    assert "Rank 01" in proc.stdout
    assert "doc_lru_cache" in proc.stdout
