"""Judge-Ready CLI Prototype for Agentic Code Intelligence (Theme 01).

Performs exact dense cosine retrieval using the locked P0 embedding model
(google/embeddinggemma-300m) over an included deterministic demo code corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Filter library warnings for clean judge output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("demo")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.encoder import PrePostPipelineEncoder
from src.index.dense_index import DenseIndex


class CodeSearchDemo:
    """End-to-end code retrieval search engine wrapping the locked P0 pipeline."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        corpus_path: str = "data/demo_corpus.json",
        device: str = "cpu",
    ) -> None:
        self.corpus_path = Path(corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.corpus_data: list[dict[str, Any]] = json.load(f)

        self.doc_ids = [doc["id"] for doc in self.corpus_data]
        self.doc_map = {doc["id"]: doc for doc in self.corpus_data}

        # Initialize official encoder
        self.encoder = PrePostPipelineEncoder(model_name=model_name, device=device)
        self.dim = self.encoder.mteb_model_meta.embed_dim or 768

        # Build index over demo corpus
        self.index = DenseIndex(dim=self.dim)
        corpus_texts = [f"{doc.get('title', '')}\n{doc['text']}" for doc in self.corpus_data]
        corpus_vectors = self.encoder.encode_corpus(corpus_texts, batch_size=16)
        self.index.add(self.doc_ids, corpus_vectors)

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Search the corpus with a natural language query."""
        t0 = time.perf_counter()
        query_vector = self.encoder.encode_queries([query], batch_size=1)
        scores, retrieved_ids = self.index.search(query_vector, top_k=min(top_k, len(self.doc_ids)))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for rank, (doc_id, score) in enumerate(zip(retrieved_ids[0], scores[0]), start=1):
            doc = self.doc_map[doc_id]
            results.append({
                "rank": rank,
                "score": float(score),
                "id": doc_id,
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "text": doc.get("text", ""),
            })

        return {
            "query": query,
            "top_k": top_k,
            "latency_ms": latency_ms,
            "results": results,
        }


def format_search_output(response: dict[str, Any]) -> str:
    """Format retrieval response into a clean, human-readable CLI view."""
    lines = [
        "=" * 78,
        "  SAMSUNG PRISM GENAI HACKATHON (THEME 01) - CODE RETRIEVAL PROTOTYPE",
        "=" * 78,
        f"Query:        \"{response['query']}\"",
        f"Latency:      {response['latency_ms']:.2f} ms",
        f"Results:      Top {len(response['results'])} retrieved matches",
        "-" * 78,
    ]

    for item in response["results"]:
        lines.append(f"Rank {item['rank']:02d} | Score: {item['score']:.4f} | ID: {item['id']} ({item['title']})")
        lines.append(f"Category: {item['category']}")
        lines.append("Code Snippet:")
        # Indent code snippet
        indented_code = "\n".join("    " + line for line in item["text"].splitlines())
        lines.append(indented_code)
        lines.append("-" * 78)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Code Intelligence Retrieval Demo CLI")
    parser.add_argument("--query", "-q", type=str, default=None, help="Natural language query string")
    parser.add_argument("--corpus", "-c", type=str, default="data/demo_corpus.json", help="Path to demo corpus JSON")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of ranked results (default: 10)")
    parser.add_argument("--model", "-m", type=str, default="google/embeddinggemma-300m", help="Model name")
    args = parser.parse_args()

    query = args.query
    if not query:
        # Prompt interactively if not provided
        try:
            query = input("Enter code search query: ").strip()
        except EOFError:
            query = "binary search on sorted array"
        if not query:
            query = "binary search on sorted array"

    demo = CodeSearchDemo(model_name=args.model, corpus_path=args.corpus)
    output = demo.search(query=query, top_k=args.top_k)
    print(format_search_output(output))


if __name__ == "__main__":
    main()
