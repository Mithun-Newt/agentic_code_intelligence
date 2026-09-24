"""Experiment 1: Embedding Model Comparison Runner.

Evaluates candidate embedding models on a held-out train validation slice
of the CoIR/Apps dataset using exact flat cosine similarity retrieval.
Computes NDCG@10, MRR, Recall@10, Recall@100, Recall@1000, corpus indexing time,
and per-query retrieval latency.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index.dense_index import DenseIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_retrieval_metrics(ranks: list[int]) -> dict[str, float]:
    """Compute standard IR metrics for queries with a single ground-truth positive.

    Args:
        ranks: 1-indexed ranks of the relevant document for each query.

    Returns:
        Dictionary with NDCG@10, MRR, Recall@10, Recall@100, Recall@1000.
    """
    ndcg_10_list = [1.0 / np.log2(r + 1) if r <= 10 else 0.0 for r in ranks]
    mrr_list = [1.0 / r for r in ranks]
    r10_list = [1.0 if r <= 10 else 0.0 for r in ranks]
    r100_list = [1.0 if r <= 100 else 0.0 for r in ranks]
    r1000_list = [1.0 if r <= 1000 else 0.0 for r in ranks]

    return {
        "ndcg_at_10": float(np.mean(ndcg_10_list)),
        "mrr": float(np.mean(mrr_list)),
        "recall_at_10": float(np.mean(r10_list)),
        "recall_at_100": float(np.mean(r100_list)),
        "recall_at_1000": float(np.mean(r1000_list)),
    }


def load_model_from_config(model_cfg: dict[str, Any], device: str = "cpu") -> SentenceTransformer:
    """Load SentenceTransformer model based on YAML configuration with fallbacks."""
    load_path = model_cfg["load_path"]
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    logger.info("Loading model '%s' from '%s' (trust_remote_code=%s)...",
                model_cfg["name"], load_path, trust_remote_code)

    try:
        model = SentenceTransformer(
            load_path,
            device=device,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        fallback = model_cfg.get("fallback_load_path")
        if fallback:
            logger.warning("Primary load failed (%s). Attempting fallback: %s", e, fallback)
            model = SentenceTransformer(
                fallback,
                device=device,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise e

    # Apply max_seq_length cap for fair and CPU-bounded evaluation
    max_seq_length = model_cfg.get("max_seq_length", 512)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_seq_length

    return model


def prepare_validation_data(
    num_queries: int = 200,
    num_corpus: int = 1500,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], dict[str, str]]:
    """Load a deterministic held-out validation slice of train queries and corpus.

    Returns:
        query_ids: list of query IDs
        raw_queries: list of raw natural language problem statements
        corpus_ids: list of unique corpus document IDs
        raw_corpus: list of raw Python code snippet texts
        qrels: mapping of query_id -> positive corpus_id
    """
    logger.info("Loading CoIR/Apps dataset splits for internal validation...")
    train_ds = load_dataset("CoIR-Retrieval/apps", "default")["train"]
    corpus_ds = load_dataset("CoIR-Retrieval/apps", "corpus")["corpus"]
    queries_ds = load_dataset("CoIR-Retrieval/apps", "queries")["queries"]

    q_map = {row["_id"]: row["text"] for row in queries_ds}
    c_map = {row["_id"]: row["text"] for row in corpus_ds}

    # Deterministic slice of train pairs
    rng = np.random.RandomState(seed)
    total_train = len(train_ds)
    selected_indices = rng.choice(total_train, size=num_queries, replace=False)
    selected_indices.sort()

    sample_train = train_ds.select(selected_indices)
    query_ids = [row["query-id"] for row in sample_train]
    positive_corpus_ids = [row["corpus-id"] for row in sample_train]
    qrels = {qid: cid for qid, cid in zip(query_ids, positive_corpus_ids)}

    raw_queries = [q_map[qid] for qid in query_ids]

    # Assemble corpus: all positive documents + deterministic distractors
    corpus_id_set = set(positive_corpus_ids)
    all_corpus_ids = list(positive_corpus_ids)

    # Distractor pool from corpus
    distractor_indices = rng.choice(len(corpus_ds), size=num_corpus * 2, replace=False)
    for idx in distractor_indices:
        cid = corpus_ds[int(idx)]["_id"]
        if cid not in corpus_id_set:
            corpus_id_set.add(cid)
            all_corpus_ids.append(cid)
            if len(all_corpus_ids) >= num_corpus:
                break

    raw_corpus = [c_map[cid] for cid in all_corpus_ids]
    logger.info("Validation slice prepared: %d queries against %d corpus documents",
                len(query_ids), len(all_corpus_ids))

    return query_ids, raw_queries, all_corpus_ids, raw_corpus, qrels


def evaluate_model(
    model_cfg: dict[str, Any],
    query_ids: list[str],
    raw_queries: list[str],
    corpus_ids: list[str],
    raw_corpus: list[str],
    qrels: dict[str, str],
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run full evaluation for a single candidate model configuration."""
    torch.set_num_threads(10)
    model = load_model_from_config(model_cfg, device=device)

    # 1. Format inputs with documented prefixes
    q_prefix = model_cfg.get("query_prefix", "")
    p_prefix = model_cfg.get("passage_prefix", "")

    formatted_corpus = [f"{p_prefix}{text}" for text in raw_corpus] if p_prefix else raw_corpus
    formatted_queries = [f"{q_prefix}{text}" for text in raw_queries] if q_prefix else raw_queries

    model_batch_size = model_cfg.get("batch_size", batch_size)

    # 2. Encode corpus & measure indexing time
    logger.info("Indexing %d corpus documents (batch_size=%d)...", len(formatted_corpus), model_batch_size)
    t0_index = time.time()
    with torch.inference_mode():
        corpus_embeddings = model.encode(
            formatted_corpus,
            batch_size=model_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    index_time = time.time() - t0_index
    logger.info("Corpus indexing completed in %.2fs (%.1f docs/sec)",
                index_time, len(formatted_corpus) / index_time)

    # 3. Build exact dense index
    embed_dim = corpus_embeddings.shape[1]
    index = DenseIndex(dim=embed_dim)
    index.add(corpus_ids, corpus_embeddings)

    # 4. Encode queries & retrieve
    logger.info("Encoding %d queries and retrieving top candidates...", len(formatted_queries))
    t0_queries = time.time()
    with torch.inference_mode():
        query_embeddings = model.encode(
            formatted_queries,
            batch_size=model_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    t0_search = time.time()
    scores, retrieved_ids = index.search(query_embeddings, top_k=min(1000, len(corpus_ids)))
    total_query_time = time.time() - t0_queries
    avg_query_latency_ms = (total_query_time / len(formatted_queries)) * 1000.0

    # 5. Compute ranks & metrics
    ranks: list[int] = []
    for qid, candidate_list in zip(query_ids, retrieved_ids):
        target_cid = qrels[qid]
        try:
            rank = candidate_list.index(target_cid) + 1
        except ValueError:
            rank = 999999
        ranks.append(rank)

    metrics = compute_retrieval_metrics(ranks)

    result_dict: dict[str, Any] = {
        "experiment": "Experiment 1",
        "model_name": model_cfg["name"],
        "model_short_name": model_cfg["short_name"],
        "is_control": model_cfg.get("is_control", False),
        "embedding_dim": embed_dim,
        "max_seq_length": model_cfg.get("max_seq_length", 512),
        "num_queries": len(query_ids),
        "num_corpus": len(corpus_ids),
        "index_time_seconds": round(index_time, 2),
        "query_latency_ms": round(avg_query_latency_ms, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }

    logger.info("--- %s RESULTS ---", model_cfg["name"])
    logger.info("NDCG@10:    %.4f", result_dict["ndcg_at_10"])
    logger.info("MRR:        %.4f", result_dict["mrr"])
    logger.info("Recall@10:  %.4f", result_dict["recall_at_10"])
    logger.info("Recall@100: %.4f", result_dict["recall_at_100"])
    logger.info("Recall@1000:%.4f", result_dict["recall_at_1000"])
    logger.info("Index Time: %.2fs", result_dict["index_time_seconds"])
    logger.info("Query Latency: %.2f ms/query", result_dict["query_latency_ms"])

    return result_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 Embedding Model Comparison")
    parser.add_argument("--config", type=str, default="configs/experiment_1_models.yaml")
    parser.add_argument("--model", type=str, default="all", help="Model short_name or 'all'")
    parser.add_argument("--num-queries", type=int, default=200, help="Number of validation queries")
    parser.add_argument("--num-corpus", type=int, default=1500, help="Number of corpus candidates")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models that already have result JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models_to_run = config["models"]
    if args.model != "all":
        models_to_run = [
            m for m in models_to_run
            if m["short_name"] == args.model or m["name"] == args.model
        ]
        if not models_to_run:
            raise ValueError(f"Model '{args.model}' not found in {args.config}")

    # Prepare shared validation slice once
    query_ids, raw_queries, corpus_ids, raw_corpus, qrels = prepare_validation_data(
        num_queries=args.num_queries,
        num_corpus=args.num_corpus,
    )

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_cfg in models_to_run:
        out_path = results_dir / f"experiment_1_{model_cfg['short_name']}.json"

        if args.skip_existing and out_path.exists():
            logger.info("Found existing result for %s at %s. Skipping re-computation.",
                        model_cfg["name"], out_path)
            with open(out_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            all_results.append(res)
            continue

        logger.info("\n==========================================")
        logger.info("RUNNING EXPERIMENT 1 FOR: %s", model_cfg["name"])
        logger.info("==========================================")

        res = evaluate_model(
            model_cfg=model_cfg,
            query_ids=query_ids,
            raw_queries=raw_queries,
            corpus_ids=corpus_ids,
            raw_corpus=raw_corpus,
            qrels=qrels,
            batch_size=args.batch_size,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info("Saved result JSON to: %s", out_path)
        all_results.append(res)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: EMBEDDING MODEL COMPARISON SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model':<35} | {'NDCG@10':<8} | {'MRR':<8} | {'Recall@10':<9} | {'Index Time':<11} | {'Latency':<10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['model_name']:<35} | {r['ndcg_at_10']:<8.4f} | {r['mrr']:<8.4f} | {r['recall_at_10']:<9.4f} | {r['index_time_seconds']:<10.1f}s | {r['query_latency_ms']:<8.1f}ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
