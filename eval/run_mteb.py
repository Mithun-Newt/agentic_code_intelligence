"""Official MTEB AppsRetrieval Evaluation Script for Experiment 0 (Baseline).

Runs PrePostPipelineEncoder on the CoIR AppsRetrieval benchmark,
generates the official evaluation JSON, and logs results to RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mteb
from src.encoder import PrePostPipelineEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(
    model_name: str = "intfloat/e5-base-v2",
    batch_size: int = 64,
    output_path: str = "eval/outputs/appsretrieval_results.json",
    results_md_path: str = "eval/RESULTS.md",
) -> dict:
    """Run baseline evaluation on AppsRetrieval and save results."""
    logger.info("Initializing PrePostPipelineEncoder with %s...", model_name)
    model = PrePostPipelineEncoder(model_name=model_name, device="cpu")

    logger.info("Loading AppsRetrieval task from MTEB...")
    task = mteb.get_task("AppsRetrieval")

    logger.info("Running MTEB evaluation (batch_size=%d)...", batch_size)
    t0 = time.time()
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": batch_size},
    )
    elapsed_time = time.time() - t0
    logger.info("Evaluation finished in %.2f seconds (%.2f minutes)", elapsed_time, elapsed_time / 60)

    # Extract task result
    task_result = list(result.task_results)[0]
    result_dict = task_result.to_dict()

    # Ensure output directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    logger.info("Saved evaluation JSON to: %s", out_file)

    # Also save to appsretrieval_results.json in root for hackathon submission convenience
    root_json = Path("appsretrieval_results.json")
    with open(root_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    logger.info("Saved copy of evaluation JSON to: %s", root_json)

    # Extract NDCG@10 and MRR metrics
    test_scores = result_dict.get("scores", {}).get("test", [])
    ndcg_10 = None
    mrr = None
    recall_10 = None
    if test_scores and isinstance(test_scores, list):
        score_data = test_scores[0]
        ndcg_10 = score_data.get("ndcg_at_10")
        mrr = score_data.get("mrr_at_10") or score_data.get("mrr")
        recall_10 = score_data.get("recall_at_10")

    logger.info("==========================================")
    logger.info("EXPERIMENT 0 BASELINE RESULTS:")
    logger.info("Task: AppsRetrieval (test split)")
    logger.info("Model: %s", model_name)
    logger.info("NDCG@10: %s", f"{ndcg_10:.4f}" if ndcg_10 is not None else "N/A")
    logger.info("MRR: %s", f"{mrr:.4f}" if mrr is not None else "N/A")
    logger.info("Recall@10: %s", f"{recall_10:.4f}" if recall_10 is not None else "N/A")
    logger.info("Elapsed time: %.2fs", elapsed_time)
    logger.info("==========================================")

    # Update RESULTS.md
    update_results_md(results_md_path, model_name, ndcg_10, mrr, recall_10, elapsed_time)

    return result_dict


def update_results_md(
    results_path: str,
    model_name: str,
    ndcg_10: float | None,
    mrr: float | None,
    recall_10: float | None,
    elapsed_seconds: float,
) -> None:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ndcg_str = f"{ndcg_10:.4f}" if ndcg_10 is not None else "N/A"
    mrr_str = f"{mrr:.4f}" if mrr is not None else "N/A"
    recall_str = f"{recall_10:.4f}" if recall_10 is not None else "N/A"
    latency_str = f"{elapsed_seconds:.1f}s"

    content = (
        "# Experiment Results Log\n\n"
        "Tracking all iterations against the Experiment 0 floor as mandated by the project architecture.\n\n"
        "| Exp # | Hypothesis / Description | Model | NDCG@10 | MRR | Recall@10 | Latency | Status |\n"
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|\n"
        f"| **0** | Baseline: e5-base-v2, exact cosine retrieval, no preprocessing | `{model_name}` | **{ndcg_str}** | **{mrr_str}** | **{recall_str}** | {latency_str} | Completed |\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Updated %s with official baseline results", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTEB AppsRetrieval Baseline Evaluation")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2", help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--output", type=str, default="eval/outputs/appsretrieval_results.json", help="Path to output JSON")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
