# Experiment Results Log

Tracking all iterations against the Experiment 0 floor as mandated by the project architecture.

| Exp # | Hypothesis / Description | Model | NDCG@10 | MRR | Recall@10 | Latency | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | Baseline: e5-base-v2, exact cosine retrieval, no preprocessing | `intfloat/e5-base-v2` | **0.1152** | **0.0988** | **0.1692** | 4980.2s | Completed |

## Experiment 1: Embedding Model Comparison (Internal Validation)

Evaluated on held-out train validation slice (200 queries vs. 1,500 corpus candidates, exact flat cosine similarity, CPU).

| Experiment | Change | NDCG@10 | MRR | Recall@10 | Recall@100 | Recall@1000 | Index time | Query latency | Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Exp 1 (Control) | `intfloat/e5-base-v2` | 0.4871 | 0.4723 | 0.5600 | 0.7600 | 0.9850 | 325.0s | 402.5ms | Baseline control; e5 query:/passage: prefixes |
| Exp 1 | `jina-embeddings-v2-base-code` | 0.6687 | 0.6515 | 0.7400 | 0.8750 | 1.0000 | 622.0s | 843.3ms | Code-trained bi-encoder with ALiBi (max_seq=512) |
| Exp 1 | `google/embeddinggemma-300m` | **0.8645** | **0.8471** | **0.9250** | **0.9750** | **0.9900** | 605.4s | 709.8ms | Multilingual 300M Gemma 3; task query + title doc prompts |
