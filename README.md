# Samsung PRISM Generative AI Hackathon (3rd Edition)
## Theme 01 — Agentic Code Intelligence: Dense Code Retrieval Engine (P0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MTEB Version](https://img.shields.io/badge/MTEB-2.21.6-green.svg)](https://github.com/embeddings-benchmark/mteb)
[![CPU Only](https://img.shields.io/badge/Device-CPU_Optimized-orange.svg)](#cpu-requirement)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## 1. Problem Statement

Code retrieval bridges the semantic divide between human intent expressed in natural language problem descriptions (e.g., algorithm specifications, edge case criteria, performance constraints) and syntactically structured source code implementations.

Standard dense retrieval baselines often fail on algorithmic code (e.g., the **CoIR AppsRetrieval** benchmark) due to:
1. **Severe Lexical Mismatch**: Problem statements rarely contain exact keyword names of the target functions or data structures.
2. **Structural Complexity**: Code exhibits deep hierarchical logic, nested control flow, and token distributions fundamentally distinct from natural language prose.
3. **Sub-optimal Embedding Alignment**: Off-the-shelf embedding models lack sufficient domain-specific instruction conditioning to map algorithmic queries into the code representation subspace.

This project delivers a **production-grade, judge-ready, CPU-optimized code retrieval engine** adhering strictly to the Samsung PRISM Hackathon Theme 01 specification and official engineering architecture.

---

## 2. System Architecture

The retrieval system conforms to the official MTEB `AbsEncoder` interface, enabling zero-overhead plug-and-play evaluation and reproducible standalone execution.

```
                    ┌────────────────────────────────────────────────────────┐
                    │            Natural Language Search Query               │
                    │   e.g. "binary search on sorted array in O(log n)"     │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │       Asymmetric Query Prompt        │
                             │  "task: search result | query: {q}"  │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │       PrePostPipelineEncoder         │
                             │     google/embeddinggemma-300m       │
                             │  (L2-normalized unit embeddings)     │
                             └──────────────────┬───────────────────┘
                                                │  768-dim query vector
                                                ▼
┌─────────────────────────────────┐  Corpus     ┌──────────────────────────────────┐
│      Raw Code Repository /      │  Vectors    │        DenseIndex (FAISS)        │
│       Code Snippet Corpus       ├────────────►│       Exact Inner Product        │
│ (Prompt: "title: none | text: ")│             │  (Cosine Similarity on S^(d-1))  │
└─────────────────────────────────┘             └─────────────────┬────────────────┘
                                                                  │
                                                                  ▼
                                                ┌──────────────────────────────────┐
                                                │    Top-K Ranked Code Matches     │
                                                │  [Rank, Score, ID, Source Code]  │
                                                └──────────────────────────────────┘
```

---

## 3. Retrieval Methodology & Model

### Retrieval Method
- **Similarity Metric**: Exact Cosine Similarity implemented via FAISS `IndexFlatIP` over L2-normalized 768-dimensional unit embeddings ($S^{d-1}$ hypersphere).
- **Asymmetric Prompting**: Query and code documents use specialized task instruction prefixes aligned with the model's contrastive pretraining objective:
  - **Query Prefix**: `task: search result | query: `
  - **Document/Corpus Prefix**: `title: none | text: `
- **Normalization**: Rigorous unit $L_2$ normalization guarantees dot product equates directly to cosine similarity:
  $$\text{sim}(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}}$$

### Selected Model: `google/embeddinggemma-300m`
- **Architecture**: 300M parameter multilingual dense encoder derived from Gemma 3.
- **Embedding Dimension**: 768
- **Context Length**: Capped at 512 tokens for fair and bounded CPU memory execution.
- **Seamless Local Mirroring**: Automated fallback to `RedHatAI/embeddinggemma-300m` (identical architecture and weights) to ensure zero authentication barriers or gated token failures during offline judge execution.

---

## 4. Official Test Results (CoIR AppsRetrieval)

Evaluated on the full official **CoIR AppsRetrieval** `test` split (3,765 test queries, 8,765 test code documents) via MTEB 2.21:

| Metric | Experiment 0 (Baseline Floor) | Final P0 Run (`google/embeddinggemma-300m`) | Absolute Gain (Δ) | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **NDCG@10** | 0.11523 | **0.80492** | **+0.68969** | **+598.5% (~7.0x)** |
| **MRR@10** | 0.09878 | **0.77097** | **+0.67219** | **+680.5% (~7.8x)** |
| **Recall@10** | 0.16919 | **0.91049** | **+0.74130** | **+438.1% (~5.4x)** |
| **Recall@100** | 0.34343 | **0.98114** | **+0.63771** | **+185.7% (~2.9x)** |
| **Recall@1000** | 0.68127 | **0.99655** | **+0.31528** | **+46.3%** |

*All results are saved in the submission artifact [`appsretrieval_results.json`](appsretrieval_results.json) and tracked in [`eval/RESULTS.md`](eval/RESULTS.md).*

---

## 5. Repository Structure

```
├── README.md                           # Comprehensive documentation & evaluation report
├── Dockerfile                          # Minimal reproducible container environment
├── .dockerignore                       # Docker build context exclusions
├── requirements.txt                    # Pinned, CPU-compatible dependencies
├── pytest.ini                          # Automated test discovery configuration
├── appsretrieval_results.json          # Official submission test evaluation results
├── demo.py                             # Judge-ready CLI interactive code search demo
├── configs/
│   └── experiment_1_models.yaml        # Model candidate configurations & prompts
├── data/
│   └── demo_corpus.json                # Deterministic sample code corpus for offline demo
├── src/
│   ├── __init__.py
│   ├── encoder.py                      # PrePostPipelineEncoder(AbsEncoder) wrapping EmbeddingGemma
│   └── index/
│       ├── __init__.py
│       └── dense_index.py              # FAISS IndexFlatIP exact cosine search with NumPy fallback
├── eval/
│   ├── run_mteb.py                     # Official MTEB AppsRetrieval evaluation script
│   ├── run_experiment.py               # Experiment runner for validation slice comparisons
│   ├── RESULTS.md                      # Experiment tracking log with baseline floor & P0 run
│   ├── outputs/
│   │   ├── appsretrieval_results.json  # Standard test evaluation output
│   │   └── appsretrieval_results_baseline.json # Preserved baseline results
│   └── results/                        # Detailed Experiment 1 validation artifacts
└── tests/
    ├── test_baseline.py                # Encoder interface, L2 norm, and FAISS indexing tests
    └── test_demo.py                    # Demo corpus integrity, ranking, and CLI subprocess tests
```

---

## 6. Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mithun-Newt/agentic_code_intelligence.git
   cd agentic_code_intelligence
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 7. How to Run the Demo

The repository includes a clean, standalone CLI demo that indexes the sample corpus and retrieves the top ranked code snippets in real time.

### Single Query Execution
```bash
python demo.py --query "binary search on sorted array in logarithmic time" --top-k 3
```

### Sample Output:
```text
==============================================================================
  SAMSUNG PRISM GENAI HACKATHON (THEME 01) - CODE RETRIEVAL PROTOTYPE
==============================================================================
Query:        "binary search on sorted array in logarithmic time"
Latency:      98.42 ms
Results:      Top 3 retrieved matches
------------------------------------------------------------------------------
Rank 01 | Score: 0.5912 | ID: doc_binary_search (Binary Search Algorithm)
Category: Searching Algorithms
Code Snippet:
    def binary_search(arr: list[int], target: int) -> int:
        """Perform binary search on a sorted integer array.
        
        Returns the 0-based index of target if found, else -1.
        """
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1
------------------------------------------------------------------------------
Rank 02 | Score: 0.3845 | ID: doc_inorder_traversal (Binary Tree Inorder Traversal)
Category: Tree Data Structures
Code Snippet: ...
------------------------------------------------------------------------------
Rank 03 | Score: 0.3621 | ID: doc_trie_prefix_tree (Trie Prefix Tree for String Search)
Category: Data Structures
Code Snippet: ...
------------------------------------------------------------------------------
```

### Interactive CLI Mode
Running without arguments prompts interactively for queries:
```bash
python demo.py
```

---

## 8. Running the Test Suite

Run the full automated test suite covering encoder compliance, $L_2$ normalization, FAISS indexing, ranking determinism, and CLI execution:

```bash
python -m pytest -v
```

Expected result:
```
tests/test_baseline.py::test_encoder_is_abs_encoder PASSED               [ 11%]
tests/test_baseline.py::test_encoder_output_shape_and_norm PASSED        [ 22%]
tests/test_baseline.py::test_dense_index_orthogonal_exact_search PASSED  [ 33%]
tests/test_baseline.py::test_end_to_end_mini_retrieval PASSED            [ 44%]
tests/test_demo.py::test_demo_corpus_integrity PASSED                    [ 55%]
tests/test_demo.py::test_demo_indexing_and_dimension PASSED              [ 66%]
tests/test_demo.py::test_demo_retrieval_ranking PASSED                   [ 77%]
tests/test_demo.py::test_demo_result_structure PASSED                    [ 88%]
tests/test_demo.py::test_demo_cli_execution PASSED                       [100%]

======================= 9 passed in ~50s =======================
```

---

## 9. Running the Official MTEB Evaluation

To replicate the official full-test evaluation against the MTEB `AppsRetrieval` benchmark:

```bash
python eval/run_mteb.py --model "google/embeddinggemma-300m" --batch-size 16 --output "eval/outputs/appsretrieval_results.json"
```

This command:
1. Loads `google/embeddinggemma-300m` via `PrePostPipelineEncoder` onto CPU.
2. Streams the official CoIR Apps test split (8,765 documents, 3,765 test qrels).
3. Encodes queries and corpus with official prompt formatting.
4. Generates the submission-ready JSON artifact at `eval/outputs/appsretrieval_results.json` and updates `eval/RESULTS.md`.

---

## 10. Docker Deployment

A minimal `Dockerfile` is provided for containerized and isolated reproducibility.

### Build the Docker Image
```bash
docker build -t agentic-code-retrieval:latest .
```

### Run the Container Demo
```bash
docker run --rm agentic-code-retrieval:latest --query "Dijkstra shortest path algorithm" --top-k 3
```

### Run Tests inside Container
```bash
docker run --rm --entrypoint python agentic-code-retrieval:latest -m pytest -v
```

---

## 11. CPU Requirement & Resource Profile

- **Device**: 100% CPU-compatible (no GPU, CUDA, or external hardware accelerators required).
- **Parallelism**: Utilizes PyTorch intra-op thread parallelism (`torch.set_num_threads`).
- **Memory Footprint**: Average RAM usage under 3.5 GB during full-corpus batch encoding.
- **Latency**: Single query retrieval over sample corpus executes in **~90–120 ms** on standard consumer CPUs.

---

## 12. Project Limitations & Future Scope

While the P0 dense retrieval pipeline achieves an outstanding **0.80492 NDCG@10** on the official benchmark, the following known boundaries apply:

1. **Exact Identifier Overlap**: Dense semantic retrieval excels at understanding programmatic intent, but rare or custom variable/function names occasionally benefit from exact token matching (hybrid lexical BM25 fusion planned for future stages).
2. **Maximum Sequence Length**: Input code length is capped at 512 tokens; extremely large multi-class files are truncated rather than parsed via AST-based chunking.
3. **Index Scaling**: Exact cosine search (`IndexFlatIP`) is $O(N)$ with respect to corpus size; scaling to millions of source files in production will require hierarchical indexing (e.g., FAISS HNSW or IVF-PQ).
