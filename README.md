# Samsung PRISM Hackathon (3rd Edition) - Theme 01: Agentic Code Intelligence

## Experiment 0: Baseline Code Retrieval Pipeline

This repository implements the official baseline evaluation pipeline for **Theme 01 — Agentic Code Intelligence** of the Samsung PRISM Generative AI Hackathon.

### Architecture (Experiment 0: Baseline)

The baseline establishes the empirical performance floor using the official unmodified evaluation benchmark:

```
CoIR Apps Dataset (APPS Problem Specs & Python Solutions)
       │
       ▼
PrePostPipelineEncoder (wrapping intfloat/e5-base-v2)
  - Query prefix: "query: "
  - Document prefix: "passage: "
  - L2-normalized embeddings (unit length)
       │
       ▼
Exact Dense Cosine Retrieval (Inner Product on Unit Vectors)
       │
       ▼
MTEB AppsRetrieval Evaluation Harness
       │
       ▼
Outputs:
  - NDCG@10 and MRR metrics
  - appsretrieval_results.json (Official MTEB format)
  - eval/RESULTS.md (Experiment tracking table)
```

---

### Project Structure

```
├── README.md                 # Project and baseline documentation
├── requirements.txt          # Minimal, pinned CPU-compatible dependencies
├── pytest.ini                # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── encoder.py            # PrePostPipelineEncoder(AbsEncoder) wrapping e5-base-v2
│   └── index/
│       ├── __init__.py
│       └── dense_index.py    # FAISS IndexFlatIP exact cosine search (with NumPy fallback)
├── eval/
│   ├── run_mteb.py           # Official MTEB AppsRetrieval baseline evaluation runner
│   ├── RESULTS.md            # Experiment scorecard tracking NDCG@10, MRR, latency
│   └── outputs/              # Destination for generated evaluation JSON files
└── tests/
    └── test_baseline.py      # Automated smoke tests for encoder and retrieval index
```

---

### Environment Setup

1. **Activate Python Virtual Environment (Python 3.10+)**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

### Running Smoke Tests

Run the test suite covering the MTEB `AbsEncoder` interface, L2 normalization, FAISS/NumPy indexing, and end-to-end mini-retrieval:

```powershell
python -m pytest tests/test_baseline.py -v
```

---

### Running Experiment 0 Evaluation

To run the official baseline evaluation against MTEB's `AppsRetrieval` benchmark:

```powershell
python eval/run_mteb.py --batch-size 64
```

This will:
1. Load `intfloat/e5-base-v2` onto CPU.
2. Download and stream the `CoIR-Retrieval/apps` test split.
3. Compute dense embeddings with E5 prompt prefixes.
4. Compute exact cosine similarity scores.
5. Write the submission-ready JSON to `eval/outputs/appsretrieval_results.json` (and `appsretrieval_results.json`).
6. Update `eval/RESULTS.md` with the baseline `NDCG@10` and `MRR`.
