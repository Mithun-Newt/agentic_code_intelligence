# Samsung PRISM Generative AI Hackathon (3rd Edition 2026–27)
## Slide Deck Content: Theme 01 — Agentic Code Intelligence

> **Template Reference**: `CollegeName_TeamName_Submission.pptx`  
> **Official Topic**: Theme 01 — Agentic Code Intelligence  
> **Evaluation Phase**: P0 Retrieval Accuracy Screening & Prototype Evaluation  
> **Notice**: All metrics, models, execution times, and architecture details reflect strictly verified experimental facts from the codebase.

---

### Slide 1: Title Slide

- **Event**: SAMSUNG PRISM Generative AI Hackathon (3rd Edition 2026–27)
- **Theme ID**: Theme 01 — Agentic Code Intelligence
- **Project Title**: High-Precision Dense Code Retrieval Engine for Algorithmic Codebases
- **Team Name**: CodeNexus
- **College Name**: Vellore Institute of Technology
- **Team Members & Emails**:
  - Member 1: Mithun Venkatesan — mithun.venkatesan2023@vitstudent.ac.in
  - Member 2: Kavin M — kavin.m2023@vitstudent.ac.in
  - Member 3: Akhiil Dheep B — akhiildheep.b2023@vitstudent.ac.in
  - Member 4: None
- **Submission GitHub Link**: `https://github.com/Mithun-Newt/agentic_code_intelligence.git`

---

### Slide 2: Theme

- **Domain**: Agentic Code Intelligence & Code Retrieval
- **Problem Context**:
  - AI agents analyzing and modifying codebases spend the majority of latency and context budget locating relevant files and code blocks.
  - Core bottleneck: Given a natural language query describing requirements, edge cases, or algorithm behavior, retrieve and rank the most relevant code snippets from thousands of candidates.
- **Official Samsung Hackathon Scope**:
  - **In-Scope**: Retrieval accuracy improvement, query and document pre/post-processing, embedding model optimization, exact cosine similarity retrieval on CPU.
  - **Out-of-Scope**: Final LLM answer generation, code explanation generation, conversational chatbots, or LLM-based rerankers exceeding screening resource boundaries.
- **Target Goal**: P0 Retrieval Accuracy on the official **CoIR AppsRetrieval** benchmark (3,765 test queries, 8,765 test code documents).

---

### Slide 3: Existing Solutions & Gaps

- **Baseline Architecture (Experiment 0 Floor)**:
  - Model: `intfloat/e5-base-v2` with default asymmetric prefixes (`query: ` / `passage: `).
  - Retrieval: Flat exact inner product cosine retrieval over unit-normalized embeddings.
  - Official Baseline Performance:
    - **NDCG@10**: `0.1152`
    - **MRR@10**: `0.0988`
    - **Recall@10**: `0.1692`
- **Identified Gaps in Existing Approaches**:
  1. **Extreme Semantic-to-Syntactic Gap**: Natural language queries describe abstract problem statements, while target code contains terse variable names, control flow, and library calls with near-zero literal overlap.
  2. **Sub-optimal Pretraining Alignments**: General-purpose text embedding models (e.g., standard E5) fail to distinguish subtle algorithmic patterns (e.g., dynamic programming vs. greedy recursion).
  3. **High Context / Memory Waste**: Large models or LLM-based ranking pipelines cannot scale to thousands of snippets within standard CPU operational limits.

---

### Slide 4: Our Solutions & Architecture Diagram

- **System Architecture**:
  - Fully compliant with MTEB `AbsEncoder` interface (`PrePostPipelineEncoder`).
  - Native 100% CPU execution using PyTorch multi-threading (10 intra-op threads) and FAISS `IndexFlatIP`.

```
[ Natural Language Query ]
            │
            ▼
[ Asymmetric Task Instruction: "task: search result | query: " ]
            │
            ▼
[ PrePostPipelineEncoder (google/embeddinggemma-300m) ]
            │ (768-dim L2-normalized vector)
            ▼
[ DenseIndex (FAISS IndexFlatIP exact cosine search) ] ◄── [ Pre-Indexed Corpus Vectors ]
            │                                              (Prompt: "title: none | text: ")
            ▼
[ Ranked Top-10 Code Matches + Scores + Latency ]
```

- **Key Technical Decisions**:
  - **Selected Model**: `google/embeddinggemma-300m` (300M parameters, 768 dimensions, cosine similarity).
  - **Instruction Conditioning**: Task-specific prompt formatting (`task: search result | query: ` for queries; `title: none | text: ` for code passages).
  - **Bounded Context**: Capped at 512 tokens for strict CPU latency and memory predictability.
  - **Zero-Friction Fallback**: Built-in automated fallback to un-gated `RedHatAI/embeddinggemma-300m` mirror to guarantee offline judge execution without HuggingFace authentication failures.

---

### Slide 5: Demo & Product Walkthrough

- **Judge-Ready CLI Prototype (`demo.py`)**:
  - Standalone, offline-capable CLI demo using a deterministic 15-algorithm code corpus (`data/demo_corpus.json`).
  - Accepts arbitrary natural language queries via `--query` or interactive terminal input.
  - Outputs ranked top-$k$ matches with rank, cosine similarity score, document ID, title, category, and formatted code snippet.
  - Measures and prints exact query retrieval latency in milliseconds.
- **Walkthrough Example**:
  - **Query**: `"Dijkstra shortest path algorithm with priority queue"`
  - **Retrieval Latency**: `96.27 ms` (CPU)
  - **Top Match (Rank 01)**: `doc_dijkstra` (Score: `0.5828`, Graph Algorithms)
  - **Rank 02**: `doc_bfs_graph` (Score: `0.2847`)
  - **Rank 03**: `doc_knapsack_dp` (Score: `0.2306`)

---

### Slide 6: Tools and Tech Stack Used

- **Core Machine Learning & NLP**:
  - `sentence-transformers` (v6.1.0): Encoder wrapper and vector inference.
  - `transformers` (v4.57.6) & `torch` (v2.14.0+cpu): CPU tensor runtime.
- **Evaluation & Benchmarking**:
  - `mteb` (v2.21.6): Official Massive Text Embedding Benchmark harness for `AppsRetrieval`.
  - `datasets` (HuggingFace): CoIR Apps benchmark test dataset streaming.
- **Vector Indexing & Retrieval**:
  - `faiss-cpu` (v1.15.1): Exact inner product flat index (`IndexFlatIP`) with unit vector normalization.
  - NumPy: Vectorized fallback and metric computations.
- **Quality Assurance & Deployment**:
  - `pytest` (v9.1.1): Automated 9-test test suite across baseline compliance and demo integrity.
  - Docker (`python:3.10-slim`): Reproducible container specification.

---

### Slide 7: Impact & Use Case

- **Developer Productivity**:
  - Reduces the search bottleneck for agentic developer tools (e.g., IDE extensions, autonomous coding agents).
  - Allows agents to pinpoint relevant algorithmic blocks without scanning entire repositories sequentially.
- **Computational Efficiency**:
  - 100% CPU executable; eliminates the requirement for costly GPU infrastructure for code search.
  - Single-query latency under **150 ms** on consumer-grade hardware.
- **Standardized Compatibility**:
  - Adheres directly to MTEB's `AbsEncoder` specification, ensuring zero friction for integration into broader retrieval-augmented generation (RAG) pipelines.

---

### Slide 8: Innovation Highlights, Results and Limitations

- **Official CoIR Apps Test Results (Full Split)**:

| Metric | Experiment 0 Baseline (`e5-base-v2`) | Final P0 Engine (`embeddinggemma-300m`) | Gain (Δ) | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **NDCG@10** | 0.11523 | **0.80492** | **+0.68969** | **+598.5% (~7.0x)** |
| **MRR@10** | 0.09878 | **0.77097** | **+0.67219** | **+680.5% (~7.8x)** |
| **Recall@10** | 0.16919 | **0.91049** | **+0.74130** | **+438.1% (~5.4x)** |
| **Recall@100** | 0.34343 | **0.98114** | **+0.63771** | **+185.7% (~2.9x)** |
| **Recall@1000** | 0.68127 | **0.99655** | **+0.31528** | **+46.3%** |

- **Innovation Highlights**:
  - **Empirical Model Selection**: Rigorous fair-comparison protocol on a 200-query held-out validation slice before committing to full test evaluation.
  - **Asymmetric Instruction Tuning**: Leveraged task-specific instruction prompts to bridge the semantic intent divide.
- **Honest Limitations**:
  - Exact token/identifier matching: Pure dense retrieval occasionally misses arbitrary variable names (lexical hybrid BM25 fusion planned for P1).
  - Sequence truncation: 512-token ceiling truncates very large source files without AST splitting.
  - Linear search: $O(N)$ flat index is optimal for benchmarks (<10k docs) but requires approximate nearest neighbors (HNSW) for multi-million file repos.

---

### Slide 9: What’s Next

- **Phase 1 (P1) — Multi-Version Repository Indexing**:
  - Incremental vector indexing: Update only modified git blobs/commits rather than full corpus re-indexing.
  - Commit-aware caching for rapid sub-second index rebuilds.
- **Hybrid Sparse-Dense Retrieval**:
  - Integrate BM25 token matching with Reciprocal Rank Fusion (RRF) to capture rare symbol names alongside deep semantic intent.
- **AST-Aware Hierarchical Chunking**:
  - Parse code with Tree-sitter into syntax-aware functions and classes to preserve lexical scope across large files.

---

### Slide 10: Brownie Points Slide (Differentiation)

- **Why This Submission Stands Out**:
  1. **Massive Verified Gain**: Delivered a **+0.6897 NDCG@10** (+598%) leap over the baseline on the unmodified official test split.
  2. **Production Code Quality**: No mock numbers or synthetic evaluation. Evaluated all 3,765 test queries and 8,765 test documents on CPU in 112 minutes.
  3. **Judge-First Usability**:
     - 100% offline-capable CLI demo with deterministic code samples.
     - 9 automated unit/integration tests passing with 100% pass rate.
     - Automated gated model mirror fallback prevents runtime authorization crashes.
  4. **Strict Architectural Integrity**: Resisted premature feature bloat; focused entirely on maximizing P0 retrieval accuracy as prioritized in Samsung's guidelines.

---

### Slide 11: Checklist — Updated on Public GitHub

- **Working prototype code — public or shared GitHub repo**: **YES** (`https://github.com/Mithun-Newt/agentic_code_intelligence.git`)
- **README with reproducible setup instructions**: **YES** (Comprehensive setup, demo, test, and evaluation instructions)
- **Demo video, max 5 minutes (YouTube or Drive link)**: **YES** (`https://drive.google.com/file/d/1ygrzH9wNctm0r5rT4pi3kkJrUA_bPuJz/view?usp=sharing`)
- **Presentation file (PPT or PDF)**: **YES** (`submission/CollegeName_TeamName_Submission.pptx` & `CollegeName_TeamName_Submission.pdf`)

---

### Slide 12: Thank You

- **Thank You!**
- Organized by the Language AI Team and the PRISM Team, Samsung R&D Institute India.
- **Q&A**: Ready for technical evaluation and live prototype demonstration.
