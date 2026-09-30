# AI Usage Disclosure Form
### Samsung PRISM Generative AI Hackathon (3rd Edition 2026–27)

---

### 1. Team Details

- **Team Name**: CodeNexus
- **Project / Product Name**: Agentic Code Intelligence — High-Precision Dense Code Retrieval Engine (Theme 01)
- **Organization / Institution (if any)**: Vellore Institute of Technology
- **Submission Date**: 2026-09-29

---

### 2. AI Usage Declaration

**Did your team use any Artificial Intelligence (AI) in developing this project?**  
**YES**

---

### 3. Purpose of AI Usage (Brief Details)

- **Idea generation / brainstorming**: Explored candidate embedding models (E5, Jina Code, EmbeddingGemma) and task-specific instruction prompt strategies compatible with MTEB `AppsRetrieval` and CPU execution boundaries.
- **Code generation or assistance**: Implemented the `PrePostPipelineEncoder(AbsEncoder)` wrapper, FAISS `IndexFlatIP` retrieval index, candidate comparison runner, and interactive CLI demo.
- **UI / UX design**: N/A (Project is a backend retrieval pipeline and command-line search prototype; no graphic UI developed).
- **Content creation**: Generated detailed technical documentation (`README.md`), slide deck content matching the official Samsung template (`submission/PPT_CONTENT.md`), and submission audit checklists.
- **Data analysis**: Analyzed Experiment 1 internal validation metrics across 200 queries and extracted official full-test evaluation metrics from MTEB output JSON.
- **Testing / debugging**: Implemented automated test suites (`test_baseline.py`, `test_demo.py`), resolved MTEB 2.21 datetime JSON serialization issues, and handled HuggingFace gated repository 403 fallback handling.
- **Other**: N/A

---

### 4. Feature Origin Classification

#### Feature 1: Experiment 0 MTEB Baseline Pipeline & Dense Index Utility
1. **Feature Name**: Baseline PrePostPipelineEncoder & FAISS DenseIndex
2. **Classification**: Both (Human-Architected + AI-Assisted)
3. **Description**:
   - **AI Tools Used**: Claude & Google Antigravity
   - **Prompt Purpose**: Implement `PrePostPipelineEncoder` inheriting from MTEB `SentenceTransformerEncoderWrapper` / `AbsEncoder` and create an isolated FAISS `IndexFlatIP` dense indexing utility with NumPy fallback.
   - **Output Produced**: `src/encoder.py`, `src/index/dense_index.py`, `eval/run_mteb.py`.
   - **Human Modification / Review**: Reviewed inheritance compliance with MTEB 2.21, verified unit vector $L_2$ normalization math, and confirmed baseline evaluation on CPU.

---

#### Feature 2: MTEB Task Result Serialization & Datetime Handling
1. **Feature Name**: Robust MTEB JSON Serialization
2. **Classification**: Both (Human-Architected + AI-Assisted)
3. **Description**:
   - **AI Tools Used**: Google Antigravity
   - **Prompt Purpose**: Debug and resolve `TypeError: Object of type datetime is not JSON serializable` occurring during `json.dump()` of MTEB `TaskResult.to_dict()`.
   - **Output Produced**: Added `default=str` serialization handler in `eval/run_mteb.py`.
   - **Human Modification / Review**: Validated that saved JSON artifacts are strictly valid JSON and retain all metric keys without corruption.

---

#### Feature 3: Experiment 1 Candidate Model Comparison Runner
1. **Feature Name**: Offline Validation Slice Evaluation Suite
2. **Classification**: Both (Human-Architected + AI-Assisted)
3. **Description**:
   - **AI Tools Used**: Claude & Google Antigravity
   - **Prompt Purpose**: Build a deterministic validation harness (200 held-out train queries vs. 1,500 candidate documents) to benchmark `intfloat/e5-base-v2`, `jina-embeddings-v2-base-code`, and `google/embeddinggemma-300m` under identical flat cosine retrieval conditions.
   - **Output Produced**: `configs/experiment_1_models.yaml`, `eval/run_experiment.py`, `eval/results/experiment_1_*.json`.
   - **Human Modification / Review**: Configured fixed random seed (42), reviewed per-model asymmetric query/passage prompts, capped sequence lengths to 512 for fair CPU evaluation, and selected `embeddinggemma-300m` as the decisive winner.

---

#### Feature 4: Gated Model Fallback & Asymmetric Prompt Conditioning
1. **Feature Name**: EmbeddingGemma Integration with Automated Mirror Fallback
2. **Classification**: Both (Human-Architected + AI-Assisted)
3. **Description**:
   - **AI Tools Used**: Google Antigravity
   - **Prompt Purpose**: Enable `google/embeddinggemma-300m` to execute seamlessly in unauthenticated or offline judge environments by falling back to the identical-weight `RedHatAI/embeddinggemma-300m` mirror, and attach official task prompts.
   - **Output Produced**: Mirror fallback and prompt mapping logic in `src/encoder.py`.
   - **Human Modification / Review**: Verified weight and architecture identity, tested fallback behavior under synthetic 403 errors, and confirmed 768-dimensional normalized output.

---

#### Feature 5: Standalone Interactive CLI Demo & Deterministic Code Corpus
1. **Feature Name**: Judge-Ready CLI Code Search Prototype
2. **Classification**: Both (Human-Architected + AI-Assisted)
3. **Description**:
   - **AI Tools Used**: Google Antigravity
   - **Prompt Purpose**: Create an offline-ready, standalone CLI tool (`demo.py`) that indexes a curated 15-algorithm code corpus (`data/demo_corpus.json`), accepts natural language queries, and outputs ranked code snippets with latency.
   - **Output Produced**: `demo.py`, `data/demo_corpus.json`.
   - **Human Modification / Review**: Curated representative algorithms (binary search, Dijkstra, LRU cache, merge sort), verified score rankings, and silenced library warnings for clean CLI output.

---

#### Feature 6: Automated Test Suite & Regression Verification
1. **Feature Name**: Comprehensive Pytest Suite
2. **Classification**: Both (Human-Architected + AI-Assisted)
3. **Description**:
   - **AI Tools Used**: Google Antigravity
   - **Prompt Purpose**: Create automated test cases covering encoder AbsEncoder compliance, vector unit length, FAISS exact search, demo corpus integrity, and CLI subprocess execution.
   - **Output Produced**: `tests/test_baseline.py`, `tests/test_demo.py`, `pytest.ini`.
   - **Human Modification / Review**: Verified that all 9 tests pass with 100% pass rate in CI/local runs.

---

#### Feature 7: Containerization & Submission Documentation
1. **Feature Name**: Docker Environment & Technical Documentation
2. **Classification**: Both (Human-Architected + AI-Assisted)
3. **Description**:
   - **AI Tools Used**: Google Antigravity
   - **Prompt Purpose**: Generate a minimal `python:3.10-slim` container definition, detailed project documentation (`README.md`), slide deck markdown (`submission/PPT_CONTENT.md`), and final submission checklists.
   - **Output Produced**: `Dockerfile`, `.dockerignore`, `README.md`, `submission/PPT_CONTENT.md`, `submission/SUBMISSION_CHECKLIST.md`.
   - **Human Modification / Review**: Verified all documented commands, verified metric numbers against official test output, and ensured no unverified or out-of-scope features were claimed.

---

### 5. Ethical & Compliance Confirmation

- **AI usage complies with hackathon guidelines and policies**: **YES**
- **No proprietary, copyrighted, or unauthorized data misused**: **I AGREE**

---

### 6. Declaration & Sign-Off

- **Name of Team Representative**: Mithun Venkatesan
- **Role**: Team Lead (CodeNexus)
- **Signature**: Mithun Venkatesan
- **Date**: 2026-09-29
