# Gaussian Process Regression with LLM Relevance Judgements and Dense Kernels for Natural Language Recommendation
---

## 🛠 Prerequisites

- Python 3.8 or higher  
---

## 🔧 Installation

1. **Clone the repository**  
   ```bash
    python3 -m venv venv
    source venv/bin/activate

2. **Install dependencies**
   ```bash
    pip install \
    numpy scipy scikit-learn pandas \
    torch sentence-transformers tqdm \
    matplotlib openai pydantic

## Usage
1. **Baseline Dense Retrieval**
    ```bash
    python -m src.baseline_runner

The default setting uses TravelDest dataset with top_k_passages = 50, k_retrieval = 126400 (all passages) and fusion_mode = sum.

2. **running GPR-LLM** 
    ```bash
    python -m src.gp_runner

The default setting uses TravelDest dataset with LLM budget = 10, sample_strategy = random, kernel = rbf, epsilon = 0, top_k_passages = 3, normalize_y = True, alpha = 1e-1 and length_scale = 1.0.

3. **running LLM Reranking**
    ```bash
    python -m src.llm_rerank_runner

The default setting uses TravelDest dataset with budget = 1 and 5, top_k_passages = 3, fusion_mode = sum, and use cross encoder reranking.