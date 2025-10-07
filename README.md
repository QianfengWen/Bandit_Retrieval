# Natural Language Recommendation via Multimodal Item Scoring Using Gaussian Process Regression with LLM Relevance Judgments
---

## ⚠️ Repository Setup

- This project stores datasets via Git LFS. Install it before cloning: `git lfs install`.
## 🛠 Prerequisites

- Python 3.8 or higher
---

## 🔧 Installation

1. **Clone the repository**  
   ```bash
    python3 -m venv venv
    source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
    pip install \
    numpy scipy scikit-learn pandas \
    torch sentence-transformers tqdm \
    matplotlib requests pydantic
   ```
3. **Configure API credentials**
   ```bash
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   ```

## Usage
1. **Baseline Dense Retrieval**
    ```bash
    python -m src.baseline_runner

The default run targets the TravelDest dataset with `embedder = all-MiniLM-L6-v2`, `top_k_passages = 3`, and city fusion mode set to `mean`.
Results are saved under `output/baseline/<dataset>/`.

2. **Running LLM-based Relevance Scoring**
    ```bash
    python -m src.llm_relevance_scorer_runner

The default run targets the TravelDest dataset with `embedder = all-MiniLM-L6-v2`, `llm_model = openai/gpt-4o`, `scoring_mode = expected_relevance`, `budget = 100`, `top_k_passages = 3`, and fusion mode `mean`. Scores are fetched in real time via OpenRouter and cached locally. Switch to `pointwise` scoring to use raw ordinal labels.
Evaluation CSVs are stored in `output/llm_score/<dataset>/`.

3. **Running GPR-LLM** 
    ```bash
    python -m src.gp_runner

The default run targets the TravelDest dataset with `embedder = all-MiniLM-L6-v2`, `llm_model = openai/gpt-4o`, `scoring_mode = expected_relevance`, `llm_budget = 100`, `tau = |passages|` (limits ε-exploration to the top dense-ranked candidates), `sample_strategy = random`, `kernel = rbf`, `epsilon = 0.1`, `top_k_passages = 3`, and fusion mode `mean` (city-level average). Additional GP hyperparameters retain their previous defaults.
Evaluation summaries are stored under `output/gpr_llm/<dataset>/`.
