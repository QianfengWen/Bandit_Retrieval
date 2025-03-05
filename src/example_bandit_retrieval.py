import numpy as np
from src.Retrieval.bandit_retrieval import bandit_retrieval_embeddings_based
from src.LLM.ChatGPT import ChatGPT
# from sentence_transformers import SentenceTransformer
from src.Dataset.antique import Antique
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.scidocs import Scidocs
from src.Embedding.embedding import create_embeddings, load_embeddings
from src.Evaluation.evaluation import precision_k, evaluate
import numpy as np
import os
from tqdm import tqdm


def handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, query_texts, passage_texts):
    if model_name and os.path.exists(query_embeddings_path) and os.path.exists(passage_embeddings_path):
        print(f"Loading embeddings from {query_embeddings_path} and {passage_embeddings_path}")
        return load_embeddings(query_embeddings_path, passage_embeddings_path)
    else:
        print(f"Creating embeddings for {model_name}")
        return create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path)

def main():
    # Sample data
    model_name = "all-MiniLM-L6-v2"
    dataset_name = "scidocs"
    method = "embeddings"
    # method = "score"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"
    scidocs = Scidocs()
    question_ids, queries, passage_ids, passages, relevance_map = scidocs.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)
    
    # Create LLM interface with ground truth relevance
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Parameters
    beta = 2.0
    llm_budget = 20
    k_cold_start = 10
    k_retrieval = llm_budget
    batch_size = 3
    
    # print("=== Bandit Retrieval Demo ===")
    
    bandit_retrieval_results, baseline_retrieval_results = [], []
    # # # 1. Indices-based retrieval
    # if method == "score":
    #     print("\n=== Indices-based Retrieval ===")
    #     retrieval_results = []
    #     for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Query", total=len(queries)):
    #         print(f"\nQuery: {query}")
            
    #         start_time = time.time()
    #         bandit_res, baseline_res = bandit_retrieval_indices_based(
    #             passage_ids=passage_ids.copy(),
    #             passage_embeddings=passage_embeddings,
    #             passages=passages,
    #             llm=llm,
    #             query=query,
    #             query_embedding=query_embeddings[i],
    #             query_id=query_id,
    #             beta=beta,
    #             llm_budget=llm_budget,
    #             k_cold_start=k_cold_start,
    #             k_retrieval=k_retrieval
    #         )
    #         elapsed = time.time() - start_time
    #         bandit_retrieval_results.append(bandit_res)
    #         p_k = precision_k(bandit_res, relevance_map[query_id], k_retrieval)
    #         print(f"Precision@{k_retrieval}: {p_k}")

    #         baseline_p_k = precision_k(baseline_res, relevance_map[query_id], k_retrieval)
    #         print(f"Baseline Precision@{k_retrieval}: {baseline_p_k}")

    # 2. Embeddings-based retrieval
    # First, get embeddings for queries and passages
    print("\n=== Embeddings-based Retrieval ===")
    for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Query", total=len(queries)):
        print(f"\nQuery: {query}")
        
        bandit_res, baseline_res = bandit_retrieval_embeddings_based(
            passage_ids=passage_ids.copy(),
            passage_embeddings=passage_embeddings,
            passages=passages,
            llm=llm,
            query=query,
            query_embedding=query_embeddings[i],
            query_id=query_id,
            beta=beta,
            llm_budget=llm_budget,
            k_cold_start=k_cold_start,
            k_retrieval=k_retrieval,
            batch_size=batch_size
        )
        bandit_retrieval_results.append(bandit_res)
        baseline_retrieval_results.append(baseline_res)

        k = k_cold_start
        print("\n********* Bandit Retrieval Results: **********")
        while k <= k_retrieval:
            p_k = precision_k(bandit_res, relevance_map[query_id], k)
            print(f"\nPrecision@{k}: {p_k}")

            baseline_p_k = precision_k(baseline_res, relevance_map[query_id], k)
            print(f"Baseline Precision@{k}: {baseline_p_k}")
            k += 10

    print("=== Bandit Retrieval Demo ===")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Method: {method}")
    print(f"Beta: {beta}")
    print(f"LLM Budget: {llm_budget}")
    print(f"Cold Start K: {k_cold_start}")
    print(f"Retrieval K: {k_retrieval}")
    
    bandit_retrieval_results = np.array(bandit_retrieval_results)
    baseline_retrieval_results = np.array(baseline_retrieval_results)

    k = k_cold_start
    while k <= k_retrieval:
        baseline_eval_res = evaluate(question_ids, passage_ids, baseline_retrieval_results, relevance_map, precision_k, k)
        print(f"\nBaseline Retrieval Precision@{k}: {baseline_eval_res:.4f}")

        bandit_eval_res = evaluate(question_ids, passage_ids, bandit_retrieval_results, relevance_map, precision_k, k)
        print(f"Bandit Retrieval Precision@{k}: {bandit_eval_res:.4f}")
        k += 10
        
if __name__ == "__main__":
    main() 