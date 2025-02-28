import numpy as np
from src.Retrieval.bandit_retrieval import bandit_retrieval_indices_based, bandit_retrieval_embeddings_based
from src.Retrieval.llm import LLM
from sentence_transformers import SentenceTransformer
import time
from src.Dataset.msmarco import MSMARCO
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
        return load_embeddings(query_embeddings_path, passage_embeddings_path)
    else:
        return create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path)

def main():
    # Sample data
    model_name = "all-MiniLM-L6-v2"
    dataset_name = "scidocs"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"
    scidocs = Scidocs()
    question_ids, queries, passage_ids, passages, relevance_map = scidocs.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)
    
    # Create LLM interface with ground truth relevance
    llm = LLM(relevance_map)
    
    # Parameters
    beta = 2.0
    llm_budget = 500
    k_cold_start = 50
    k_retrieval = 50
    
    # print("=== Bandit Retrieval Demo ===")
    
    # # 1. Indices-based retrieval
    # print("\n=== Indices-based Retrieval ===")
    # for i, query in enumerate(queries):
    #     print(f"\nQuery: {query}")
        
    #     start_time = time.time()
    #     results = bandit_retrieval_indices_based(
    #         passage_ids=passage_ids.copy(),
    #         passages=passages,
    #         llm=llm,
    #         query=query,
    #         query_id=i,
    #         beta=beta,
    #         llm_budget=llm_budget,
    #         k_cold_start=k_cold_start,
    #         k_retrieval=k_retrieval
    #     )
    #     elapsed = time.time() - start_time
        
    #     print(f"Retrieved passages (time: {elapsed:.2f}s):")
    #     for rank, pid in enumerate(results):
    #         print(f"  {rank+1}. {passages[pid][:100]}...")
    
    # 2. Embeddings-based retrieval
    # First, get embeddings for queries and passages
    
    retrieval_results = []
    print("\n=== Embeddings-based Retrieval ===")
    for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Query", total=len(queries)):
        print(f"\nQuery: {query}")
        
        start_time = time.time()
        results = bandit_retrieval_embeddings_based(
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
            k_retrieval=k_retrieval
        )
        elapsed = time.time() - start_time
        retrieval_results.append(results)
        p_k = precision_k(results, relevance_map[query_id], k_retrieval)
        print(f"Precision@{k_retrieval}: {p_k}")
    retrieval_results = np.array(retrieval_results)

    eval_res = evaluate(question_ids, passage_ids, retrieval_results, relevance_map, precision_k, k_retrieval)
    print(f"Precision@{k_retrieval}: {eval_res:.4f}")
        

if __name__ == "__main__":
    main() 