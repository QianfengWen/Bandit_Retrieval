from src.Dataset.travel_dest import TravelDest
from src.Retrieval.retrieval import dense_retrieval
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score, eval_rec

import numpy as np
from tqdm import tqdm
from collections import defaultdict

def main():
    ################### Load Data ###################
    model_name = "all-MiniLM-L6-v2"
    dataset_name = "travel_dest"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    dataset = TravelDest()
    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city, prelabel_relevance = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)



    ################### COnfiguration ###################
    verbose=False
    k_start = 10
    k_eval = 50
    k_retrieval = min(1000, len(prelabel_relevance))
    top_k_passages = 5

    
    
    ################### Evaluation ###################
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    for q_id, query_embedding in tqdm(zip(question_ids, query_embeddings), desc="Query", total=len(question_ids)):
        item, score = dense_retrieval(passage_ids, passage_embeddings, query_embedding, k_retrieval= k_retrieval, return_score=True)
        
        k_start = 10
        bandit_cities = fusion_score(item, score, passage_to_city, top_k_passages=top_k_passages, return_scores=False)
        while k_start <= k_eval:
            prec_k, rec_k, map_k = eval_rec(bandit_cities, list(relevance_map[q_id].keys()), k_start, verbose=verbose)
            prec_k_dict[k_start].append(prec_k)
            rec_k_dict[k_start].append(rec_k)
            map_k_dict[k_start].append(map_k)
            if verbose:
                print(f"Precision@{k_start}: {prec_k}\n")
                print(f"Recall@{k_start}: {rec_k}\n")
                print(f"MAP@{k_start}: {map_k}")
            k_start += 10
            
            if verbose:
                print("\n\n\n\n\n")

        print("=== LLM Reranking Demo ===")
        print(f"K Retrieval: {k_retrieval}")
        print(f"Top K Passages: {top_k_passages}")
        
        for k in prec_k_dict.keys():
            print(f"Precision@{k}: {np.mean(prec_k_dict[k])}\n")
            print(f"Recall@{k}: {np.mean(rec_k_dict[k])}\n")
            print(f"MAP@{k}: {np.mean(map_k_dict[k])}\n")


if __name__ == "__main__":
    main()