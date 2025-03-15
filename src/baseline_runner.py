from src.Dataset.travel_dest import TravelDest
from src.Retrieval.retrieval import dense_retrieval
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score, eval_rec, save_results

import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
def main():
    ################### Load Data ###################
    model_name = "all-MiniLM-L6-v2"
    dataset_name = "travel_dest"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    dataset = TravelDest()
    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city, prelabel_relevance = dataset.load_data()
    with open('data/travel_dest/passages.json', 'w') as f:
        json.dump(passages, f, indent=4)
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)



    ################### Configuration ###################
    verbose=False
    k_start_initial = 10
    k_eval = 10
    k_retrieval = 1000
    top_k_passages = 50
    save_flag = True
    
    configs = {
        "k_retrieval": k_retrieval,
        "top_k_passages": top_k_passages
    }

    result_path = f"{dataset_name}_{model_name}_baseline_results.csv"
    
    ################### Evaluation ###################
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    for q_id, query_embedding in tqdm(zip(question_ids, query_embeddings), desc="Query", total=len(question_ids)):
        print("=== Dense Retrieval Demo ===")
        print("query_id: ", q_id)
        print("query_embedding: ", query_embedding.shape)
        print("passage_embeddings: ", passage_embeddings.shape)
        item, score = dense_retrieval(passage_ids, passage_embeddings, query_embedding, k_retrieval = k_retrieval, return_score=True)
        print("item: ", item)
        print("score: ", score)
        for pid, sc in zip(item, score):
            print("\npassage: ", passages[pid].encode('utf-8')  )
            print("score: ", sc)
        
        k_start = k_start_initial
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
        
        results = {}
        for k in prec_k_dict.keys():
            print(f"Precision@{k}: {np.mean(prec_k_dict[k])}\n")
            print(f"Recall@{k}: {np.mean(rec_k_dict[k])}\n")
            print(f"MAP@{k}: {np.mean(map_k_dict[k])}\n")
            results[f"precision@{k}"] = np.mean(prec_k_dict[k]).round(4)
            results[f"recall@{k}"] = np.mean(rec_k_dict[k]).round(4)
            results[f"map@{k}"] = np.mean(map_k_dict[k]).round(4)

        break

    if save_flag:
        assert save_results(configs, results, result_path) == True, "Results not saved"
        print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main()