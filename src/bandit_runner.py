from src.Retrieval.retrieval import bandit_retrieval
from src.LLM.ChatGPT import ChatGPT
from src.Dataset.travel_dest import TravelDest
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score, eval_rec, save_results

from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm



def main():
    ############## Load Dataset ##############
    dataset = TravelDest()
    dataset_name = "travel_dest"
    model_name = "all-MiniLM-L6-v2"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city, prelabel_relevance = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)
    


    ############## Config ##############
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    beta = 2.0
    llm_budget = 200
    k_cold_start = 200
    batch_size = 5
    k_retrieval = 1000
    cache = prelabel_relevance
    update_cache = "data/travel_dest/update_cache.csv"
    verbose = False
    k_eval = 50
    k_start = 10
    top_k_passages = 3
    save_flag = True

    gpucb_percentage = (llm_budget - k_cold_start) / llm_budget

    configs = {
        "llm_budget": llm_budget,
        "gpucb_percentage": gpucb_percentage,
        "k_retrieval": k_retrieval,
        "top_k_passages": top_k_passages,
        "batch_size": batch_size,
        "beta": beta
    }

    result_path = f"{dataset_name}_{model_name}_bandit_results.csv"


    ############## Evaluation ##############
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Query", total=len(queries)):
        if verbose:
            print(f"Query: {query}")
        
        item, score = bandit_retrieval(
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
            batch_size=batch_size,
            verbose=verbose,
            return_score=True,
            cache=cache,
            update_cache=update_cache
        )

        if verbose:
            print("\n********* Results: **********")

        k_start = 10
        while k_start <= k_eval:
            bandit_cities = fusion_score(item, score, passage_to_city, top_k_passages=top_k_passages, return_scores=False)
            prec_k, rec_k, map_k = eval_rec(bandit_cities, list(relevance_map[query_id].keys()), k_start, verbose=verbose)
            
            prec_k_dict[k_start].append(prec_k)
            rec_k_dict[k_start].append(rec_k)
            map_k_dict[k_start].append(map_k)
            if verbose:
                print(f"Precision@{k_start} Bandit: {prec_k}\n")
                print(f"Recall@{k_start} Bandit: {rec_k}\n")
                print(f"MAP@{k_start} Bandit: {map_k}")
            k_start += 10
        
        if verbose:
            print("\n\n\n\n\n")


    print("=== Bandit Retrieval Demo ===")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Beta: {beta}")
    print(f"LLM Budget: {llm_budget}")
    print(f"Cold Start K: {k_cold_start}")
    print(f"Retrieval K: {k_retrieval}")
    print(f"Batch Size: {batch_size}")
    print(f"Top K Passages: {top_k_passages}")
    
    results = {}

    for k in prec_k_dict.keys():
        print(f"Precision@{k} Bandit: {np.mean(prec_k_dict[k])}\n")
        print(f"Recall@{k} Bandit: {np.mean(rec_k_dict[k])}\n")
        print(f"MAP@{k} Bandit: {np.mean(map_k_dict[k])}\n")
        results[f"precision@{k}"] = np.mean(prec_k_dict[k]).round(4)
        results[f"recall@{k}"] = np.mean(rec_k_dict[k]).round(4)
        results[f"map@{k}"] = np.mean(map_k_dict[k]).round(4)

    if save_flag:
        assert save_results(configs, results, result_path) == True, "Results not saved"
        print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main() 