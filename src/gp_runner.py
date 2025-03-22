from src.Retrieval.retrieval import gp_retrieval
from src.LLM.ChatGPT import ChatGPT
from src.Dataset.travel_dest import TravelDest
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score, eval_rec, save_results

from collections import defaultdict
import numpy as np
import os, json
from tqdm import tqdm



def main(llm_budget=200, top_k=100, k_retrieval=1000, top_k_passages=5, batch_size=5):
    ############## Load Dataset ##############
    dataset = TravelDest()
    dataset_name = "travel_dest"
    model_name = "all-MiniLM-L6-v2"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city, prelabel_relevance = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)
    
    ############## Config ##############
    cache_path = f"data/{dataset_name}/cache.csv"
    output_prefix = f"output/{dataset_name}/{model_name}"
    
    os.makedirs(f"{output_prefix}/evaluation_results", exist_ok=True)
    os.makedirs(f"{output_prefix}/retrieval_results", exist_ok=True)


    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    llm_budget = llm_budget
    top_k = top_k
    batch_size = batch_size
    k_retrieval = k_retrieval
    # cache = None
    cache = dataset.load_cache()
    # update_cache = cache_path
    update_cache = None
    verbose = False
    k_eval = 50
    k_start = 10
    top_k_passages = top_k_passages
    save_flag = True

    configs = {
        "llm_budget": llm_budget,
        "top_k_sample": top_k,
        "k_retrieval": k_retrieval,
        "top_k_passages": top_k_passages,
        "batch_size": batch_size
    }
    evaluation_path = f"{output_prefix}/evaluation_results/{model_name}_{llm_budget}_{top_k}_{k_retrieval}_{top_k_passages}_{batch_size}.csv"
    retrieval_results_path = f"{output_prefix}/retrieval_results/{model_name}_{llm_budget}_{top_k}_{k_retrieval}_{top_k_passages}_{batch_size}.json"

    ############## Evaluation ##############
    retrieval_cities = defaultdict()
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Query", total=len(queries)):
        if verbose:
            print(f"Query: {query}")
        
        item, score = gp_retrieval(
            passage_ids=passage_ids.copy(),
            passage_embeddings=passage_embeddings,
            passages=passages,
            llm=llm,
            query=query,
            query_embedding=query_embeddings[i],
            query_id=query_id,
            llm_budget=llm_budget,
            top_k=top_k,
            k_retrieval=k_retrieval,
            batch_size=batch_size,
            verbose=verbose,
            return_score=True,
            cache=cache,
            update_cache=update_cache
        )

        if verbose:
            print("\n********* Results: **********")

        bandit_cities = fusion_score(item, score, passage_to_city, top_k_passages=top_k_passages, return_scores=False)
        retrieval_cities[query_id] = bandit_cities

        k_start = 10
        while k_start <= k_eval:
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


    print("=== GP Retrieval Demo ===")
    
    results = {}

    for k in prec_k_dict.keys():
        print(f"Precision@{k} Bandit: {np.mean(prec_k_dict[k])}\n")
        print(f"Recall@{k} Bandit: {np.mean(rec_k_dict[k])}\n")
        print(f"MAP@{k} Bandit: {np.mean(map_k_dict[k])}\n")
        results[f"precision@{k}"] = np.mean(prec_k_dict[k]).round(4)
        results[f"recall@{k}"] = np.mean(rec_k_dict[k]).round(4)
        results[f"map@{k}"] = np.mean(map_k_dict[k]).round(4)

    if save_flag:
        with open(retrieval_results_path, "w", encoding='utf-8') as f:
            json.dump(retrieval_cities, f, indent=4)
        assert save_results(configs, results, evaluation_path) == True, "Results not saved"
        print(f"Results saved to {evaluation_path}")

if __name__ == "__main__":
    main(llm_budget=200, top_k=100)