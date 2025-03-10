from src.Retrieval.bandit_retrieval import bandit_retrieval
from src.LLM.ChatGPT import ChatGPT
from src.Dataset.travel_dest import TravelDest
from src.Embedding.embedding import create_embeddings, load_embeddings
from src.Evaluation.evaluation import precision_k, recall_k, mean_average_precision_k

import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict



def handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, query_texts, passage_texts):
    if model_name and os.path.exists(query_embeddings_path) and os.path.exists(passage_embeddings_path):
        print(f"Loading embeddings from {query_embeddings_path} and {passage_embeddings_path}")
        return load_embeddings(query_embeddings_path, passage_embeddings_path)
    else:
        print(f"Creating embeddings for {model_name}")
        return create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path)


def fusion_score(passage_ids, scores, passage_city_map, top_k_passages=50, return_scores=False):
    """
    Aggregate scores by city based on top-rated passages.

    Args:
        passage_ids (list): List of passage IDs.
        scores (list): List of scores corresponding to each passage.
        passage_city_map (dict): Mapping from passage ID to city.
        top_k_passages (int): Number of top-rated passages to consider for each city.
        return_scores (bool): If True, return both cities and scores.

    Returns:
        list or dict: 
            - If return_scores=False: List of cities sorted by average top-k score.
            - If return_scores=True: Dict of {city: average score}, sorted by score.
    """
    city_scores = defaultdict(list)

    # aggregate scores by city
    for pid, score in zip(passage_ids, scores):
        city = passage_city_map.get(pid)
        if city is not None:
            city_scores[city].append(score)

    # compute average top-k score for each city
    city_average_scores = {}
    for city, city_score_list in city_scores.items():
        top_k_scores = sorted(city_score_list, reverse=True)[:top_k_passages]
        city_average_scores[city] = sum(top_k_scores) if top_k_scores else 0.0

    # sort cities by average score in descending order
    sorted_cities = sorted(city_average_scores.items(), key=lambda x: x[1], reverse=True)

    if return_scores:
        return dict(sorted_cities)  # return {city: score} if return_scores=True
    else:
        return [city for city, _ in sorted_cities] 


def eval_rec(cities, ground_truth, k, verbose=False):
    """
    Evaluate the recommendation performance.

    Args:
        cities (list or dict): List or sorted dict of recommended cities.
        ground_truth (list): List of ground truth cities.
        k (int): Number of top recommendations to evaluate.
        verbose (bool): Whether to print debug information.

    Returns:
        tuple: (precision@k, recall@k, mean average precision@k)
    """
    # handle both list and dict input
    if isinstance(cities, dict):
        top_k_cities = list(cities.keys())[:min(k, len(cities))]
    elif isinstance(cities, list):
        top_k_cities = cities[:min(k, len(cities))]
    else:
        raise TypeError("cities must be a list or a sorted dict")

    if verbose:
        print("\n--- Evaluation ---")
        print(f"Ground truth: {', '.join(map(str, ground_truth))}")

        try:
            print(f"Top {k} cities: {', '.join(map(str, top_k_cities))}")
        except UnicodeEncodeError:
            # Handle encoding issues for non-ASCII characters
            encoded_cities = [city.encode('ascii', 'replace').decode() for city in top_k_cities]
            print(f"Top {k} cities: {', '.join(encoded_cities)}")

    # compute metrics
    prec_k = precision_k(top_k_cities, ground_truth, k)
    rec_k = recall_k(top_k_cities, ground_truth, k)
    map_k = mean_average_precision_k(top_k_cities, ground_truth, k)

    return prec_k, rec_k, map_k



def main():
    ############## Load Dataset ##############
    dataset = TravelDest()
    dataset_name = "travel_dest"
    model_name = "all-MiniLM-L6-v2"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)
    
    ############## Parameter ##############
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    beta = 2.0
    llm_budget = 50
    k_cold_start = 50
    batch_size = 10
    verbose = True
    k_eval = 50
    k_start = 10
    top_k_passages = 3
    k_retrieval = len(passages)
    
    baseline_prec_k = defaultdict(list)
    baseline_rec_k = defaultdict(list)
    baseline_map_k = defaultdict(list)
    bandit_prec_k = defaultdict(list)
    bandit_rec_k = defaultdict(list)
    bandit_map_k = defaultdict(list)

    ############## Bandit Retrieval ##############
    for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Query", total=len(queries)):
        if verbose:
            print(f"Query: {query}")
        
        bandit_res, bandit_score, baseline_res, baseline_score = bandit_retrieval(
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
            return_score=True
        )

        if verbose:
            print("\n********* Results: **********")

        k_start = 10
        while k_start <= k_eval:
            bandit_cities = fusion_score(bandit_res, bandit_score, passage_to_city, top_k_passages=top_k_passages, return_scores=False)
            prec_k, rec_k, map_k = eval_rec(bandit_cities, list(relevance_map[query_id].keys()), k_start, verbose=verbose)

            baseline_cities = fusion_score(baseline_res, baseline_score, passage_to_city, top_k_passages=top_k_passages, return_scores=False)
            prec_k_baseline, rec_k_baseline, map_k_baseline = eval_rec(baseline_cities, list(relevance_map[query_id].keys()), k_start, verbose=verbose)
            
            baseline_prec_k[k_start].append(prec_k_baseline)
            baseline_rec_k[k_start] .append(rec_k_baseline)
            baseline_map_k[k_start].append(map_k_baseline)
            bandit_prec_k[k_start].append(prec_k)
            bandit_rec_k[k_start].append(rec_k)
            bandit_map_k[k_start].append(map_k)
            if verbose:
                print(f"Precision@{k_start}: {prec_k_baseline}")
                print(f"Precision@{k_start} Bandit: {prec_k}\n")
                print(f"Recall@{k_start}: {rec_k_baseline}")
                print(f"Recall@{k_start} Bandit: {rec_k}\n")
                print(f"MAP@{k_start}: {map_k_baseline}")
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
    

    for k in baseline_prec_k.keys():
        print(f"Precision@{k}: {np.mean(baseline_prec_k[k])}")
        print(f"Precision@{k} Bandit: {np.mean(bandit_prec_k[k])}\n")
        print(f"Recall@{k}: {np.mean(baseline_rec_k[k])}")
        print(f"Recall@{k} Bandit: {np.mean(bandit_rec_k[k])}\n")
        print(f"MAP@{k}: {np.mean(baseline_map_k[k])}")
        print(f"MAP@{k} Bandit: {np.mean(bandit_map_k[k])}\n")
        
if __name__ == "__main__":
    main() 