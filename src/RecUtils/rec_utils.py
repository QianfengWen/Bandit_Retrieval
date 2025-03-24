from src.Evaluation.evaluation import precision_k, recall_k, mean_average_precision_k
from src.GPUCB.retrieval_gpucb import RetrievalGPUCB
from collections import defaultdict
import pandas as pd
import numpy as np

def fusion_score(passage_ids, scores, passage_city_map, top_k_passages=50, return_scores=False, fusion_mode="average"):
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
        if fusion_mode == "average":
            city_average_scores[city] = sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0.0
        elif fusion_mode == "max":
            city_average_scores[city] = max(top_k_scores) if top_k_scores else 0.0
        elif fusion_mode == "sum":
            city_average_scores[city] = sum(top_k_scores) if top_k_scores else 0.0
       

    # sort cities by average score in descending order
    sorted_cities = sorted(city_average_scores.items(), key=lambda x: x[1], reverse=True)
    print("sorted_cities: ", sorted_cities[:50])

    if return_scores:
        return dict(sorted_cities)  # return {city: score} if return_scores=True
    else:
        return [city for city, _ in sorted_cities] 


def fusion_score_gp(
        gp: RetrievalGPUCB, 
        passage_ids: list, 
        passage_dict: dict[list], 
        passage_embeddings: np.array, 
        top_k_passages=5, 
        k_retrieval=1000,
        return_scores=False, 
        fusion_method="mean"
    ):
    """
    Aggregate scores by city based on top-rated passages.

    Args:
        gp (RetrievalGPUCB): GP-UCB model instance.
        passage_ids (list): List of passage IDs.
        passage_dict (dict): Mapping from city ID to passage IDs.
        passage_embeddings (np.array): Array of passage embeddings.
        top_k_passages (int): Number of top-rated passages to consider for each city.
        return_scores (bool): If True, return both cities and scores.
        fusion_method (str): "mean" or "max".

    Returns:
        list or dict: 
            - If return_scores=False: List of cities sorted by average top-k score.
            - If return_scores=True: Dict of {city: average score}, sorted by score.
    """

    # Create a lookup for fast ID-to-index mapping (O(1) lookup)
    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}
    city_scores = {}

    _, scores = gp.get_top_k(passage_embeddings, k_retrieval, return_scores=True)
    top_k_cutoff = scores[-1]

    for city_id, city_passage_ids in passage_dict.items():
        # Use list comprehension with direct lookup
        city_passage_idx = [id_to_index[pid] for pid in city_passage_ids]
        city_passage_embeddings = passage_embeddings[city_passage_idx]

        # Get top-k passages and scores
        if len(city_passage_embeddings) > 0:
            _, selected_scores = gp.get_top_k(
                city_passage_embeddings, 
                top_k_passages, 
                return_scores=True
            )
        filtered_scores = [score for score in selected_scores if score > top_k_cutoff]

        if fusion_method == "mean":
            city_scores[city_id] = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0
        elif fusion_method == "max":
            city_scores[city_id] = max(filtered_scores) if filtered_scores else 0
        elif fusion_method == "sum":
            city_scores[city_id] = sum(filtered_scores) if filtered_scores else 0
        else:
            raise ValueError("Invalid fusion method: Use 'mean', 'max' or 'sum'.")
    # print("city_scores: \n", city_scores)

    # Sort cities by score in descending order
    sorted_cities = sorted(city_scores.items(), key=lambda x: x[1], reverse=True)

    if return_scores:
        return dict(sorted_cities)
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


def save_results(configs, results, file_path):
    """
    Save experimental configurations and results to a CSV file.

    Args:
        configs (dict): Dictionary containing the configuration parameters for the experiment.
            Each key-value pair represents a configuration setting.
        results (dict[metric@k: float]): Dictionary containing the evaluation results.
            Each key-value pair represents a metric and its value.
        file_path (str): Path to the CSV file where results will be saved.
            If file exists, results will be appended.

    Returns:
        bool: True if save was successful, False otherwise.

    """
    try:
        # Create a single row DataFrame from configs
        df_new = pd.DataFrame([configs])
        
        # Add results to the same row
        for metric, value in results.items():
            df_new[metric] = value
            
        # If file exists, append; otherwise create new file
        try:
            df_existing = pd.read_csv(file_path)
            # check if the columns are the same
            if set(df_existing.columns) != set(df_new.columns):
                raise ValueError("Columns do not match")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_combined = df_new
            
        # Save to CSV
        df_combined.to_csv(file_path, index=False)
        return True
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False