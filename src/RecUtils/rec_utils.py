from collections import defaultdict
import pandas as pd
import numpy as np

from ..Evaluation.evaluation import precision_k, recall_k, mean_average_precision_k, ndcg_k
from ..GPR.gp import GaussianProcess


def fusion_score(
    passage_ids,
    scores,
    passage_city_map,
    top_k_passages=50,
    return_scores=False,
    fusion_mode="average",
):
    """Aggregate passage-level scores into city-level scores."""
    fusion_fn = {
        "average": lambda vals: sum(vals) / len(vals) if vals else 0.0,
        "sum": lambda vals: sum(vals) if vals else 0.0,
        "max": lambda vals: max(vals) if vals else 0.0,
    }
    if fusion_mode not in fusion_fn:
        raise ValueError("fusion_mode must be one of {'average', 'sum', 'max'}")

    city_scores = defaultdict(list)
    for pid, score in zip(passage_ids, scores):
        city = passage_city_map.get(pid)
        if city is not None:
            city_scores[city].append(score)

    aggregated = {
        city: fusion_fn[fusion_mode](sorted(values, reverse=True)[:top_k_passages])
        for city, values in city_scores.items()
    }

    ranked = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked) if return_scores else [city for city, _ in ranked]


def fusion_score_gp(
    gp: GaussianProcess,
    passage_ids: list,
    passage_dict: dict,
    passage_embeddings: np.array,
    top_k_passages=5,
    k_retrieval=1000,
    return_scores=False,
    fusion_method="mean",
):
    """Aggregate Gaussian Process scores at the city level."""
    fusion_fn = {
        "mean": lambda vals: sum(vals) / len(vals) if vals else 0.0,
        "sum": lambda vals: sum(vals) if vals else 0.0,
        "max": lambda vals: max(vals) if vals else 0.0,
    }
    if fusion_method not in fusion_fn:
        raise ValueError("fusion_method must be one of {'mean', 'sum', 'max'}")

    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}
    _, global_scores = gp.get_top_k(passage_embeddings, k_retrieval, return_scores=True)
    cutoff = global_scores[-1] if global_scores else float("-inf")

    city_scores = {}
    for city_id, city_passage_ids in passage_dict.items():
        if not city_passage_ids:
            city_scores[city_id] = 0.0
            continue

        city_indices = [id_to_index[pid] for pid in city_passage_ids]
        city_embeddings = passage_embeddings[city_indices]
        if len(city_embeddings) == 0:
            city_scores[city_id] = 0.0
            continue

        _, selected_scores = gp.get_top_k(
            city_embeddings,
            top_k_passages,
            return_scores=True,
        )
        filtered = [score for score in selected_scores if score > cutoff]
        city_scores[city_id] = fusion_fn[fusion_method](filtered)

    ranked = sorted(city_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked) if return_scores else [city for city, _ in ranked]


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
    ndcg = ndcg_k(top_k_cities, ground_truth, k)

    return prec_k, rec_k, map_k, ndcg


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
