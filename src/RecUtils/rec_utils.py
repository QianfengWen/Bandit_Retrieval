from src.Evaluation.evaluation import precision_k, recall_k, mean_average_precision_k
from collections import defaultdict
import pandas as pd

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