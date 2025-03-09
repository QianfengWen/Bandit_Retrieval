import numpy as np
from src.Retrieval.bandit_retrieval import rec_retrieval
from src.LLM.ChatGPT import ChatGPT
# from sentence_transformers import SentenceTransformer
from src.Dataset.antique import Antique
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.scidocs import Scidocs
from src.Dataset.travel_dest import TravelDest
from src.Embedding.embedding import create_embeddings, load_embeddings
from src.Evaluation.evaluation import precision_k, recall_k, mean_average_precision_k, evaluate
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


def eval_rec(passage_ids, scores, passage_city_map, ground_truth, k):
    """
    """
    city_scores = {}
    for passage_id, score in zip(passage_ids, scores):
        city = passage_city_map[passage_id]
        if city not in city_scores:
            city_scores[city] = []
        city_scores[city].append(score)
    
    city_scores = {k: np.mean(v) for k, v in city_scores.items()}
    top_k_cities = sorted(city_scores, key=city_scores.get, reverse=True)[:k]
    try:
        print(f"Top {k} cities: {top_k_cities}")
    except UnicodeEncodeError:
        # Encode each city name individually and create a new encoded list
        encoded_cities = [city.encode('ascii', 'replace').decode() for city in top_k_cities]
        print(f"Top {k} cities: {encoded_cities}")
    print(f"Ground truth: {ground_truth}")
    prec_k = precision_k(top_k_cities, ground_truth, k)
    rec_k = recall_k(top_k_cities, ground_truth, k)
    map_k = mean_average_precision_k(top_k_cities, ground_truth, k)

    return prec_k, rec_k, map_k


def main():
    # Sample data
    model_name = "all-MiniLM-L6-v2"
    dataset_name = "travel_dest"
    method = "embeddings"
    dataset = TravelDest()
    # method = "score"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"
    
    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)
    
    # Create LLM interface with ground truth relevance
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Parameters
    beta = 2.0
    llm_budget = 50
    k_cold_start = 20
    k_retrieval = 1000
    batch_size = 5
    
    # print("=== Bandit Retrieval Demo ===")
    baseline_prec_k = dict()
    baseline_rec_k = dict()
    baseline_map_k = dict()
    bandit_prec_k = dict()
    bandit_rec_k = dict()
    bandit_map_k = dict()

    # First, get embeddings for queries and passages
    print("\n=== Embeddings-based Retrieval ===")
    for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Query", total=len(queries)):
        print(f"\nQuery: {query}")
        
        bandit_res, bandit_score, baseline_res, baseline_score = rec_retrieval(
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
            relevance_map=relevance_map
        )

        print("\n********* Bandit Retrieval Results: **********")
        k_eval = 50
        k_start = 10
        while k_start <= k_eval:
            prec_k, rec_k, map_k = eval_rec(bandit_res, bandit_score, passage_to_city, list(relevance_map[query_id].keys()), k_start)
            prec_k_baseline, rec_k_baseline, map_k_baseline = eval_rec(baseline_res, baseline_score, passage_to_city, list(relevance_map[query_id].keys()), k_start)
            print(f"\nPrecision@{k_start}: {prec_k}")
            print(f"Precision@{k_start} Baseline: {prec_k_baseline}\n")
            print(f"Recall@{k_start}: {rec_k}")
            print(f"Recall@{k_start} Baseline: {rec_k_baseline}\n")
            print(f"MAP@{k_start}: {map_k}")
            print(f"MAP@{k_start} Baseline: {map_k_baseline}\n")
            baseline_prec_k[k_start] = prec_k_baseline
            baseline_rec_k[k_start] = rec_k_baseline
            baseline_map_k[k_start] = map_k_baseline
            bandit_prec_k[k_start] = prec_k
            bandit_rec_k[k_start] = rec_k
            bandit_map_k[k_start] = map_k
            k_start += 10



    print("=== Bandit Retrieval Demo ===")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Method: {method}")
    print(f"Beta: {beta}")
    print(f"LLM Budget: {llm_budget}")
    print(f"Cold Start K: {k_cold_start}")
    print(f"Retrieval K: {k_retrieval}")
    print(f"Batch Size: {batch_size}")
    
    # bandit_retrieval_results = np.array(bandit_retrieval_results)
    # baseline_retrieval_results = np.array(baseline_retrieval_results)

    for k in baseline_prec_k.keys():
        print(f"Precision@{k}: {np.mean(baseline_prec_k[k])}")
        print(f"Precision@{k} Bandit: {np.mean(bandit_prec_k[k])}")
        print(f"Recall@{k}: {np.mean(baseline_rec_k[k])}")
        print(f"Recall@{k} Bandit: {np.mean(bandit_rec_k[k])}")
        print(f"MAP@{k}: {np.mean(baseline_map_k[k])}")
        print(f"MAP@{k} Bandit: {np.mean(bandit_map_k[k])}")

    # k = 10
    # while k <= k_retrieval:
    #     baseline_eval_res = evaluate(question_ids, passage_ids, baseline_retrieval_results, relevance_map, precision_k, k)
    #     print(f"\nBaseline Retrieval Precision@{k}: {baseline_eval_res:.4f}")

    #     bandit_eval_res = evaluate(question_ids, passage_ids, bandit_retrieval_results, relevance_map, precision_k, k)
    #     print(f"Bandit Retrieval Precision@{k}: {bandit_eval_res:.4f}")
    #     k += 10
        
if __name__ == "__main__":
    main() 