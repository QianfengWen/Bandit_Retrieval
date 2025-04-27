from src.Retrieval.retrieval import gp_retrieval
from src.LLM.ChatGPT import ChatGPT
from src.Dataset.dataloader import handle_dataset
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score_gp, eval_rec, save_results

from collections import defaultdict
import numpy as np
import os, json
from tqdm import tqdm

def main(
        dataset_name,
        kernel="rbf",
        llm_budget=200, 
        sample_strategy="random", 
        epsilon=0.1,
        city_max_sample=1,
        top_k_passages=1,
        k_retrieval=1000,
        batch_size=5, 
        fusion_method="sum",
        random_seed=42,
        normalize_y=True,
        alpha=1e-5,
        length_scale=1.0
    ):
    ############## Load Dataset ##############
    dataset_name = dataset_name
    dataset = handle_dataset(dataset_name)
   
    model_name = "all-MiniLM-L6-v2"
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    # Load data
    question_ids, queries, passage_ids, passages, relevance_map, passage_dict, passage_to_city, prelabel_relevance = dataset.load_data()

    query_embeddings, passage_embeddings = handle_embeddings(
        model_name, query_embeddings_path, passage_embeddings_path, queries, passages
    )

    ############## Config ##############
    cache_path = f"data/{dataset_name}/cache.csv"
    output_prefix = f"output/{dataset_name}/{model_name}"

    os.makedirs(f"{output_prefix}/evaluation_results", exist_ok=True)
    os.makedirs(f"{output_prefix}/retrieval_results", exist_ok=True)

    # LLM Setup
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    cache = dataset.load_cache()
    update_cache = f"data/{dataset_name}/cache.csv"

    # Centralized configuration
    configs = {
        "kernel": kernel,
        "llm_budget": llm_budget,
        "sample_strategy": sample_strategy,
        "epsilon": epsilon,
        "city_max_sample": city_max_sample,
        "top_k_passages": top_k_passages,
        "k_retrieval": len(passage_ids),
        "batch_size": batch_size,
        "fusion_method": fusion_method,
        "random_seed": random_seed,
    }

    # Automatically generate file names based on config values
    config_str = "_".join([f"{k}={v}" for k, v in configs.items()])
    
    evaluation_path = f"{output_prefix}/evaluation_results/{config_str}.csv"
    retrieval_results_path = f"{output_prefix}/retrieval_results/{config_str}.json"

    ############## Evaluation ##############
    retrieval_cities = defaultdict()
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    for i, (query, query_id) in tqdm(enumerate(zip(queries, question_ids)), desc="Processing Queries", total=len(queries)):
        if i % 10 == 0:
            print(f"Processing Query {i + 1}/{len(queries)}...")

        # Run GP Retrieval
        gp = gp_retrieval(
            query=query,
            query_embedding=query_embeddings[i],
            query_id=query_id,
            passage_ids=passage_ids.copy(),
            passage_embeddings=passage_embeddings,
            passages=passages,
            passage_dict=passage_dict,
            kernel=kernel,
            llm=llm,
            llm_budget=llm_budget,
            epsilon=epsilon,
            city_max_sample=city_max_sample,
            sample_strategy=sample_strategy,
            batch_size=batch_size,
            cache=cache,
            update_cache=update_cache,
            verbose=False,
            random_seed=random_seed,
            normalize_y=normalize_y,
            alpha=alpha,
            length_scale=length_scale
        )

        # Fusion of Passage Scores into City Scores
        bandit_cities = fusion_score_gp(
            gp=gp,
            passage_ids=passage_ids,
            passage_dict=passage_dict,
            passage_embeddings=passage_embeddings,
            top_k_passages=top_k_passages,
            k_retrieval=len(passage_ids),
            return_scores=False,
            fusion_method=fusion_method
        )
        retrieval_cities[query_id] = bandit_cities

        # Evaluate Retrieval Results at Multiple k Levels
        for k in range(10, 51, 10):
            prec_k, rec_k, map_k = eval_rec(
                bandit_cities, 
                list(relevance_map[query_id].keys()), 
                k,
                verbose=False
            )
            prec_k_dict[k].append(prec_k)
            rec_k_dict[k].append(rec_k)
            map_k_dict[k].append(map_k)

    ############## Print and Save Results ##############
    print("\n=== GP Retrieval Results ===")

    results = {}
    for k in prec_k_dict.keys():
        mean_prec_k = np.mean(prec_k_dict[k]).round(4)
        mean_rec_k = np.mean(rec_k_dict[k]).round(4)
        mean_map_k = np.mean(map_k_dict[k]).round(4)

        print(f"Precision@{k}: {mean_prec_k}")
        print(f"Recall@{k}: {mean_rec_k}")
        print(f"MAP@{k}: {mean_map_k}")

        # Store in result dict for saving
        results[f"precision@{k}"] = mean_prec_k
        results[f"recall@{k}"] = mean_rec_k
        results[f"map@{k}"] = mean_map_k

    if True:
        # Save retrieval results
        with open(retrieval_results_path, "w", encoding="utf-8") as f:
            json.dump(retrieval_cities, f, indent=4)

        # Save evaluation results
        if save_results(configs, results, evaluation_path):
            print(f"Results saved to {evaluation_path}")
        else:
            raise RuntimeError("Failed to save results.")

if __name__ == "__main__":
    main(dataset_name="travel_dest", llm_budget=10, sample_strategy="random", kernel="rbf", epsilon=0, top_k_passages=3, normalize_y=True, alpha=1e-1, length_scale=1.0)