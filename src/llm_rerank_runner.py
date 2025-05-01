from src.Dataset.dataloader import handle_dataset
from src.Retrieval.retrieval import llm_rerank, cross_encoder_rerank
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score, eval_rec, save_results
from sentence_transformers import CrossEncoder

import numpy as np
from tqdm import tqdm
from collections import defaultdict

def main(dataset_name, budget=50, top_k_passages=3, fusion_mode="sum", cross_encoder_reranking=False):
    ################### Load Data ###################
    dataset_name = dataset_name
    model_name = "all-MiniLM-L6-v2"
    dataset = handle_dataset(dataset_name)
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

     
    question_ids, queries, passage_ids, passages, relevance_map, passage_dict, passage_city_map, prelabel_relevance = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)

    ################### Configuration ###################
    verbose=False
    k_start = 10
    k_eval = 50
    budget = budget
    top_k_passages = top_k_passages
    fusion_mode = fusion_mode
    save_flag = True

    configs = {
        "budget": budget,
        "top_k_passages": top_k_passages,
        "fusion_mode": fusion_mode
    }
    if cross_encoder_reranking:
        ############## Cross-Encoder #############
        cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        cross_encoder_model = CrossEncoder(cross_encoder_model_name)
        result_path = f"{dataset_name}_{model_name}_cross_encoder_reranking_results.csv"
    else:
        result_path = f"{dataset_name}_{model_name}_llm_reranking_results.csv"
    
    ################### Evaluation ###################
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)
    ndcg_k_dict = defaultdict(list)

    for q_id, query, query_embedding in tqdm(zip(question_ids, queries, query_embeddings), desc="Query", total=len(question_ids)):
        if cross_encoder_reranking:
            item, score = cross_encoder_rerank(passage_ids, passage_embeddings, passages, query_embedding, q_id, k_retrieval= budget, return_score=True, cross_encoder_model=cross_encoder_model, query_text=query)
        else:
            item, score = llm_rerank(passage_ids, passage_embeddings, query_embedding, q_id, k_retrieval= budget, cache=prelabel_relevance, return_score=True)
        k_start = 10
        bandit_cities = fusion_score(item, score, passage_city_map, top_k_passages=top_k_passages, return_scores=False, fusion_mode=fusion_mode)
        while k_start <= k_eval:
            prec_k, rec_k, map_k, ndcg_k_score = eval_rec(bandit_cities, list(relevance_map[q_id].keys()), k_start, verbose=verbose)
            prec_k_dict[k_start].append(prec_k)
            rec_k_dict[k_start].append(rec_k)
            map_k_dict[k_start].append(map_k)
            ndcg_k_dict[k_start].append(ndcg_k_score)
            if verbose:
                print(f"Precision@{k_start}: {prec_k}\n")
                print(f"Recall@{k_start}: {rec_k}\n")
                print(f"MAP@{k_start}: {map_k}")
                print(f"NDCG@{k_start}: {ndcg_k_score}")
            k_start += 10
            
        if verbose:
            print("\n\n\n\n\n")

    print("=== LLM Reranking Demo ===")
    print(f"LLM Budget: {budget}")
    print(f"Top K Passages: {top_k_passages}")
    
    results = {}
    for k in prec_k_dict.keys():
        print(f"Precision@{k}: {np.mean(prec_k_dict[k])}\n")
        print(f"Recall@{k}: {np.mean(rec_k_dict[k])}\n")
        print(f"MAP@{k}: {np.mean(map_k_dict[k])}\n")
        print(f"NDCG@{k}: {np.mean(ndcg_k_dict[k])}\n")
        results[f"precision@{k}"] = np.mean(prec_k_dict[k]).round(4)
        results[f"recall@{k}"] = np.mean(rec_k_dict[k]).round(4)
        results[f"map@{k}"] = np.mean(map_k_dict[k]).round(4)
        results[f"ndcg@{k}"] = np.mean(ndcg_k_dict[k]).round(4)

    if save_flag:
        assert save_results(configs, results, result_path) == True, "Results not saved"
        print(f"Results saved to {result_path}")

if __name__ == "__main__":
    for name in ["travel_dest"]:
        for budget in [1, 5]:
            main(dataset_name=name, budget=budget, top_k_passages=3, fusion_mode="sum", cross_encoder_reranking=True)
