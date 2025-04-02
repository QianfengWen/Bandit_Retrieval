import argparse

from src.Dataset.dataloader import handle_dataset
from src.Evaluation.evaluation import precision_k, mean_average_precision_k, recall_k, normalized_dcg_k

from src.Retrieval.retrieval import dense_retrieval, llm_rerank
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import save_results

import numpy as np
from tqdm import tqdm
from collections import defaultdict

def main(dataset_name, model_name, top_k_passages, save_flag=True):
    ################### Load Data ###################

    dataset = handle_dataset(dataset_name)
    query_embeddings_path = f"src/data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"src/data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    question_ids, queries, passage_ids, passages, relevance_map = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)

    cache = dataset.load_cache()
    ################### Configuration ###################

    configs = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "top_k_passages": top_k_passages,
    }

    result_path = f"ir_{dataset_name}_{model_name}_llm_result_100.csv"
    ################### Evaluation ###################
    k_retrieval = top_k_passages
    ndcg_k_dict = defaultdict(list)
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    print("=== LLM Reranking ===")
    for q_id, query_embedding in tqdm(zip(question_ids, query_embeddings), desc="Reranking", total=len(question_ids)):
        items = llm_rerank(passage_ids, passage_embeddings, query_embedding, q_id, k_retrieval=k_retrieval, cache=cache, return_score=False)
        gt = set([p_id for p_id, relevance in relevance_map[q_id].items() if relevance >= dataset.relevance_threshold])

        for k_start in args.cutoff:
            prec_k = precision_k(items, gt, k_start)
            rec_k = recall_k(items, gt, k_start)
            map_k = mean_average_precision_k(items, gt, k_start)
            ndcg_k = normalized_dcg_k(items, gt, k_start)

            prec_k_dict[k_start].append(prec_k)
            rec_k_dict[k_start].append(rec_k)
            map_k_dict[k_start].append(map_k)
            ndcg_k_dict[k_start].append(ndcg_k)

    print(f"=== LLM reranking Demo for {model_name}===")

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


def arg_parser():
    parser = argparse.ArgumentParser(description='IR-based baseline')
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')
    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--cutoff", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    main(dataset_name=args.dataset_name, model_name=args.emb_model, top_k_passages=100)
