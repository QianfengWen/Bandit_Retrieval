import argparse
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.Dataset.dataloader import handle_dataset
from src.Embedding.embedding import handle_embeddings
from src.Evaluation.evaluation import precision_k, recall_k, mean_average_precision_k, normalized_dcg_k
from src.LLM.ChatGPT import ChatGPT
from src.RecUtils.rec_utils import save_results
from src.Retrieval.retrieval import bandit_retrieval


def main(dataset_name, model_name, acq_func, beta, llm_budget, k_cold_start, kernel, batch_size, save_flag=True):
    ################### Load Data ###################

    dataset = handle_dataset(dataset_name)
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"

    query_ids, queries, passage_ids, passages, relevance_map = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path,
                                                             queries, passages)

    if args.debug:
        print("DEBUG MODE")
        query_ids = query_ids[:1]
        queries = queries[:1]
        print(f"Query IDs: {query_ids}")
        print(f"Queries: {queries}")
        llm_budget = 10
        k_cold_start = 5
        verbose = True
    else:
        verbose = False

    ################### Configuration ###################
    output_prefix = f"data/{dataset_name}/{model_name}_bandit"
    result_path = f"{output_prefix}_results.csv"
    k_retrieval = max(args.cutoff)
    cache = dataset.load_cache()
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    gpucb_percentage = (llm_budget - k_cold_start) / llm_budget

    if acq_func == "greedy":
        print(f"For greedy, set k_cold_start to {llm_budget}")
        k_cold_start = llm_budget
    configs = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "k_retrieval": k_retrieval,
        "llm_budget": llm_budget,
        "k_cold_start": k_cold_start,

        "beta": beta,
        "kernel": kernel,
        "acq_func": acq_func,
        "gpubc_percentage": gpucb_percentage,
        "batch_size": args.batch_size,
    }

    for k, v in configs.items():
        print(f"{k}: {v}")
    ################### Evaluation ###################
    ndcg_k_dict = defaultdict(list)
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    print("=== Bandit Runner ===")
    for i, (query, q_id) in tqdm(enumerate(zip(queries, query_ids)), desc="Query", total=len(queries)):
        # TODO: for reranking, modify passage ids
        items, scores, founds = bandit_retrieval(
            passage_ids=passage_ids.copy(),
            passage_embeddings=passage_embeddings,
            passages=passages,
            llm=llm,
            query=query,
            query_embedding=query_embeddings[i],
            query_id=q_id,
            beta=beta,
            acq_func=acq_func,
            kernel=kernel,
            llm_budget=llm_budget,
            k_cold_start=k_cold_start,
            k_retrieval=k_retrieval,
            batch_size=batch_size,
            verbose=verbose,
            return_score=True,
            cache=cache,
            update_cache=dataset.cache_path,
        )
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
    parser.add_argument('--acq_func', type=str, default='ucb', choices=['ucb', 'random', 'greedy'])
    parser.add_argument('--beta', type=float, default=2, help='beta for bandit')
    parser.add_argument('--kernel', type=str, default='rbf', help='kernel for bandit')
    parser.add_argument('--llm_budget', type=int, default=50, help='llm budget for bandit')
    parser.add_argument('--cold_start', type=int, default=25, help='cold start for bandit')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size for bandit')

    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--cutoff", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    main(dataset_name=args.dataset_name, model_name=args.emb_model, acq_func=args.acq_func, beta=args.beta,
         llm_budget=args.llm_budget, k_cold_start=args.cold_start, kernel=args.kernel, batch_size=args.batch_size)
