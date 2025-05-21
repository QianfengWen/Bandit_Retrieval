import argparse
import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.GPUCB.run_gpucb import bandit_retrieval
from src.LLM.llm import handle_llm


import wandb
from src.metric import precision_k, recall_k, mean_average_precision_k, normalized_dcg_k
from src.utils import load_dataset

MODE="bandit"

def main(dataset_name, model_name, acq_func, beta, llm_budget, k_cold_start, kernel, batch_size, args, save_flag=True):
    base_path = os.path.dirname(os.path.abspath(__file__))

    k_retrieval = max(args.cutoff)
    llm = handle_llm(args.llm_name)

    if acq_func == "greedy":
        print(f"For greedy, set k_cold_start to {llm_budget}")
        k_cold_start = llm_budget

    configs = dict(vars(args))
    configs['runner'] = MODE

    for k, v in configs.items():
        print(f"{k}: {v}")

    if not args.wandb_disable:
        run = wandb.init(
            project="bandit_v3",
            config=configs,
            group=args.wandb_group,
        )
    else:
        run = None

    ################### Load Data ###################
    dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings = (
        load_dataset(base_path, dataset_name, model_name, args.llm_name))

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


    ################### Evaluation ###################
    ndcg_k_dict = defaultdict(list)
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)

    total_pred = {}

    print("=== Bandit Runner ===")
    for i, (query, q_id) in tqdm(enumerate(zip(queries, query_ids)), desc="Query", total=len(queries)):
        items, scores, founds = bandit_retrieval(
            passage_ids=passage_ids.copy(),
            passage_embeddings=passage_embeddings,
            passages=passages,
            llm=llm,
            query=query,
            query_embedding=query_embeddings[i],
            query_id=q_id,
            use_query=args.use_query,
            alpha=args.alpha,
            beta=beta,
            length_scale=args.length_scale,
            nu=args.nu,
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
        total_pred[q_id] = {
            'gt': list(set),
            'pred': items
        }

        for k_start in args.cutoff:
            prec_k = precision_k(items, gt, k_start)
            rec_k = recall_k(items, gt, k_start)
            map_k = mean_average_precision_k(items, gt, k_start)
            ndcg_k = normalized_dcg_k(items, relevance_map[q_id], k_start)

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

        results[f"precision@{k}"] = np.mean(prec_k_dict[k])
        results[f"recall@{k}"] = np.mean(rec_k_dict[k])
        results[f"map@{k}"] = np.mean(map_k_dict[k])
        results[f"ndcg@{k}"] = np.mean(ndcg_k_dict[k])

    if run is not None:
        updated_dict = {}
        for k, v in results.items():
            new_key = str(k).replace("@", "/")
            updated_dict[new_key] = v
        wandb.log(updated_dict)

        os.makedirs(f"results/{dataset_name}/", exist_ok=True)
        with open(f"results/{dataset_name}/{run.name}.json", "w", encoding="utf-8") as f:
            json.dump(total_pred, f, indent=1, ensure_ascii=False)

def arg_parser():
    parser = argparse.ArgumentParser(description='IR-based baseline')
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')

    parser.add_argument("--llm_name", type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit')
    parser.add_argument('--llm_budget', type=int, default=50, help='llm budget for bandit')
    parser.add_argument('--cold_start', type=int, default=25, help='cold start for bandit')
    parser.add_argument("--use_query", type=int, default=None, help="relevance of query")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for bandit')

    parser.add_argument('--acq_func', type=str, default='ucb', choices=['ucb', 'random', 'greedy'])
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=2, help='beta for bandit')
    parser.add_argument("--length_scale", type=float, default=1)
    parser.add_argument("--nu", type=float, default=None, help='nu for Matern Kernel')
    parser.add_argument('--kernel', type=str, default='rbf', help='kernel for bandit')

    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--cutoff", type=int, nargs="+", default=[1, 10, 50, 100])

    parser.add_argument("--wandb_disable", action="store_true", help="disable wandb")
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb group")

    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    # TODO: add seed
    main(dataset_name=args.dataset_name, model_name=args.emb_model, acq_func=args.acq_func, beta=args.beta,
         llm_budget=args.llm_budget, k_cold_start=args.cold_start, kernel=args.kernel, batch_size=args.batch_size, args=args)
