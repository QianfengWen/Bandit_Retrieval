import argparse
import json
import os

import wandb
from src.Baseline import llm_rerank


from tqdm import tqdm

from src.evaluate import evaluate
from src.utils import load_dataset, seed_everything

MODE="llm_reranking"

def main(dataset_name, model_name, top_k_passages, args):
    if not args.wandb_disable:
        configs = dict(vars(args))
        configs['runner'] = MODE
        run = wandb.init(
            project="bandit_v4",
            config=configs,
            group=args.wandb_group,
        )
    else:
        run = None

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings = (
        load_dataset(base_path, dataset_name, model_name, args.llm_name, args.prompt_type))

    print("\n")
    output = {}
    for q_id, query_embedding in tqdm(zip(query_ids, query_embeddings), desc=" > LLM Reranking", total=len(query_ids)):
        pred, _ = llm_rerank(
            query_id=q_id,
            query_embedding=query_embedding,
            passage_ids=passage_ids,
            passage_embeddings=passage_embeddings,
            top_k_passages=top_k_passages,
            score_type=args.score_type,
            cache=cache,
        )
        output[q_id] = pred

    cutoff = [int(k) for k in args.cutoff if int(k) <= top_k_passages]
    metric, results = evaluate(output, relevance_map, cutoff, threshold=dataset.relevance_threshold)

    if run is not None:
        updated_dict = {}
        for k, v in metric.items():
            new_key = str(k).replace("@", "/")
            updated_dict[new_key] = v
        wandb.log(updated_dict)

        os.makedirs(f"results/{dataset_name}/", exist_ok=True)
        with open(f"results/{dataset_name}/{run.name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)

def arg_parser():
    parser = argparse.ArgumentParser(description='Reranking with LLM')
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')
    parser.add_argument("--llm_name", type=str)
    parser.add_argument('--llm_budget', type=int, default=100, help='top k passages for reranking')
    parser.add_argument("--prompt_type", type=str, choices=['zeroshot', 'fewshot'])
    parser.add_argument("--score_type", type=str, choices=['er', 'pr'], default='er')

    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--cutoff", type=int, nargs="+", default=[1, 5, 10, 50, 100, 1000])

    parser.add_argument("--wandb_disable", action="store_true", help="disable wandb")
    parser.add_argument("--wandb_group", type=str, default="llm_rerank", help="wandb group")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    seed_everything()
    main(dataset_name=args.dataset_name, model_name=args.emb_model, top_k_passages=args.llm_budget, args=args)
