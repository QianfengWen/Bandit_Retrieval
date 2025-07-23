import argparse
import json
import os

import wandb
from src.Baseline import  rankgpt

from tqdm import tqdm

from src.evaluate import evaluate
from src.utils import load_dataset, seed_everything


def main(dataset_name, model_name, top_k_passages, args):
    if not args.wandb_disable:
        configs = dict(vars(args))
        run = wandb.init(
            project="bandit_rankgpt",
            config=configs,
            group=args.wandb_group,
        )
    else:
        run = None

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings = (
        load_dataset(base_path, dataset_name, model_name))

    if args.llm_name == "unsloth/Qwen3-14B-unsloth-bnb-4bit":
        from src.Baseline.RankGPT.llm import RankQwen
        llm = RankQwen()
    elif args.llm_name == "openai/gpt4o":
        raise NotImplementedError("This function should be implemented")
    else:
        raise ValueError(f"Unsupported model name: {args.llm_name}")


    print("\n")
    results = {}
    for query, q_id, query_embedding in tqdm(zip(queries, query_ids, query_embeddings), desc=" > RankGPT Reranking", total=len(query_ids)):
        pred = rankgpt(
            query=query,
            query_embedding=query_embedding,
            passages=passages,
            passage_ids=passage_ids,
            passage_embeddings=passage_embeddings,
            cutoff=top_k_passages,
            llm=llm,
            window_size=args.window_size,
            step=args.step,
            verbose=args.verbose,
        )
        results[q_id] = {"pred": pred}

    cutoff = [int(k) for k in args.cutoff if int(k) <= top_k_passages]
    metric, results = evaluate(results, relevance_map, cutoff, threshold=dataset.relevance_threshold)

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
    parser = argparse.ArgumentParser(description='Reranking with RankGPT')
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')
    parser.add_argument('--llm_name', type=str, default='unsloth/Qwen3-14B-unsloth-bnb-4bit', help='LLM model name')
    parser.add_argument('--llm_budget', type=int, default=50, help='top k passages for reranking')

    parser.add_argument('--window_size', type=int, default=20, help='window size for RankGPT')
    parser.add_argument('--step', type=int, default=10, help='step size for sliding window')

    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--cutoff", type=int, nargs="+", default=[1, 5, 10, 30, 50, 100])
    parser.add_argument("--binary_relevance", type=int, default=1)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb_disable", action="store_true", help="disable wandb")
    parser.add_argument("--wandb_group", type=str, default="llm_rerank", help="wandb group")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    seed_everything()
    main(dataset_name=args.dataset_name, model_name=args.emb_model, top_k_passages=args.llm_budget, args=args)
