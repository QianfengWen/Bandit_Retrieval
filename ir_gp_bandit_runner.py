import argparse
import json
import os
from tqdm import tqdm
import wandb
from src.GPBandit.run_bandit import gp_bandit_retrieval_optimized
from src.LLM.factory import handle_llm
from src.evaluate import evaluate
from src.utils import load_dataset, seed_everything

MODE="bandit"

def main(dataset_name, model_name, acq_func, beta, llm_budget, k_cold_start, kernel, args):
    if not args.wandb_disable:
        configs = dict(vars(args))
        configs['runner'] = MODE
        run = wandb.init(
            project="bandit_wsdm",
            config=configs,
            group=args.wandb_group,
            tags=["gpytorch", str(args.seed), "batch"]
        )
    else:
        run = None

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings = (
        load_dataset(base_path, dataset_name, model_name, args.llm_name, args.prompt_type))

    if args.debug:
        print("\n > DEBUG MODE")
        query_ids = query_ids[6]
        queries = queries[6]
        print(f" >> Query IDs: {query_ids}")
        print(f" >> Queries: {queries}")

        llm_budget = 10
        k_cold_start = 5

    k_retrieval = max(args.cutoff)
    if args.offline:
        print("\n > OFFLINE MODE")
        llm = None
    else:
        llm = handle_llm(args.llm_name,args.prompt_type,args.score_type)

    print("\n")
    results = {}
    for query, q_id, q_emb in tqdm(zip(queries, query_ids, query_embeddings), desc=" > Bandit Ranking", total=len(queries)):
        preds, scores, founds = gp_bandit_retrieval_optimized(
            query=query,
            query_id=q_id,
            query_embedding=q_emb,
            passages=passages,
            passage_ids=passage_ids.copy(),
            passage_embeddings=passage_embeddings,

            llm=llm,
            llm_budget=llm_budget,
            k_cold_start=k_cold_start,
            score_type=args.score_type,

            kernel=kernel,
            acq_func=acq_func,
            alpha=args.alpha,
            alpha_method=args.alpha_method,
            train_alpha=args.train_alpha,
            length_scale=args.length_scale,
            beta=beta,
            # nu=args.nu,
            # xi=args.xi,

            use_query=args.use_query,
            # normalize_passage=args.normalize,
            ard=args.ard,

            k_retrieval=k_retrieval,
            batch_size=1,

            offline=args.offline,
            return_score=True,
            cache=cache,
            update_cache=dataset.cache_path,
            verbose=args.verbose,
        )
        results[q_id] = {
            "pred": preds,
            "score": scores,
            "found": founds,
        }
    metric, results = evaluate(results, relevance_map, args.cutoff, threshold=dataset.relevance_threshold)

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
    parser = argparse.ArgumentParser(description='IR-based baseline')
    parser.add_argument("--seed", type=int, default=42, help="seed for random number generator")
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')

    parser.add_argument("--llm_name", type=str, default='unsloth/Qwen3-14B-unsloth-bnb-4bit')
    parser.add_argument('--llm_budget', type=int, default=50, help='llm budget for bandit')
    parser.add_argument("--prompt_type", type=str, default="zeroshot")
    parser.add_argument("--score_type", type=str, choices=['er', 'pr'], default='er')

    parser.add_argument('--cold_start', type=int, default=25, help='cold start for bandit')
    parser.add_argument('--acq_func', type=str, default='ucb')
    parser.add_argument('--kernel', type=str, default='rbf', help='kernel for bandit')
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--alpha_method", type=str, default=None)
    parser.add_argument("--train_alpha", action="store_true", help="train alpha parameter")
    parser.add_argument("--length_scale", type=float, default=1)
    parser.add_argument('--beta', type=float, default=2, help='beta for bandit')
    parser.add_argument("--nu", type=float, default=2.5, help='nu for Matern Kernel')
    parser.add_argument("--xi", type=float, default=0.05, help='xi for EI/PI')

    parser.add_argument("--use_query", type=int, default=3, help="relevance of query")
    parser.add_argument("--normalize", action="store_true", help="normalize the passage embeddings")
    parser.add_argument("--ard", action="store_true", help="use ARD")

    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--cutoff", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    parser.add_argument("--binary_relevance", type=int, default=1)

    parser.add_argument("--wandb_disable", action="store_true", help="disable wandb")
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb group")

    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    parser.add_argument("--offline", action="store_true", help="offline mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    seed_everything(args.seed)
    main(dataset_name=args.dataset_name, model_name=args.emb_model, acq_func=args.acq_func, beta=args.beta,
         llm_budget=args.llm_budget, k_cold_start=args.cold_start, kernel=args.kernel, args=args)
