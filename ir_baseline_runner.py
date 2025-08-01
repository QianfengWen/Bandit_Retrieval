import argparse
import json
import os

from tqdm import tqdm
import wandb

from src.Baseline import dense_retrieval
from src.evaluate import evaluate
from src.utils import load_dataset

MODE="dense_retrieval"

def main(dataset_name, model_name, args):
    if not args.wandb_disable:
        configs = dict(vars(args))
        configs['runner'] = MODE
        run = wandb.init(
            project="bandit_dense_retrieval",
            config=configs,
            group=args.wandb_group,
        )
    else:
        run = None

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings = (
        load_dataset(base_path, dataset_name, model_name))

    print("\n")
    results = {}
    for q_id, query_embedding in tqdm(zip(query_ids, query_embeddings), desc=" > Dense Retrieval", total=len(query_ids)):
        preds, scores = dense_retrieval(
            query_embedding=query_embedding,
            passage_ids=passage_ids,
            passage_embeddings=passage_embeddings,
            cutoff=max(args.cutoff),
            return_score=True
        )

        results[q_id] = {
            "pred": preds,
            "score": scores
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
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')
    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for embedding")
    parser.add_argument("--cutoff", type=int, nargs="+", default=[1, 5, 10, 20, 30, 50, 100])
    parser.add_argument("--binary_relevance", type=int, default=1)

    parser.add_argument("--wandb_disable", action="store_true", help="disable wandb")
    parser.add_argument("--wandb_group", type=str, default="dense_retrieval", help="wandb group")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    main(dataset_name=args.dataset_name, model_name=args.emb_model, args=args)
