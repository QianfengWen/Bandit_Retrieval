import argparse
import os

import numpy as np
from tqdm import tqdm
import wandb
from src.LLM.factory import handle_llm

from src.utils import load_dataset, cosine_similarity, seed_everything


def main(dataset_name, model_name, top_k, args):
    verbose = False
    config = dict(vars(args))
    config['runner'] = 'label'
    wandb.init(
        project='bandit_label',
        config=config,
        name=f"{dataset_name}_{args.prompt_type}_{args.part}/{args.total}",
    )

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings =(
        load_dataset(base_path, dataset_name, model_name, args.llm_name, args.prompt_type))


    if args.total is not None and args.part is not None:
        print(f"\n\n> {args.part}th part in total {args.total} part")
        start = len(queries) // args.total * (args.part-1)
        end = len(queries) // args.total * args.part
        print(f"> Starting in index {start} to {end}")

        queries = queries[start:end]
        query_ids = query_ids[start:end]
        query_embeddings = query_embeddings[start:end]

    llm = handle_llm(llm_name=args.llm_name, prompt_type=args.prompt_type, score_type=args.score_type)
    hit, total = 0, 0

    print("\n")
    for q_id, query, query_embedding in tqdm(zip(query_ids, queries, query_embeddings), desc=" > Labeling with LLM",
                                             total=len(queries)):
        sim_matrx = cosine_similarity(query_embedding, passage_embeddings)
        sorted_indices = np.argsort(sim_matrx)[::-1][:top_k]
        sorted_passages = [(passage_ids[i], passages[i]) for i in sorted_indices if passage_ids[i] not in cache[q_id]]
        total += top_k
        hit += (top_k - len(sorted_passages))

        for i in tqdm(range(0, len(sorted_passages), args.batch_size), disable=not verbose):
            batch_query_ids = [q_id] * args.batch_size
            batch_queries = [query] * args.batch_size
            batch = sorted_passages[i:i + args.batch_size]
            batch_passage_ids = [p_id for p_id, _ in batch]
            batch_passages = [passage for _, passage in batch]
            # Call the batch scoring
            llm.get_score(
                queries=batch_queries,
                passages=batch_passages,
                query_ids=batch_query_ids,
                passage_ids=batch_passage_ids,
                cache=cache,
                update_cache=dataset.cache_path,
                score_type=args.score_type,
            )

    print(f"> Total: {total}, Hit: {hit}, Hit rate: {hit / total:.4f}")


def arg_parser():
    parser = argparse.ArgumentParser(description='Labeling with LLM')
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')
    parser.add_argument("--part", type=int)
    parser.add_argument("--total", type=int)

    parser.add_argument("--llm_name", type=str, help="LLM name, default is ChatGPT")
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--score_type", type=str, choices=['er', 'pr'], default='er')

    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--top_k", type=int, default=100, help="top k passages to retrieve")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    seed_everything()
    main(dataset_name=args.dataset_name, model_name=args.emb_model, top_k=args.top_k, args=args)
