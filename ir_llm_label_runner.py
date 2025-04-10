import argparse
import json
import os

from src.Dataset.dataloader import handle_dataset
from src.LLM.ChatGPT import ChatGPT

from src.Retrieval.retrieval import dense_retrieval, calculate_cosine_similarity
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import save_results

import numpy as np
from tqdm import tqdm
from collections import defaultdict
import wandb

def main(dataset_name, model_name, top_k):
    verbose = False
    ################### Load Data ###################

    base_path = os.path.dirname(os.path.abspath(__file__))

    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"
    cache_path = f"data/{dataset_name}/cache.csv"

    query_embeddings_path = os.path.join(base_path, query_embeddings_path)
    passage_embeddings_path = os.path.join(base_path, passage_embeddings_path)
    cache_path = os.path.join(base_path, cache_path)

    dataset = handle_dataset(dataset_name, cache_path)
    question_ids, queries, passage_ids, passages, relevance_map = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)
    cache = dataset.load_cache()

    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    rating_results = defaultdict(dict)
    hit, total = 0, 0

    for q_id, query, query_embedding in tqdm(zip(question_ids, queries, query_embeddings), desc="Labeling with LLM", total=len(queries)):
        sim_matrx = calculate_cosine_similarity(query_embedding, passage_embeddings)
        sorted_indices = np.argsort(sim_matrx)[::-1][:top_k]
        sorted_passages = [(passage_ids[i], passages[i]) for i in sorted_indices if passage_ids[i] not in cache[q_id]]
        total +=  top_k
        hit += (top_k - len(sorted_passages))

        for p_id, passage in tqdm(sorted_passages, total=len(sorted_passages), disable=not verbose):
            if p_id in cache[q_id]:
                continue
            scores = llm.get_score(query, [passage], query_id=q_id, passage_ids=[p_id], cache=cache, update_cache=dataset.cache_path)
            rating_results[q_id][p_id] = scores[0]

    print(f"Total: {total}, Hit: {hit}, Hit rate: {hit/total:.4f}")
    json.dump(rating_results, open(f"data/{dataset_name}/{model_name}_llm_label_results.json", "w"))

def arg_parser():
    parser = argparse.ArgumentParser(description='IR-based baseline')
    parser.add_argument('--dataset_name', type=str, default='covid', help='dataset name')
    parser.add_argument('--emb_model', type=str, default='all-MiniLM-L6-v2', help='embedding model')
    parser.add_argument("--top_k", type=int, default=100, help="top k passages to retrieve")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    main(dataset_name=args.dataset_name, model_name=args.emb_model, top_k=args.top_k)
