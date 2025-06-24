import os
import random

import numpy as np
import sklearn
import torch
from scipy.special import softmax

from src.Dataset.factory import handle_dataset
from src.embedding import handle_embeddings


def cosine_similarity(query_embeddings, passage_embeddings):
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.reshape(1, -1)
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    passage_norm = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(query_norm, passage_norm.T)
    return similarity_matrix.flatten()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    sklearn.utils.check_random_state(seed)


    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for Huggingface transformers (if available)
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass


def load_dataset(base_path, dataset_name, model_name, llm_name=None, prompt_type=None):
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"
    cache_path = f"data/{dataset_name}/logit_{prompt_type}_{llm_name}_cache.csv" if llm_name else f"data/{dataset_name}/cache.csv"

    query_embeddings_path = os.path.join(base_path, query_embeddings_path)
    passage_embeddings_path = os.path.join(base_path, passage_embeddings_path)
    cache_path = os.path.join(base_path, cache_path)

    print("\n >> Loading dataset...")
    dataset = handle_dataset(dataset_name, cache_path)
    query_ids, queries, passage_ids, passages, relevance_map = dataset.load_data()
    print(f" >>> Query: {len(queries)}")
    print(f" >>> Passage: {len(passages)}")

    print("\n >> Loading embeddings...")
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path,
                                                             queries, passages)
    cache = dataset.load_cache() if llm_name else None
    return dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings

def get_cache(cache, query_id, passage_id):
    """
        Get the cache for a specific query and passage.
        Args:
            cache: The cache dictionary.
            query_id: The ID of the query.
            passage_id: The ID of the passage.
        Returns:
            The cached value if it exists, otherwise None.
    """
    if cache and query_id in cache and passage_id in cache[query_id]:
        return cache[query_id][passage_id]
    return None

def logit2entropy(logit: list[float]):
    """
        Convert logit to entropy.
        Args:
            logit: The logit output from the model.
        Returns:
            The entropy of the logit.
    """
    logit = np.array(logit, dtype=np.float64)

    exp_logits = np.exp(logit - np.max(logit))
    probs = exp_logits / np.sum(exp_logits)
    return -np.sum(probs * np.log(probs + 1e-12))


def logit2confidence(logit: list[float]):
    """
        Convert logit to confidence.
        Args:
            logit: The logit output from the model.
        Returns:
            The confidence of the logit.
    """
    probs = softmax(logit)
    return 1.0 - np.max(probs)
