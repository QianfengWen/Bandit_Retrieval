import os
import random

import numpy as np
import torch

from src.Dataset.factory import handle_dataset
from src.embedding import handle_embeddings


def cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

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


def load_dataset(base_path, dataset_name, model_name, llm_name, prompt_type):
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
    cache = dataset.load_cache()
    return dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings


def logit2entropy(logit):
    """
    Convert logit to entropy.
    Args:
        logit: The logit output from the model.
    Returns:
        The entropy of the logit.
    """
    logit = torch.tensor(logit)
    prob = torch.softmax(logit, dim=-1)
    entropy = torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
    return entropy.cpu().numpy()
