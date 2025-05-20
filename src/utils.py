import os

import numpy as np
from src.Dataset.covid import Covid
from src.Dataset.fiqa import FiQA
from src.Dataset.scidocs import Scidocs
from src.Dataset.scifact import Scifact
from src.Dataset.touche import Touche
from src.Dataset.nfcorpus import NfCorpus
from src.Dataset.dl19 import DL19
from src.Dataset.dl20 import DL20
from src.LLM.chatgpt import ChatGPT
from src.LLM.llama import Llama
from src.embedding import handle_embeddings


def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)


def handle_llm(llm_name, prompt_type=None):
    if llm_name is None:
        raise NotImplementedError ("No LLM name provided")
        llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    elif "chatgpt" in llm_name.lower():
        raise NotImplementedError ("ChatGPT is not supported in this version")
        llm = ChatGPT(api_key=os.getenv("CHATGPT_API_KEY"))
    elif "llama" in llm_name.lower():
         llm = Llama(model_name=llm_name, prompt_type=prompt_type)
    else:
        raise Exception(f"Unknown llm name: {llm_name}")
    return llm


def handle_dataset(dataset_name, cache_path=None):
    if dataset_name == "covid":
        dataset = Covid(cache_path=cache_path)
    elif dataset_name == "touche":
        dataset = Touche(cache_path=cache_path)
    elif dataset_name == 'dl19':
        dataset = DL19(cache_path=cache_path)
    elif dataset_name == 'dl20':
        dataset = DL20(cache_path=cache_path)
    elif dataset_name == 'nfcorpus':
        dataset = NfCorpus(cache_path=cache_path)
    elif dataset_name == 'scidocs':
        dataset = Scidocs(cache_path=cache_path)
    elif dataset_name == 'scifact':
        dataset = Scifact(cache_path=cache_path)
    elif dataset_name == 'fiqa':
        dataset = FiQA(cache_path=cache_path)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return dataset


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
