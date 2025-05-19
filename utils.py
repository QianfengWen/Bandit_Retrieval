import os

from src.Dataset.dataloader import handle_dataset
from src.Embedding.embedding import handle_embeddings


def load_dataset(dataset_name, model_name, llm_name):
    print("Loading path...")
    base_path = os.path.dirname(os.path.abspath(__file__))
    query_embeddings_path = f"data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{model_name}_passage_embeddings.pkl"
    cache_path = f"data/{dataset_name}/{llm_name}_cache.csv" if llm_name else f"data/{dataset_name}/cache.csv"

    query_embeddings_path = os.path.join(base_path, query_embeddings_path)
    passage_embeddings_path = os.path.join(base_path, passage_embeddings_path)
    cache_path = os.path.join(base_path, cache_path)
    print(f"Query: {query_embeddings_path}")
    print(f"Passage: {passage_embeddings_path}")
    print(f"Cache: {cache_path}")

    print("Loading dataset...")
    dataset = handle_dataset(dataset_name, cache_path)
    query_ids, queries, passage_ids, passages, relevance_map = dataset.load_data()
    print(f"Query: {len(queries)}")
    print(f"Passage: {len(passages)}")
    print("Loading embeddings...")
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path,
                                                             queries, passages)
    cache = dataset.load_cache()
    return dataset, cache, relevance_map, queries, passages, query_ids, passage_ids, query_embeddings, passage_embeddings