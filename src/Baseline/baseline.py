import numpy as np

from src.utils import calculate_cosine_similarity


def dense_retrieval(
        passage_ids: list, 
        passage_embeddings: list, 
        query_embedding, 
        k_retrieval: int=1000, 
        return_score: bool=False,
    ):
    """
    baseline dense retrieval using cosine similarity
    """
    cosine_similairty_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
    top_k_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_retrieval]
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    if return_score:
        top_k_scores = [cosine_similairty_matrix[idx] for idx in top_k_idx]
        return top_k_ids, top_k_scores
    return top_k_ids, None


def llm_rerank(
        passage_ids: list[int], 
        passage_embeddings: list, 
        query_embedding, 
        query_id: int, 
        k_retrieval: int,
        return_score: bool=False, 
        cache: dict=None    
    ):
    """
    rerank using LLM
    """
    passage_ids, dense_score = dense_retrieval(passage_ids, passage_embeddings, query_embedding, k_retrieval=k_retrieval, return_score=True)
    dense_score_dict = {pid: score for pid, score in zip(passage_ids, dense_score)}
    if cache:
        valid_cached_items = {
            pid: score for pid, score in cache[str(query_id)].items() if str(pid) in passage_ids
        }
        sorted_item = sorted(valid_cached_items.items(), key=lambda x: (x[1], dense_score_dict[x[0]]), reverse=True)[:k_retrieval]
        sorted_passages = [key for key, _ in sorted_item]
        if return_score:
            sorted_scores = [value for _, value in sorted_item]
            return sorted_passages, sorted_scores
        return sorted_passages
    else:
        print(f"Cache miss for query {query_id}, using LLM ...")
        print("Please run llm_baseline_runner.py to generate LLM scores for the query")
    
    