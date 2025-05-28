import numpy as np

from src.utils import cosine_similarity


def dense_retrieval(
        query_embedding,
        passage_ids: list[str],
        passage_embeddings,
        top_k_passages: int,
        return_score: bool=False,
    ):
    """
    Perform dense retrieval using cosine similarity.
    Args:
        query_embedding: The embedding of the query.
        passage_ids (list): List of passage IDs.
        passage_embeddings : Passage embeddings.
        top_k_passages (int): Number of passages to retrieve.
        return_score (bool): Whether to return the scores.
    """
    sim_matrix = cosine_similarity(query_embedding, passage_embeddings)
    top_k_idx = np.argsort(sim_matrix)[::-1][:top_k_passages]
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    if return_score:
        top_k_scores = [float(sim_matrix[idx]) for idx in top_k_idx]

        return top_k_ids, top_k_scores
    return top_k_ids, None


def llm_rerank(
        query_id: str,
        query_embedding,
        passage_ids: list[str],
        passage_embeddings,
        top_k_passages: int,
        score_type: str,
        return_score: bool=False, 
        cache: dict=None    
    ):
    """
    Rerank the top-k passages from dense retrieval using LLM scores.
    Args:
        passage_ids (list): List of passage IDs.
        query_id (str): The query ID.
        passage_embeddings : Passage embeddings.
        query_embedding : Query embedding.
        top_k_passages (int): Number of passages to retrieve.
        score_type (str): Type of score to use for reranking.
        return_score (bool): Whether to return the scores.
        cache (dict): Cache for storing LLM scores.

    """
    passage_ids, dense_score = dense_retrieval(
        query_embedding=query_embedding,
        passage_ids=passage_ids,
        passage_embeddings=passage_embeddings,
        top_k_passages=top_k_passages,
        return_score=True
    )
    assert len(passage_ids) == top_k_passages
    dense_score_dict = {pid: score for pid, score in zip(passage_ids, dense_score)}
    if cache:
        assert str(query_id) in cache, f"Query ID {query_id} not found in cache"
        valid_cached_items = {
            pid: llm_output[score_type] for pid, llm_output in cache[str(query_id)].items() if pid in passage_ids
        }
        assert len(valid_cached_items) == top_k_passages, f"Expected {top_k_passages} valid cached items, but got {len(valid_cached_items)}"
        sorted_item = sorted(valid_cached_items.items(), key=lambda x: (x[1], dense_score_dict[x[0]]), reverse=True)[:top_k_passages]
        sorted_passages = [key for key, _ in sorted_item]
        if return_score:
            sorted_scores = [value for _, value in sorted_item]
            return sorted_passages, sorted_scores
        return sorted_passages, None
    else:
        print("Please run llm_baseline_runner.py to generate LLM scores for the query")
        return None, None
    
    