import numpy as np
from sentence_transformers import CrossEncoder

from src.utils import cosine_similarity


def dense_retrieval(
        query_embedding,
        passage_ids: list[str],
        passage_embeddings,
        cutoff: int,
        return_score: bool=False,
    ):
    """
    Perform dense retrieval using cosine similarity.
    Args:
        query_embedding: The embedding of the query.
        passage_ids (list): List of passage IDs.
        passage_embeddings : Passage embeddings.
        cutoff (int): Number of passages to retrieve.
        return_score (bool): Whether to return the scores.
    """
    sim_matrix = cosine_similarity(query_embedding, passage_embeddings)
    top_k_idx = np.argsort(sim_matrix)[::-1][:cutoff]
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    if return_score:
        top_k_scores = [float(sim_matrix[idx]) for idx in top_k_idx]

        return top_k_ids, top_k_scores
    return top_k_ids, None

def cross_encoder(
        query: str,
        query_embedding,
        passages: list[str],
        passage_ids: list[str],
        passage_embeddings,
        cutoff: int,
        return_score: bool=False,
    ):
    pid2p = {pid: p for pid, p in zip(passage_ids, passages)}
    passage_ids, dense_score = dense_retrieval(
        query_embedding=query_embedding,
        passage_ids=passage_ids,
        passage_embeddings=passage_embeddings,
        cutoff=cutoff,
        return_score=True
    )

    assert len(passage_ids) == cutoff
    dense_score_dict = {pid: score for pid, score in zip(passage_ids, dense_score)}

    qp_pair = [[query, pid2p[pid]] for pid in passage_ids]
    ce_score = []

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    for i in range(0, len(qp_pair), 32):
        batch = qp_pair[i:i + 32]
        ce_score.extend(model.predict(batch, convert_to_numpy=True).tolist())

    sorted_item = sorted(
        zip(passage_ids, ce_score),
        key=lambda x: (x[1], dense_score_dict[x[0]]),
        reverse=True
    )[:cutoff]
    sorted_passages = [key for key, _ in sorted_item]
    assert set(sorted_passages) == set(passage_ids), "Sorted passages do not match the original passage IDs"
    if return_score:
        sorted_scores = [value for _, value in sorted_item]
        return sorted_passages, sorted_scores
    return sorted_passages, None



def llm_rerank(
        query_id: str,
        query_embedding,
        passage_ids: list[str],
        passage_embeddings,
        cutoff: int,
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
        cutoff (int): Number of passages to retrieve.
        score_type (str): Type of score to use for reranking.
        return_score (bool): Whether to return the scores.
        cache (dict): Cache for storing LLM scores.

    """
    passage_ids, dense_score = dense_retrieval(
        query_embedding=query_embedding,
        passage_ids=passage_ids,
        passage_embeddings=passage_embeddings,
        cutoff=cutoff,
        return_score=True
    )

    assert len(passage_ids) == cutoff
    dense_score_dict = {pid: score for pid, score in zip(passage_ids, dense_score)}
    if cache:
        assert str(query_id) in cache, f"Query ID {query_id} not found in cache"
        valid_cached_items = {
            pid: llm_output[score_type] for pid, llm_output in cache[str(query_id)].items() if pid in passage_ids
        }
        assert len(valid_cached_items) == cutoff, f"Expected {cutoff} valid cached items, but got {len(valid_cached_items)}"
        sorted_item = sorted(valid_cached_items.items(), key=lambda x: (x[1], dense_score_dict[x[0]]), reverse=True)[:cutoff]
        sorted_passages = [key for key, _ in sorted_item]
        assert set(sorted_passages) == set(passage_ids), "Sorted passages do not match the original passage IDs"
        if return_score:
            sorted_scores = [value for _, value in sorted_item]
            return sorted_passages, sorted_scores
        return sorted_passages, None
    else:
        raise ValueError("Cache is not provided. Please run llm_label_runner.py to generate LLM scores for the query.")

