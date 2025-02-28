import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from ..GPUCB.retrieval_gpucb import RetrievalGPUCB
from tqdm import tqdm

def retrieve_k(query_embeddings: np.ndarray, distance_type: str, passage_embeddings: np.ndarray, k=100) -> np.ndarray:
    """
    Retrieve top-k passages for each query using L2 distance and return the indices of the top-k passages.

    We assume that the embeddings are already normalized.

    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param passage_embeddings: numpy array of shape (num_passages, embedding_dim)
    :param k: Number of passages to retrieve for each query
    :return: numpy array of shape (num_queries, k) containing the indices of the top-k passages for each query
    """
    print(f"Retrieving top-{k} passages using {distance_type} distance")
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("Embedding dimensions of queries and passages do not match")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for {distance_type} distance retrieval")
    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    if distance_type == "euclidean":
        l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
        _, indices = torch.topk(l2_distance_matrix, k, dim=1, largest=False, sorted=True)
        
    return indices.detach().cpu().numpy()


def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)

def bandit_retrieval_indices_based(passage_ids: list, passages: list, llm, query, query_id=None, beta=2.0, llm_budget: int=10, k_cold_start: int=5, k_retrieval: int=10) -> list:
    """
    Bandit retrieval using GP-UCB, based on indices of passages.
    
    Args:
        passage_ids: List of passage IDs
        passages: List of passage texts
        llm: LLM interface for scoring passages
        query: Query text
        query_id: Optional query ID for ground truth lookups
        beta: Exploration-exploitation trade-off parameter
        llm_budget: Number of LLM calls to make
        k_cold_start: Number of random samples to collect initially
        k_retrieval: Number of passages to retrieve at the end
    
    Returns:
        List of passage IDs ranked by relevance
    """
    if llm_budget < 1:
        raise ValueError("LLM budget must be at least 1")
    
    # Ensure k_cold_start is not larger than the budget
    k_cold_start = min(k_cold_start, llm_budget)
    
    # Initialize GP-UCB for indices-based retrieval
    gpucb = RetrievalGPUCB(beta=beta, is_embeddings_based=False)
    
    # Keep track of all available passage IDs
    available_ids = passage_ids.copy()
    
    # Keep track of observed passage IDs and scores
    observed_ids = []
    scores = {}
    
    # Cold start: evaluate k_cold_start passages randomly
    cold_start_ids = np.random.choice(available_ids, min(k_cold_start, len(available_ids)), replace=False).tolist()
    
    for target_id in cold_start_ids:
        # Remove from available IDs
        available_ids.remove(target_id)
        
        # Get relevance score using LLM
        passage_idx = passage_ids.index(target_id)
        score = llm.get_score(query, passages[passage_idx], query_id=query_id, passage_id=target_id)
        
        # Store observation
        scores[target_id] = score
        observed_ids.append(target_id)
        
        # Update GP-UCB model
        gpucb.update(observed_ids.index(target_id), score)  # Use index in observed list as feature
    
    # Exploration-exploitation: use GP-UCB to select the remaining passages
    for i in range(llm_budget - k_cold_start):
        if not available_ids:
            break  # No more passages to evaluate
        
        # Convert available IDs to indices in the observed list (for use as features)
        available_indices = list(range(len(observed_ids) + len(available_ids)))
        current_indices = list(range(len(observed_ids)))
        candidate_indices = [idx for idx in available_indices if idx not in current_indices]
        
        # Select next passage to evaluate
        next_idx = gpucb.select(candidate_indices)
        
        # Map the index back to a passage ID
        if next_idx < len(observed_ids):
            # This shouldn't happen, but handle it just in case
            next_id = observed_ids[next_idx]
        else:
            # This is a new index, map it to an available passage ID
            available_idx = next_idx - len(observed_ids)
            next_id = available_ids[min(available_idx, len(available_ids)-1)]
        
        # Remove from available IDs
        if next_id in available_ids:
            available_ids.remove(next_id)
        
        # Get relevance score using LLM
        passage_idx = passage_ids.index(next_id)
        score = llm.get_score(query, passages[passage_idx], query_id=query_id, passage_id=next_id)
        
        # Store observation
        scores[next_id] = score
        observed_ids.append(next_id)
        
        # Update GP-UCB model with the index in the observed list as feature
        gpucb.update(observed_ids.index(next_id), score)
    
    # Return the top-k passages based on final GP predictions
    all_ids = observed_ids + available_ids
    
    # Get mean predictions for all passage IDs
    features = list(range(len(all_ids)))
    mean_predictions, _ = gpucb.get_mean_std(features)
    
    # Create a mapping from index to passage ID
    idx_to_id = {idx: pid for idx, pid in enumerate(all_ids)}
    
    # Sort passages by predicted relevance
    sorted_indices = np.argsort(-mean_predictions)
    sorted_ids = [idx_to_id[idx] for idx in sorted_indices]
    
    # Return top-k passages
    return sorted_ids[:k_retrieval]


def bandit_retrieval_embeddings_based(passage_ids: list, passage_embeddings: list, passages: list, llm, query, query_embedding=None, query_id=None, beta=2.0, llm_budget: int=10, k_cold_start: int=5, k_retrieval: int=10) -> list:
    """
    Bandit retrieval using GP-UCB, based on embeddings of passages.
    
    Args:
        passage_ids: List of passage IDs
        passage_embeddings: List of passage embeddings
        passages: List of passage texts
        llm: LLM interface for scoring passages
        query: Query text
        query_embedding: Optional query embedding
        query_id: Optional query ID for ground truth lookups
        beta: Exploration-exploitation trade-off parameter
        llm_budget: Number of LLM calls to make
        k_cold_start: Number of random samples to collect initially
        k_retrieval: Number of passages to retrieve at the end
    
    Returns:
        List of passage IDs ranked by relevance
    """
    if llm_budget < 1:
        raise ValueError("LLM budget must be at least 1")
    
    # Ensure k_cold_start is not larger than the budget
    k_cold_start = min(k_cold_start, llm_budget)
    
    # Initialize GP-UCB for embeddings-based retrieval
    gpucb = RetrievalGPUCB(beta=beta, is_embeddings_based=True)
    
    # Keep track of all available passage IDs and their embeddings
    available_ids = passage_ids.copy()
    id_to_embedding = {pid: emb for pid, emb in zip(passage_ids, passage_embeddings)}
    
    # Keep track of observed passage IDs and scores
    observed_ids = []
    scores = {}
    
    # Use query embedding to prioritize passages for cold start
    if query_embedding is not None and k_cold_start > 0:
        cosine_similairty_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
        cold_start_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_cold_start]
        cold_start_ids = [passage_ids[idx] for idx in cold_start_idx]
        
    else:
        # Random cold start if no query embedding or k_cold_start=0
        cold_start_ids = np.random.choice(available_ids, min(k_cold_start, len(available_ids)), replace=False).tolist()
    
    # Cold start: evaluate k_cold_start passages
    for target_id in tqdm(cold_start_ids, desc=f"Cold Start for query {query_id}"):
        # Remove from available IDs
        available_ids.remove(target_id)
        
        # Get relevance score using LLM
        passage_idx = passage_ids.index(target_id)
        score = llm.get_score(query, passages[passage_idx], query_id=query_id, passage_id=target_id)
        
        # Store observation
        scores[target_id] = score
        observed_ids.append(target_id)
        
        # Update GP-UCB model using the passage embedding as feature
        gpucb.update(id_to_embedding[target_id], score)
    
    # Exploration-exploitation: use GP-UCB to select the remaining passages
    for _ in tqdm(range(llm_budget - k_cold_start), desc=f"Bandit Retrieval for query {query_id}"):
        if not available_ids:
            break  # No more passages to evaluate
        
        # Get embeddings for available passages
        available_embeddings = [id_to_embedding[pid] for pid in available_ids]
        
        # Select next passage to evaluate
        next_embedding_idx = gpucb.select(available_embeddings)

        assert isinstance(next_embedding_idx, (int, np.integer)), "Invalid type for next_embedding_idx"
        
        next_id = available_ids[next_embedding_idx]
        
        # Remove from available IDs
        available_ids.remove(next_id)
        
        # Get relevance score using LLM
        passage_idx = passage_ids.index(next_id)
        score = llm.get_score(query, passages[passage_idx], query_id=query_id, passage_id=next_id)
        
        # Store observation
        scores[next_id] = score
        observed_ids.append(next_id)
        
        # Update GP-UCB model
        gpucb.update(id_to_embedding[next_id], score)
    
    # Return the top-k passages based on final GP predictions
    top_k_idx = gpucb.select(passage_embeddings, k_retrieval)
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]
    return top_k_ids
    
    # all_ids = observed_ids + available_ids
    
    # # Get mean predictions for all passage embeddings
    
    
    # all_embeddings = [id_to_embedding[pid] for pid in all_ids]
    # mean_predictions, _ = gpucb.get_mean_std(all_embeddings)
    
    # # Create a mapping from index to passage ID
    # idx_to_id = {idx: pid for idx, pid in enumerate(all_ids)}
    
    # # Sort passages by predicted relevance
    # sorted_indices = np.argsort(-mean_predictions)
    # sorted_ids = [idx_to_id[idx] for idx in sorted_indices]
    
    # # Return top-k passages
    # return sorted_ids[:k_retrieval]