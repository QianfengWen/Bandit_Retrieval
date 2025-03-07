import numpy as np
from src.GPUCB.retrieval_gpucb import RetrievalGPUCB
from src.LLM.llm import LLM
from tqdm import tqdm


def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)


def bandit_retrieval_embeddings_based(passage_ids: list, passage_embeddings: list, passages: list, llm: LLM, query, query_embedding, query_id, beta=2.0, llm_budget: int=10, k_cold_start: int=5, k_retrieval: int=10, batch_size: int=10, relevance_map: dict=None) -> list:
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
    
    k_cold_start = min(k_cold_start, llm_budget)
    
    # Initialize GP-UCB for embeddings-based retrieval
    gpucb = RetrievalGPUCB(beta=beta)
    
    # Keep track of all available passage IDs and their embeddings
    available_ids = passage_ids.copy()
    id_to_embedding = {pid: emb for pid, emb in zip(passage_ids, passage_embeddings)}
    
    # Keep track of observed passage IDs and scores
    observed_ids = []
    scores = {}
    
    # Use query embedding to prioritize passages for cold start
    cosine_similairty_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
    if query_embedding is not None and k_cold_start > 0:
        cold_start_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_cold_start]
        cold_start_ids = [passage_ids[idx] for idx in cold_start_idx]
        cold_start_batches = [cold_start_ids[i:i + batch_size] for i in range(0, len(cold_start_ids), batch_size)]
        
        # Cold start: evaluate k_cold_start passages in batches
        for batch in cold_start_batches:
            # Remove batch items from available IDs
            for target_id in batch:
                available_ids.remove(target_id)
            
            # Get passages corresponding to batch
            passage_idxs = [passage_ids.index(target_id) for target_id in batch]
            batch_passages = [passages[idx] for idx in passage_idxs]

            # Get relevance scores for the batch using LLM
            batch_scores = llm.get_score(query, batch_passages)
            print("batch_scores: ", batch_scores)

            # create score and passage pairs, print them
            for score, passage in zip(batch_scores, batch_passages):
                try:
                    print(f"True: {relevance_map[query_id].get(target_id, 0)}, Score: {score}, Passage: {passage}")
                except UnicodeEncodeError:
                    # Fallback to ASCII encoding if Unicode fails
                    print(f"True: {relevance_map[query_id].get(target_id, 0)}, Score: {score}, Passage: {passage.encode('ascii', 'replace').decode()}")

            # Store observations
            for target_id, score in zip(batch, batch_scores):
                scores[target_id] = score
                observed_ids.append(target_id)

                # Update GP-UCB model using the passage embedding as feature
                gpucb.update(id_to_embedding[target_id], score)
    
    num_iterations = (llm_budget - k_cold_start) // batch_size  # Number of batch iterations

    # Exploration-exploitation: use GP-UCB to select passages in batches
    for _ in tqdm(range(num_iterations), desc="Bandit Retrieval"):
        if not available_ids:
            break  # No more passages to evaluate

        # Get embeddings for available passages
        available_embeddings = [id_to_embedding[pid] for pid in available_ids]

        # Select top `batch_size` passages using GP-UCB
        next_embedding_idxs = gpucb.select(available_embeddings, batch_size)  # Get batch indices
        next_ids = [available_ids[idx] for idx in next_embedding_idxs]  # Get passage IDs

        # Remove selected passages from available list
        for next_id in next_ids:
            available_ids.remove(next_id)

        # Get passages for batch processing
        passage_idxs = [passage_ids.index(next_id) for next_id in next_ids]
        batch_passages = [passages[idx] for idx in passage_idxs]

        # Get relevance scores using LLM in batch
        batch_scores = llm.get_score(query, batch_passages)

        # Store observations and update GP-UCB model
        for next_id, score in zip(next_ids, batch_scores):
            scores[next_id] = score
            observed_ids.append(next_id)
            gpucb.update(id_to_embedding[next_id], score)  # Update model

    # Return the top-k passages based on final GP predictions
    top_k_idx = gpucb.get_top_k(passage_embeddings, k_retrieval)
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    baseline_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_retrieval]
    baseline_ids = [passage_ids[idx] for idx in baseline_idx]

    return top_k_ids, baseline_ids



def rec_retrieval(passage_ids: list, passage_embeddings: list, passages: list, llm: LLM, query, query_embedding, query_id, beta=2.0, llm_budget: int=10, k_cold_start: int=5, k_retrieval: int=10, batch_size: int=10) -> list:
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
    
    k_cold_start = min(k_cold_start, llm_budget)
    
    # Initialize GP-UCB for embeddings-based retrieval
    gpucb = RetrievalGPUCB(beta=beta)
    
    # Keep track of all available passage IDs and their embeddings
    available_ids = passage_ids.copy()
    id_to_embedding = {pid: emb for pid, emb in zip(passage_ids, passage_embeddings)}
    
    # Keep track of observed passage IDs and scores
    observed_ids = []
    scores = {}
    
    # Use query embedding to prioritize passages for cold start
    cosine_similairty_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
    if query_embedding is not None and k_cold_start > 0:
        cold_start_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_cold_start]
        cold_start_ids = [passage_ids[idx] for idx in cold_start_idx]
        cold_start_batches = [cold_start_ids[i:i + batch_size] for i in range(0, len(cold_start_ids), batch_size)]
        
        # Cold start: evaluate k_cold_start passages in batches
        for batch in cold_start_batches:
            # Remove batch items from available IDs
            for target_id in batch:
                available_ids.remove(target_id)
            
            # Get passages corresponding to batch
            passage_idxs = [passage_ids.index(target_id) for target_id in batch]
            batch_passages = [passages[idx] for idx in passage_idxs]

            # Get relevance scores for the batch using LLM
            batch_scores = llm.get_score(query, batch_passages)
            print("batch_scores: ", batch_scores)

            # Store observations
            for target_id, score in zip(batch, batch_scores):
                scores[target_id] = score
                observed_ids.append(target_id)

                # Update GP-UCB model using the passage embedding as feature
                gpucb.update(id_to_embedding[target_id], score)
    
    num_iterations = (llm_budget - k_cold_start) // batch_size  # Number of batch iterations

    # Exploration-exploitation: use GP-UCB to select passages in batches
    for _ in tqdm(range(num_iterations), desc="Bandit Retrieval"):
        if not available_ids:
            break  # No more passages to evaluate

        # Get embeddings for available passages
        available_embeddings = [id_to_embedding[pid] for pid in available_ids]

        # Select top `batch_size` passages using GP-UCB
        next_embedding_idxs = gpucb.select(available_embeddings, batch_size)  # Get batch indices
        next_ids = [available_ids[idx] for idx in next_embedding_idxs]  # Get passage IDs

        # Remove selected passages from available list
        for next_id in next_ids:
            available_ids.remove(next_id)

        # Get passages for batch processing
        passage_idxs = [passage_ids.index(next_id) for next_id in next_ids]
        batch_passages = [passages[idx] for idx in passage_idxs]

        # Get relevance scores using LLM in batch
        batch_scores = llm.get_score(query, batch_passages)

        # Store observations and update GP-UCB model
        for next_id, score in zip(next_ids, batch_scores):
            scores[next_id] = score
            observed_ids.append(next_id)
            gpucb.update(id_to_embedding[next_id], score)  # Update model

    # Return the top-k passages based on final GP predictions
    top_k_idx, top_k_scores = gpucb.get_top_k(passage_embeddings, k_retrieval, return_scores=True)
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    return top_k_ids, top_k_scores



# def bandit_retrieval_indices_based(passage_ids: list, passage_embeddings: list, passages: list, llm, query, query_embedding, query_id, beta=2.0, llm_budget: int=10, k_cold_start: int=5, k_retrieval: int=10) -> list:
#     """
#     Bandit retrieval using GP-UCB, based on indices of passages.
    
#     Args:
#         passage_ids: List of passage IDs
#         passages: List of passage texts
#         llm: LLM interface for scoring passages
#         query: Query text
#         query_id: Optional query ID for ground truth lookups
#         beta: Exploration-exploitation trade-off parameter
#         llm_budget: Number of LLM calls to make
#         k_cold_start: Number of random samples to collect initially
#         k_retrieval: Number of passages to retrieve at the end
    
#     Returns:
#         List of passage IDs ranked by relevance
#     """
#     if llm_budget < 1:
#         raise ValueError("LLM budget must be at least 1")
    
#     # Ensure k_cold_start is not larger than the budget
#     k_cold_start = min(k_cold_start, llm_budget)
    
#     # Initialize GP-UCB for indices-based retrieval
#     gpucb = RetrievalGPUCB(beta=beta)
    
#     # Keep track of all available passage IDs
#     available_ids = passage_ids.copy()
#     id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}
    
#     # Keep track of observed passage IDs and scores
#     observed_ids = []
#     scores = {}

#     cosine_similairty_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
#     if query_embedding is not None and k_cold_start > 0:
#         cold_start_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_cold_start]
#         cold_start_ids = [passage_ids[idx] for idx in cold_start_idx]
    
    
#         for target_id in cold_start_ids:
#             # Remove from available IDs
#             available_ids.remove(target_id)
            
#             # Get relevance score using LLM
#             passage_idx = passage_ids.index(target_id)
#             score = llm.get_score(query, passages[passage_idx], query_id=query_id, passage_id=target_id)
            
#             # Store observation
#             scores[target_id] = score
#             observed_ids.append(target_id)
            
#             # Update GP-UCB model
#             gpucb.update(cosine_similairty_matrix[passage_idx], score)  # Use index in observed list as feature

    
#     # Exploration-exploitation: use GP-UCB to select the remaining passages
#     for i in tqdm(range(llm_budget - k_cold_start), desc=f"Bandit Retrieval"):
#         if not available_ids:
#             break  # No more passages to evaluate
        
#         # Convert available IDs to indices in the observed list (for use as features)
#         candidate_indices = [id_to_index[pid] for pid in available_ids]
#         candidate_similarities = [cosine_similairty_matrix[idx] for idx in candidate_indices]
        
#         # Select next passage to evaluate
#         next_idx = gpucb.select(candidate_similarities)
#         next_idx = next_idx[0]
#         next_id = available_ids[next_idx]
        
#         # Remove from available IDs
#         available_ids.remove(next_id)
        
#         # Get relevance score using LLM
#         passage_idx = passage_ids.index(next_id)
#         score = llm.get_score(query, passages[passage_idx], query_id=query_id, passage_id=next_id)

        
#         # Store observation
#         scores[next_id] = score
#         observed_ids.append(next_id)
        
#         # Update GP-UCB model with the index in the observed list as feature
#         gpucb.update(cosine_similairty_matrix[passage_idx], score)
    
#     # pdb.set_trace()
#     top_k_idx = gpucb.get_top_k(cosine_similairty_matrix, k_retrieval)
#     top_k_ids = [passage_ids[idx] for idx in top_k_idx]
    
#     baseline_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_retrieval]
#     baseline_ids = [passage_ids[idx] for idx in baseline_idx]
    
#     return top_k_ids, baseline_ids