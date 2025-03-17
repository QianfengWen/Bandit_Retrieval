import numpy as np
from src.GPUCB.retrieval_gpucb import RetrievalGPUCB
from src.LLM.llm import LLM
from tqdm import tqdm
import random

def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)

def bandit_retrieval(
        passage_ids: list, 
        passage_embeddings: list, 
        passages: list, 
        llm: LLM, 
        query, 
        query_embedding, 
        query_id, 
        beta=2.0, 
        llm_budget: int=10, 
        k_cold_start: int=5, 
        k_retrieval: int=1000,
        kernel: str="rbf", 
        batch_size: int=10, 
        verbose: bool=False, 
        return_score: bool=False, 
        cache: dict=None, 
        update_cache: str=None
    ):
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
    ############### Set up ################
    if llm_budget < 1:
        raise ValueError("LLM budget must be at least 1")
    
    k_cold_start = min(k_cold_start, llm_budget)
    
    gpucb = RetrievalGPUCB(beta=beta, kernel=kernel) # set up GP-UCB
    
    available_ids = passage_ids.copy()
    id_to_embedding = {pid: emb for pid, emb in zip(passage_ids, passage_embeddings)}
    
    observed_ids = []
    scores = {}
    
    ############### Cold start ################
    cosine_similairty_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
    if query_embedding is not None and k_cold_start > 0:
        cold_start_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_cold_start]
        cold_start_ids = [passage_ids[idx] for idx in cold_start_idx]

        random.shuffle(cold_start_ids)
        cold_start_batches = [cold_start_ids[i:i + batch_size] for i in range(0, len(cold_start_ids), batch_size)]
        
        for batch in cold_start_batches:
            # remove batch items from available IDs
            for target_id in batch:
                available_ids.remove(target_id)
            
            # get passages corresponding to batch
            passage_idxs = [passage_ids.index(target_id) for target_id in batch]
            batch_passages = [passages[idx] for idx in passage_idxs]

            # get relevance scores for the batch using LLM
            batch_scores = llm.get_score(query, batch_passages, query_id=query_id, passage_ids=batch, cache=cache, update_cache=update_cache)
            
            # store observations
            for target_id, score in zip(batch, batch_scores):
                scores[target_id] = score
                observed_ids.append(target_id)

                # update GP-UCB model using the passage embedding as feature
                gpucb.update(id_to_embedding[target_id], score)
            

            if verbose: # debug print
                print("batch_scores: ", batch_scores)

                for score, passage in zip(batch_scores, batch_passages):
                    try:
                        print(f"Score: {score}, Passage: {passage}")
                    except UnicodeEncodeError:
                        print(f"Score: {score}, Passage: {passage.encode('ascii', 'replace').decode()}")
    
    ############### Exploration-exploitation ################
    num_iterations = (llm_budget - k_cold_start) // batch_size 

    # use GP-UCB to select passages in batches
    for _ in tqdm(range(num_iterations), desc="Bandit Retrieval"):
        if not available_ids:
            break  

        available_embeddings = [id_to_embedding[pid] for pid in available_ids]

        next_embedding_idxs = gpucb.select(available_embeddings, batch_size) 
        next_ids = [available_ids[idx] for idx in next_embedding_idxs] 

        for next_id in next_ids:
            available_ids.remove(next_id)

        passage_idxs = [passage_ids.index(next_id) for next_id in next_ids]
        batch_passages = [passages[idx] for idx in passage_idxs]

        batch_scores = llm.get_score(query, batch_passages, query_id=query_id, passage_ids=next_ids, cache=cache, update_cache=update_cache)

        for next_id, score in zip(next_ids, batch_scores):
            scores[next_id] = score
            observed_ids.append(next_id)
            gpucb.update(id_to_embedding[next_id], score)  # Update model
        
        if verbose: # debug print
            print("batch_scores: ", batch_scores)

            for score, passage in zip(batch_scores, batch_passages):
                try:
                    print(f"Score: {score}, Passage: {passage}")
                except UnicodeEncodeError:
                    print(f"Score: {score}, Passage: {passage.encode('ascii', 'replace').decode()}")

    # return the top-k passages based on final GP predictions
    top_k_idx, top_k_scores = gpucb.get_top_k(passage_embeddings, k_retrieval, return_scores=return_score)
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    if return_score:
        return top_k_ids, top_k_scores

    return top_k_ids
    


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
    return top_k_ids


def llm_rerank(
        passage_ids: list[int], 
        passage_embeddings: list, 
        query_embedding, 
        query_id: int, 
        k_retrieval: int=1000,
        return_score: bool=False, 
        cache: dict=None    
    ):
    """
    rerank using LLM
    """
    passage_ids = dense_retrieval(passage_ids, passage_embeddings, query_embedding, k_retrieval=k_retrieval, return_score=False)
    if cache:
        try:
            valid_cached_items = {
                pid: score for pid, score in cache[query_id].items() if pid in passage_ids
            }
            
            sorted_item = sorted(valid_cached_items.items(), key=lambda x: x[1], reverse=True)[:k_retrieval]            
            sorted_passages = [int(key) for key, _ in sorted_item]
            if return_score:
                sorted_scores = [value for _, value in sorted_item]
                return sorted_passages, sorted_scores
            return sorted_passages
        
        except:
            pass
        
    print(f"Cache miss for query {query_id}, using LLM ...")
    print("Please run llm_baseline_runner.py to generate LLM scores for the query")
    
    