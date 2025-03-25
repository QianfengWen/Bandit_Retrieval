import numpy as np
from src.GPUCB.gp import GaussianProcess
from src.GPUCB.retrieval_gpucb import RetrievalGPUCB
from src.LLM.llm import LLM
from tqdm import tqdm
import random

SAMPLE_STRATEGIES = ["random", "stratified", "city_random", "city_rank"]


def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)

def sample(
        query_embedding, 
        passage_ids, 
        passage_embeddings, 
        passage_dict, 
        sample_strategy, 
        sample_size=200, 
        top_k=100
    ):
    """
    Sample passages based on different strategies.

    Args:
        query_embedding (np.array): Query embedding vector.
        passage_ids (list): List of passage IDs.
        passage_embeddings (np.array): Matrix of passage embeddings.
        passage_dict (dict): Mapping of city_id -> list of passage_ids.
        sample_strategy (str): Sampling strategy ("random", "stratified", "city_random", "city_rank").
        sample_size (int): Total number of passages to sample.
        top_k (int): Top-k passages to consider for each strategy.

    Returns:
        list: List of sampled passage IDs.
    """

    if sample_strategy not in SAMPLE_STRATEGIES:
        raise ValueError(f"Invalid sample strategy. Choose from: {SAMPLE_STRATEGIES}")

    # Ensure deterministic results
    random.seed(42)
    np.random.seed(42)

    # Create a fast lookup from passage_id to index
    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}

    # Compute cosine similarity for ranking
    cosine_similarity_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
    sorted_idx = np.argsort(cosine_similarity_matrix)[::-1]  # Descending order

    if sample_strategy in ["random", "stratified"]:
        # Top-k sampling from the overall dense retrieval results
        top_sampled_ids = [passage_ids[idx] for idx in sorted_idx[:top_k]]

        # Remaining sample size after selecting top-k
        remaining_sample_size = max(0, sample_size - len(top_sampled_ids))
        remaining_idx = sorted_idx[top_k:]

        if remaining_sample_size == 0:
            return top_sampled_ids

        if sample_strategy == "random":
            # Randomly select from the remaining passages
            sampled_idx = np.random.choice(
                remaining_idx, 
                size=min(remaining_sample_size, len(remaining_idx)), 
                replace=False
            )

        elif sample_strategy == "stratified":
            strata = np.array_split(remaining_idx, min(remaining_sample_size, len(remaining_idx)))
            sampled_idx = [np.random.choice(stratum, size=1)[0] for stratum in strata if len(stratum) > 0]


        # Combine top and stratified/random samples
        remaining_ids = [passage_ids[idx] for idx in sampled_idx]
        sampled_ids = top_sampled_ids + remaining_ids

        # Shuffle for randomness
        random.shuffle(sampled_ids)
        return sampled_ids

    else:
        sampled_ids = []
        for city_id, city_passage_ids in passage_dict.items():
            city_passage_idx = [id_to_index[pid] for pid in city_passage_ids]
            city_passage_embeddings = passage_embeddings[city_passage_idx]
            
            if sample_strategy == "city_random":
                # Randomly sample from each city's passages
                sampled_ids.extend(random.sample(city_passage_ids, min(top_k, len(city_passage_ids))))
            
            elif sample_strategy == "city_rank":
                # Rank passages based on city-level similarity
                city_cosine_similarity = calculate_cosine_similarity(query_embedding, city_passage_embeddings)
                sorted_city_idx = np.argsort(city_cosine_similarity)[::-1][:top_k]
                sampled_ids.extend([city_passage_ids[idx] for idx in sorted_city_idx])

        # Rank all selected passages based on overall similarity
        if len(sampled_ids) > sample_size:
            sampled_ids_scores = [cosine_similarity_matrix[id_to_index[pid]] for pid in sampled_ids]
            top_k_idx = np.argsort(sampled_ids_scores)[::-1][:sample_size]
            sampled_ids = [sampled_ids[idx] for idx in top_k_idx]

        return sampled_ids

def gp_retrieval(
        # Core input data
        query: str, 
        query_embedding: np.array, 
        query_id: int,
        passage_ids: list, 
        passage_embeddings: np.array, 
        passages: list, 
        passage_dict: dict[list],
        llm: LLM, 
        kernel: str = "rbf",
        llm_budget: int = 200, 
        top_k: int = 100,
        sample_strategy: str = "random",
        batch_size: int = 5, 
        cache: dict = None, 
        update_cache = None,
        verbose: bool = False
    ):
    """
    Perform GP-based retrieval using LLM for scoring.

    Args:
        query (str): Input query.
        query_embedding (np.array): Query embedding.
        query_id (int): Query ID.

        passage_ids (list): List of passage IDs.
        passage_embeddings (np.array): Matrix of passage embeddings.
        passages (list): List of passage texts.
        passage_dict (dict): Mapping of city_id -> list of passage_ids.

        llm (LLM): LLM instance for scoring.
        llm_budget (int): Maximum number of passages to query with LLM.
        batch_size (int): Batch size for LLM calls.
        cache (dict): Cache for LLM results.
        update_cache: Whether to update the cache (boolean or dictionary).

        top_k (int): Number of top passages to return.
        sample_strategy (str): Sampling strategy ("random", "stratified", "city_random", "city_rank").

        verbose (bool): Whether to print debug info.

    Returns:
        list: Top-k passage IDs.
        list (optional): Top-k scores if return_score is True.
    """

    ############### Set Up ################
    if llm_budget < 1:
        raise ValueError("LLM budget must be at least 1")

    # Fast lookup for embeddings
    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}

    # Initialize GP-UCB model
    gpucb = GaussianProcess(kernel=kernel)

    # To store observed scores
    scores = {}
    observed_ids = []

    ############### Sampling ################
    sampled_ids = sample(
        query_embedding=query_embedding,
        passage_ids=passage_ids,
        passage_embeddings=passage_embeddings,
        passage_dict=passage_dict,
        sample_strategy=sample_strategy,
        sample_size=llm_budget,
        top_k=top_k
    )

    ############### Batch Processing ################
    batches = [
        sampled_ids[i:i + batch_size] for i in range(0, len(sampled_ids), batch_size)
    ]

    for batch in batches:
        # Get passages and embeddings for the batch using fast lookup
        passage_idxs = [id_to_index[pid] for pid in batch]
        batch_passages = [passages[idx] for idx in passage_idxs]

        # Get relevance scores using LLM
        batch_scores = llm.get_score(
            query=query,
            passages=batch_passages,
            query_id=query_id,
            passage_ids=batch,
            cache=cache,
            update_cache=update_cache
        )

        # Store scores and update GP-UCB model
        for pid, score in zip(batch, batch_scores):
            scores[pid] = score
            observed_ids.append(pid)
            gpucb.update(passage_embeddings[id_to_index[pid]], score)

        # Verbose output for debugging
        if verbose:
            for pid, score, passage in zip(batch, batch_scores, batch_passages):
                try:
                    print(f"Score: {score:.4f}, Passage: {passage}")
                except UnicodeEncodeError:
                    print(f"Score: {score:.4f}, Passage: {passage.encode('ascii', 'replace').decode()}")

    ############### Return Results ################
    return gpucb

def bandit_retrieval(
        passage_ids: list, 
        passage_embeddings: list, 
        passages: list, 
        llm: LLM, 
        query, 
        query_embedding, 
        query_id, 
        beta=2.0, 
        acq_func="ucb",
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
    
    gpucb = RetrievalGPUCB(beta=beta, kernel=kernel, acquisition_function=acq_func) # set up GP-UCB
    
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
        print("there are ", len(cold_start_ids), " cold start ids")
        print("cold start ids: ", cold_start_ids)
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
        print("there are ", len(available_ids), " available embeddings")
        
        next_embedding_idxs = gpucb.select(available_embeddings, batch_size) 
        next_ids = [available_ids[idx] for idx in next_embedding_idxs] 
        print("Next IDs: ", next_ids)

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
        return top_k_ids, top_k_scores, observed_ids

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
    passage_ids, dense_score = dense_retrieval(passage_ids, passage_embeddings, query_embedding, k_retrieval=k_retrieval, return_score=True)
    dense_score_dict = {pid: score for pid, score in zip(passage_ids, dense_score)}
    if cache:
        try:
            valid_cached_items = {
                pid: score for pid, score in cache[query_id].items() if pid in passage_ids
            }
            
            sorted_item = sorted(valid_cached_items.items(), key=lambda x: (x[1], dense_score_dict[x[0]]), reverse=True)[:k_retrieval]            
            sorted_passages = [int(key) for key, _ in sorted_item]
            if return_score:
                sorted_scores = [value for _, value in sorted_item]
                return sorted_passages, sorted_scores
            return sorted_passages
        
        except:
            pass
        
    print(f"Cache miss for query {query_id}, using LLM ...")
    print("Please run llm_baseline_runner.py to generate LLM scores for the query")
    
    