import numpy as np
from src.GPUCB.gp import GaussianProcess
from src.GPUCB.retrieval_gpucb import RetrievalGPUCB
from src.LLM.llm import LLM
from tqdm import tqdm
import random
import time

SAMPLE_STRATEGIES = ["random", "stratified", "city_random", "city_rank"]

def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    dot_product = np.dot(query_embeddings, passage_embeddings.T)
    return dot_product

def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.reshape(1, -1)

    # Normalize embeddings
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    passage_norm = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)

    # Compute cosine similarity
    similarity_matrix = np.dot(query_norm, passage_norm.T)
    return similarity_matrix.flatten()

def sample(
        query_embedding, 
        passage_ids, 
        passage_embeddings, 
        passage_dict, 
        sample_strategy, 
        sample_size=200, 
        epsilon=0.5,
        city_max_sample=1,
        random_seed=42
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
        epsilon (float): Exploration factor (0 = pure exploitation, 1 = pure exploration).
        city_max_sample (int): Maximum number of passages to sample per city (for city-based strategies).

    Returns:
        list: List of sampled passage IDs.
    """
    if sample_strategy not in SAMPLE_STRATEGIES:
        raise ValueError(f"Invalid sample strategy. Choose from: {SAMPLE_STRATEGIES}")

    # Ensure deterministic results
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create a fast lookup from passage_id to index
    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}

    # Step 1: Compute cosine similarity for ranking
    cosine_similarity_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
    sorted_idx = np.argsort(cosine_similarity_matrix)[::-1]  # Descending order
    sorted_scores = cosine_similarity_matrix[sorted_idx]

    # Step 2: Split based on epsilon (handle boundary cases)
    if epsilon == 0:   # Pure exploitation
        top_k = sample_size
        explore_k = 0
    elif epsilon == 1: # Pure exploration
        top_k = 0
        explore_k = sample_size
    else:
        top_k = int(sample_size * (1 - epsilon))
        explore_k = sample_size - top_k
    
    # Exploitation step — take top-ranked passages
    top_sampled_ids = []
    if top_k > 0:
        top_sampled_ids = [passage_ids[idx] for idx in sorted_idx[:top_k]]

    # Exploration step — handle remaining sampling based on strategy
    remaining_idx = sorted_idx[top_k:]
    remaining_ids = [passage_ids[idx] for idx in remaining_idx]

    explored_ids = []
    if explore_k > 0:
        if sample_strategy == "random":
            # Randomly sample from the remaining passages
            sampled_idx = np.random.choice(
                remaining_idx, 
                size=min(explore_k, len(remaining_idx)), 
                replace=False
            )
            explored_ids = [passage_ids[idx] for idx in sampled_idx]

        elif sample_strategy == "stratified":
            # Split into roughly equal strata and sample from each
            strata = np.array_split(remaining_idx, min(explore_k, len(remaining_idx)))
            sampled_idx = [
                np.random.choice(stratum, size=1)[0] 
                for stratum in strata if len(stratum) > 0
            ]
            explored_ids = [passage_ids[idx] for idx in sampled_idx]

        elif sample_strategy in ["city_random", "city_rank"]:
            remaining_ids = []
            for city_id, city_passage_ids in passage_dict.items():
                valid_city_passage_ids = [pid for pid in city_passage_ids if pid in id_to_index]
                if not valid_city_passage_ids:
                    continue
                
                city_passage_idx = [id_to_index[pid] for pid in valid_city_passage_ids]
                city_passage_embeddings = passage_embeddings[city_passage_idx]

                if sample_strategy == "city_random":
                    # Random sampling from each city
                    explored_ids.extend(
                        random.sample(valid_city_passage_ids, min(city_max_sample, len(valid_city_passage_ids)))
                    )

                # elif sample_strategy == "city_rank":
                #     # Rank passages based on similarity within the city
                #     city_cosine_similarity = calculate_cosine_similarity(query_embedding, city_passage_embeddings)
                #     sorted_city_idx = np.argsort(city_cosine_similarity)[::-1][:city_max_sample]
                #     rexplored_ids.extend([valid_city_passage_ids[idx] for idx in sorted_city_idx])

            # Limit final sample size after city-based sampling
            if len(explored_ids) > explore_k:
                remaining_ids = random.sample(remaining_ids, explore_k)

    # Combine exploitation and exploration samples
    sampled_ids = top_sampled_ids + explored_ids

    # Handle edge cases where not enough data is available
    if len(sampled_ids) < sample_size:
        available_ids = list(set(passage_ids) - set(sampled_ids))
        additional_samples = random.sample(
            available_ids, 
            min(sample_size - len(sampled_ids), len(available_ids))
        )
        sampled_ids.extend(additional_samples)

    # Shuffle for randomness
    random.shuffle(sampled_ids)

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
        epsilon: float = 0.5,
        city_max_sample: int = 1,
        sample_strategy: str = "random",
        batch_size: int = 5, 
        cache: dict = None, 
        update_cache = None,
        verbose: bool = False,
        random_seed: int = 42
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
    # Fast lookup for embeddings
    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}

    # Initialize GP-UCB model
    gpucb = GaussianProcess(kernel=kernel)

    gpucb.update(query_embedding, 3.0)

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
        epsilon=epsilon,
        city_max_sample=city_max_sample,
        random_seed=random_seed
    )

    ############### Batch Processing ################
    batches = [
        sampled_ids[i:i + batch_size] for i in range(0, len(sampled_ids), batch_size)
    ]

    for batch in batches:
        start = time.time()
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
        query: str, 
        query_embedding: np.array, 
        query_id: int,
        passage_ids: list, 
        passage_embeddings: np.array, 
        passages: list, 
        llm: LLM, 
        kernel: str = "rbf",
        acq_func="ucb",
        k_cold_start: int=5, 
        k_retrieval: int=1000,
        beta=2.0, 
        llm_budget: int=10, 
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
    rerank using LLM; append dense results beyond k_retrieval without reranking
    """
    # Step 1: Retrieve top N (>= k_retrieval) with dense retrieval
    passage_ids, dense_score = dense_retrieval(
        passage_ids, passage_embeddings, query_embedding, 
        k_retrieval=max(k_retrieval * 2, len(passage_ids)), return_score=True
    )
    
    # Build dense score dict for all
    dense_score_dict = {pid: score for pid, score in zip(passage_ids, dense_score)}
    
    # Separate top-k for reranking and rest
    top_k_passages = passage_ids[:k_retrieval]
    rest_passages = passage_ids[k_retrieval:]

    if cache:
        try:
            valid_cached_items = {
                pid: score for pid, score in cache[query_id].items() if pid in top_k_passages
            }
            # Use dense_score_dict as tiebreaker
            sorted_item = sorted(
                valid_cached_items.items(), 
                key=lambda x: (x[1], dense_score_dict[x[0]]), 
                reverse=True
            )
            reranked_passages = [int(pid) for pid, _ in sorted_item]
            reranked_scores = [score for _, score in sorted_item]
            
            # Fill in any missing top-k passages not found in cache
            remaining_in_top_k = [pid for pid in top_k_passages if pid not in valid_cached_items]
            reranked_passages += remaining_in_top_k
            reranked_scores += [dense_score_dict[pid] for pid in remaining_in_top_k]

            # Append the rest of the dense results
            final_passages = reranked_passages + rest_passages
            if return_score:
                final_scores = reranked_scores + [dense_score_dict[pid] for pid in rest_passages]
                return final_passages, final_scores
            return final_passages
        except:
            pass

    print(f"Cache miss for query {query_id}, using LLM ...")
    print("Please run llm_baseline_runner.py to generate LLM scores for the query")
    return passage_ids[:k_retrieval]  # fallback

def cross_encoder_rerank(
        passage_ids: list[int], 
        passage_embeddings: list, 
        passages_text: list,
        query_embedding, 
        query_id: int, 
        k_retrieval: int=1000,
        return_score: bool=False, 
        cross_encoder_model=None,
        query_text: str=None
    ):
    """
    Rerank using a cross-encoder model; append dense results beyond k_retrieval without reranking.
    
    Args:
        passage_ids: List of passage IDs
        passage_embeddings: List of passage embeddings
        query_embedding: Query embedding vector
        query_id: Query ID for cache lookup
        k_retrieval: Number of passages to retrieve
        return_score: Whether to return scores along with passage IDs
        cache: Cache dictionary for storing cross-encoder scores
        cross_encoder_model: The cross-encoder model to use for reranking
        query_text: The text of the query (required for cross-encoder)
        
    Returns:
        List of passage IDs ranked by relevance, and optionally their scores
    """
    if cross_encoder_model is None:
        raise ValueError("Cross-encoder model must be provided")
    
    if query_text is None:
        raise ValueError("Query text must be provided for cross-encoder reranking")
    
    # Step 1: Retrieve top N (>= k_retrieval) with dense retrieval
    passage_ids, dense_score = dense_retrieval(
        passage_ids, passage_embeddings, query_embedding, 
        k_retrieval=max(k_retrieval * 2, len(passage_ids)), return_score=True
    )
    
    # Build dense score dict for all
    dense_score_dict = {pid: score for pid, score in zip(passage_ids, dense_score)}
    
    # Separate top-k for reranking and rest
    top_k_passages = passage_ids[:k_retrieval]
    rest_passages = passage_ids[k_retrieval:]
    top_k_passages_idx = [passage_ids.index(pid) for pid in top_k_passages]
    
    # Get passage texts for the top-k passages
    # Note: This assumes passage_ids are indices into a passage collection
    # You may need to adjust this based on your actual data structure
    passage_texts = [passages_text[idx] for idx in top_k_passages_idx]
    
    # Create query-passage pairs for cross-encoder
    query_passage_pairs = [(query_text, passage_text) for passage_text in passage_texts]
    
    # Get cross-encoder scores
    cross_encoder_scores = cross_encoder_model.predict(query_passage_pairs)
    
    # Create a dictionary mapping passage IDs to cross-encoder scores
    cross_encoder_score_dict = {pid: score for pid, score in zip(top_k_passages, cross_encoder_scores)}
    
    # Sort passages by cross-encoder scores
    sorted_items = sorted(
        cross_encoder_score_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Extract reranked passages and scores
    reranked_passages = [pid for pid, _ in sorted_items]
    reranked_scores = [score for _, score in sorted_items]
    
    # Append the rest of the dense results
    final_passages = reranked_passages + rest_passages
    
    if return_score:
        final_scores = reranked_scores + [dense_score_dict[pid] for pid in rest_passages]
        return final_passages, final_scores
    
    return final_passages

    