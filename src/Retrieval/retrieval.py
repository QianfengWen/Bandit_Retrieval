import numpy as np
from typing import List, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
import random
import time

from ..GPR.gp import GaussianProcess
from ..LLM.llm import LLM

SAMPLE_STRATEGIES = ["random"]

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
        sample_strategy,
        sample_size=200,
        epsilon=0.5,
        random_seed=42,
        tau=None,
    ):
    """
    Select a subset of passages to send to the LLM.

    This helper first scores passages with dense retrieval and then performs an
    epsilon-greedy split between exploitation (`1 - epsilon` fraction) and
    exploration (`epsilon` fraction).  The optional ``tau`` parameter caps
    exploration to the highest ranked dense candidates so that random sampling
    never strays into the extreme tail of the retrieval list.

    Args:
        query_embedding (np.array): Embedding representation of the query.
        passage_ids (list[int]): Identifiers for every candidate passage.
        passage_embeddings (np.array): Embeddings aligned with ``passage_ids``.
        sample_strategy (str): Strategy for the exploration portion
            (``"random"`` or ``"stratified"``).
        sample_size (int): Number of passages to return.
        epsilon (float): Fraction of samples allocated to exploration.
        random_seed (int): Seed used when sampling to ensure reproducibility.
        tau (int, optional): Limits the exploration pool to the top ``tau`` dense
            retrieval results.  Defaults to the full candidate list.

    Returns:
        list[int]: Selected passage identifiers.
    """
    if sample_strategy not in SAMPLE_STRATEGIES:
        raise ValueError(f"Invalid sample strategy. Choose from: {SAMPLE_STRATEGIES}")

    # Ensure deterministic results
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Step 1: Compute cosine similarity for ranking
    cosine_similarity_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
    sorted_idx = np.argsort(cosine_similarity_matrix)[::-1]  # Descending order

    # Restrict exploration pool to top-τ dense rankings if provided
    if tau is not None:
        tau = max(0, min(int(tau), len(sorted_idx)))
        candidate_idx = sorted_idx[:tau]
    else:
        candidate_idx = sorted_idx.copy()

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
    candidate_count = len(candidate_idx)
    top_k = min(top_k, candidate_count)

    top_sampled_ids = []
    if top_k > 0:
        top_sampled_ids = [passage_ids[idx] for idx in candidate_idx[:top_k]]

    # Exploration step — handle remaining sampling based on strategy
    remaining_idx = candidate_idx[top_k:]

    explored_ids = []
    if explore_k > 0:
        explore_pool = remaining_idx if len(remaining_idx) > 0 else candidate_idx[top_k:]
        if sample_strategy == "random":
            # Randomly sample from the remaining passages
            if len(explore_pool) > 0:
                sampled_idx = np.random.choice(
                    explore_pool,
                    size=min(explore_k, len(explore_pool)),
                    replace=False
                )
                explored_ids = [passage_ids[idx] for idx in sampled_idx]

        elif sample_strategy == "stratified":
            # Split into roughly equal strata and sample from each
            if len(explore_pool) > 0:
                strata = np.array_split(explore_pool, min(explore_k, len(explore_pool)))
                sampled_idx = [
                    np.random.choice(stratum, size=1)[0]
                    for stratum in strata if len(stratum) > 0
                ]
                explored_ids = [passage_ids[idx] for idx in sampled_idx]

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
        tau: int = None,
        sample_strategy: str = "random",
        batch_size: int = 5, 
        cache: dict = None, 
        update_cache = None,
        verbose: bool = False,
        random_seed: int = 42,
        normalize_y: bool = True,
        alpha: float = 1e-3,
        length_scale: float = 1.0
    ):
    """
    Run the GP retrieval loop with LLM feedback.

    The routine repeatedly samples passages with ``epsilon``-greedy exploration,
    queries the LLM for relevance signals, and updates a Gaussian Process model
    that produces the downstream ranking.

    Args:
        query (str): Raw query text.
        query_embedding (np.array): Embedding for ``query``.
        query_id (int): Unique identifier for the query (used for caching).
        passage_ids (list[int]): Identifiers for all candidate passages.
        passage_embeddings (np.array): Embeddings aligned with ``passage_ids``.
        passages (list[str]): Passage texts used for LLM prompting.
        passage_dict (dict): Mapping from city IDs to passage IDs.
        llm (LLM): Scoring function compatible with :class:`OpenRouterLLM`.
        kernel (str): Kernel name for the GP model.
        llm_budget (int): Number of LLM calls permitted for this query.
        epsilon (float): Exploration rate for the sampler.
        tau (int): Cap on the exploration pool; defaults to ``len(passage_ids)``.
        sample_strategy (str): Strategy string forwarded to :func:`sample`.
        batch_size (int): Number of passages sent to the LLM at once.
        cache (dict): Optional cache of previously scored passages.
        update_cache (str): Path used to persist newly collected labels.
        verbose (bool): Print debugging details when True.
        random_seed (int): Seed for reproducible sampling.
        normalize_y (bool): Whether to normalise observations in the GP.
        alpha (float): Observation noise term supplied to the GP.
        length_scale (float): Kernel length-scale hyper-parameter.

    Returns:
        GaussianProcess: Updated GP model containing the observed labels.
    """

    ############### Set Up ################
    # Fast lookup for embeddings
    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}

    # Initialize GP-UCB model
    gpucb = GaussianProcess(kernel=kernel, llm_budget=llm_budget, normalize_y=normalize_y, alpha=alpha, length_scale=length_scale)

    gpucb.update(query_embedding, 3.0)

    # To store observed scores
    scores = {}
    observed_ids = []

    ############### Sampling ################
    sampled_ids = sample(
        query_embedding=query_embedding,
        passage_ids=passage_ids,
        passage_embeddings=passage_embeddings,
        sample_strategy=sample_strategy,
        sample_size=llm_budget,
        epsilon=epsilon,
        random_seed=random_seed,
        tau=tau if tau is not None else len(passage_ids)
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
        passage_ids: List[int],
        passage_embeddings: list,
        passages_text: List[str],
        query_embedding,
        query_id: int,
        query_text: Optional[str] = None,
        llm: Optional[LLM] = None,
        k_retrieval: int = 1000,
        return_score: bool = False,
        cache: Optional[dict] = None,
        update_cache: Optional[str] = None,
    ):
    """
    Rerank passages for a query using LLM guidance.

    The function first obtains dense retrieval scores, then replaces the top ``k``
    scores with LLM-derived estimates when available.  Missing scores fall back to
    dense retrieval ensuring the reranker always returns ``k_retrieval`` items.

    Args:
        passage_ids (list[int]): Identifiers for all passages.
        passage_embeddings (np.array): Embeddings aligned with ``passage_ids``.
        passages_text (list[str]): Text for each passage, used in the LLM prompt.
        query_embedding (np.array): Embedding for the query.
        query_id (int): Numerical identifier for caching.
        query_text (str, optional): Raw query text for prompting the LLM.
        llm (LLM, optional): Scoring model. Required for real-time scoring.
        k_retrieval (int): Number of passages to surface.
        return_score (bool): When True returns passages and their scores.
        cache (dict, optional): Pre-computed LLM judgements.
        update_cache (str, optional): CSV path for persisting new scores.

    Returns:
        list[int] | tuple[list[int], list[float]]: Reranked passage IDs, optionally
        paired with their aggregated scores.
    """
    passage_lookup = {pid: text for pid, text in zip(passage_ids, passages_text)}

    # Step 1: Retrieve top N (>= k_retrieval) with dense retrieval
    dense_ids, dense_score = dense_retrieval(
        passage_ids,
        passage_embeddings,
        query_embedding,
        k_retrieval=max(k_retrieval * 2, len(passage_ids)),
        return_score=True,
    )
    
    # Build dense score dict for all
    dense_score_dict = {pid: score for pid, score in zip(dense_ids, dense_score)}
    
    # Separate top-k for reranking and rest
    top_k_passages = dense_ids[:k_retrieval]
    rest_passages = dense_ids[k_retrieval:]

    llm_scores = {}
    if cache is not None and query_id is not None:
        query_cache = cache.get(query_id, {})
        llm_scores = {pid: query_cache[pid] for pid in top_k_passages if pid in query_cache}

    missing_passages = [pid for pid in top_k_passages if pid not in llm_scores]
    if missing_passages and llm is not None and query_text is not None:
        passages_for_llm = [passage_lookup.get(pid, "") for pid in missing_passages]
        llm_scores_new = llm.get_score(
            query=query_text,
            passages=passages_for_llm,
            query_id=query_id,
            passage_ids=missing_passages,
            cache=cache,
            update_cache=update_cache,
        )
        if cache is not None and query_id is not None:
            query_cache = cache.get(query_id, {})
            llm_scores.update({
                pid: query_cache.get(pid, score)
                for pid, score in zip(missing_passages, llm_scores_new)
            })
        else:
            llm_scores.update({pid: score for pid, score in zip(missing_passages, llm_scores_new)})

    # Combine scores, falling back to dense retrieval values where needed
    rerank_scores = {}
    for pid in top_k_passages:
        score = llm_scores.get(pid)
        if score is None or score < 0:
            score = dense_score_dict[pid]
        rerank_scores[pid] = score

    sorted_top = sorted(
        top_k_passages,
        key=lambda pid: (rerank_scores[pid], dense_score_dict[pid]),
        reverse=True,
    )

    final_passages = sorted_top + rest_passages
    if return_score:
        final_scores = (
            [rerank_scores[pid] for pid in sorted_top]
            + [dense_score_dict[pid] for pid in rest_passages]
        )
        return final_passages, final_scores

    return final_passages

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

    
