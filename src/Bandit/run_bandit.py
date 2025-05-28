from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize

from src.Bandit.gpei import GPEI
from src.Bandit.gppi import GPPI
from src.Bandit.gprandom import GPRandom
from src.Bandit.gpthompson_sub import GPThompsonSub
from src.Bandit.gpucb import GPUCB
from src.LLM.llm import LLM
from src.utils import cosine_similarity


def bandit_retrieval(
        llm: LLM,
        query: str,
        query_id: str,
        query_embedding: np.ndarray,
        passages: list[str],
        passage_ids: list[str],
        passage_embeddings: np.ndarray,
        llm_budget: int,
        k_cold_start: int,
        use_query:Optional[int]=None,
        alpha:float=1e-3,
        length_scale:float=1,
        nu:float=2.5,
        acq_func:str="ucb",
        beta:float=2.0,
        xi:float=0.01,
        normalize_passage:bool=False,
        ard:bool=False,
        k_retrieval: int = 1000,
        kernel: str = "rbf",
        batch_size: int = 1,
        return_score: bool = False,
        cache: dict = None,
        update_cache: str = None,
        verbose: bool = False

):
    """
    Perform bandit retrieval using GP-UCB.
    Args:
        llm: LLM instance to use for scoring passages.
        query: The query string.
        query_id: The ID of the query.
        query_embedding: The embedding of the query.
        passages: List of passage texts.
        passage_ids: List of passage IDs.
        passage_embeddings: The embeddings of the passages.
        llm_budget: Total number of LLM calls allowed.
        k_cold_start: Number of passages to retrieve in the cold start phase.
        use_query: Optional query embedding to use for the initial update.
        alpha: Regularization parameter for the GP.
        length_scale: Length scale for the GP kernel.
        nu: Parameter for the Matern kernel.
        acq_func: Acquisition function.
        beta: Exploration parameter for the GP-UCB.
        xi: Exploration parameter for the GP-EI, GP-PI.
        normalize_passage: Whether to normalize passage embeddings.
        ard: Whether to use ARD (Automatic Relevance Determination).
        k_retrieval: Number of passages to retrieve at the end.
        kernel: Kernel type for the GP.
        batch_size: Number of passages to process in each LLM call.
        return_score: Whether to return the scores of the retrieved passages.
        cache: Cache for storing LLM scores.
        update_cache: Path to the CSV file to update with new results.
        verbose: Whether to print debug information.
    """
    if verbose:
        print(f"\n > Query {query_id}")
    assert llm_budget > 0, "llm_budget should be greater than 0"
    assert k_cold_start <= llm_budget, "k_cold_start should be less than or equal to llm_budget"

    if ard:
        length_scale = [length_scale] * passage_embeddings.shape[1]
    if acq_func == "ucb":
        bandit = GPUCB(beta=beta, kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu)
    elif acq_func == "thompson":
        # bandit = GPThompson(kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu)
        bandit = GPThompsonSub(kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu)
    elif acq_func == "random":
        bandit = GPRandom(kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu)
    elif acq_func == "greedy":
        k_cold_start = llm_budget
        bandit = GPUCB(beta=beta, kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu)
    elif acq_func == "ei":
        bandit = GPEI(kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu, xi=xi)
    elif acq_func == "pi":
        bandit = GPPI(kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu, xi=xi)
    else:
        raise ValueError(f"Invalid acquisition function: {acq_func}")

    if normalize_passage:
        org_passage_embeddings = passage_embeddings.copy()
        passage_embeddings = normalize(passage_embeddings, norm='l2')

        org_query_embedding = query_embedding.copy()
        query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2').flatten()
    else:
        org_passage_embeddings = passage_embeddings
        org_query_embedding = query_embedding

    if use_query is not None:
        bandit.update(query_embedding, use_query)

    pid2passage = {pid: passages[idx] for idx, pid in enumerate(passage_ids)}
    pid2emb = {pid: org_passage_embeddings[idx] for idx, pid in enumerate(passage_ids)}
    available_pids = set(passage_ids)
    cold_scores = {}
    bandit_scores = {}

    ############### Cold start ################
    if k_cold_start > 0:
        cossim_matrix = cosine_similarity(org_query_embedding, org_passage_embeddings)
        cold_start_idx = np.argsort(cossim_matrix)[::-1][:k_cold_start]
        cold_start_passage_ids = [passage_ids[idx] for idx in cold_start_idx]
        available_pids = list(available_pids - set(cold_start_passage_ids))

        if verbose:
            print(" >> Cold start phase")
            print(" >>> There are ", len(cold_start_passage_ids), " cold start ids")

        for batch_idx in range(0, len(cold_start_passage_ids), batch_size):
            batch = cold_start_passage_ids[batch_idx:batch_idx + batch_size]

            batch_passages = [pid2passage[pid] for pid in batch]

            batch_scores, _ = llm.get_score(
                queries=[query],
                passages=batch_passages,
                query_ids=[query_id],
                passage_ids=batch,
                cache=cache,
                update_cache=update_cache,
                verbose=verbose
            )

            # store observations
            for pid, score in zip(batch, batch_scores):
                score = float(score)
                cold_scores[pid] = score
                bandit.update(pid2emb[pid], score)

        if verbose:  # debug print
            print(" >>> Cold start batch scores: ", cold_scores)

    ############### Exploration-exploitation ################
    if verbose:
        print(f" >> Exploration-exploitation phase")

    for _ in range(0, (llm_budget - k_cold_start), batch_size):
        if not available_pids:
            break

        available_embeddings = np.stack([pid2emb[p] for p in available_pids])
        next_embedding_idxes = bandit.select(available_embeddings, batch_size)
        next_ids = [available_pids[idx] for idx in next_embedding_idxes]

        for pid in next_ids:
            available_pids.remove(pid)

        batch_passages = [pid2passage[pid] for pid in next_ids]

        batch_scores, _ = llm.get_score(
            queries=[query],
            passages=batch_passages,
            query_ids=[query_id],
            passage_ids=next_ids,
            cache=cache,
            update_cache=update_cache,
            verbose=verbose
        )

        for pid, score in zip(next_ids, batch_scores):
            score = float(score)
            bandit_scores[pid] = score
            bandit.update(pid2emb[pid], score)  # Update model

    if verbose:  # debug print
        print(" >>> Bandit sbatch scores: ", bandit_scores)


    # return the top-k passages based on final GP predictions
    top_k_idx, top_k_scores = bandit.get_top_k(passage_embeddings, k_retrieval, return_scores=return_score)
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    if return_score:
        return top_k_ids, top_k_scores.tolist(), cold_scores|bandit_scores

    return top_k_ids
