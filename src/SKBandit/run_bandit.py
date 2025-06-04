from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize

from src.SKBandit.gpei import GPEI
from src.SKBandit.gppi import GPPI
from src.SKBandit.gprandom import GPRandom
from src.SKBandit.gpthompson_sub import GPThompsonSub
from src.SKBandit.gpucb import GPUCB
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

        kernel: str = "rbf",
        acq_func:str="ucb",
        alpha:float=1e-3,
        alpha_method:Optional[str]=None,
        length_scale:float=1,
        beta:float=2.0,
        nu:float=2.5,
        xi:float=0.01,

        use_query:Optional[int]=None,
        normalize_passage:bool=False,
        ard:bool=False,

        k_retrieval: int = 1000,
        batch_size: int = 1,
        return_score: bool = False,
        cache: dict = None,
        update_cache: str = None,
        verbose: bool = False

):
    """
    Perform passage retrieval using a Gaussian Process bandit (e.g., GP-UCB).

    Args:
        llm (LLM): LLM instance used to assign relevance scores to passages.
        query (str): Input query text.
        query_id (str): Identifier for the query.
        query_embedding (np.ndarray): Embedding vector for the query.
        passages (list[str]): List of passage texts to rank.
        passage_ids (list[str]): List of IDs corresponding to each passage.
        passage_embeddings (np.ndarray): Embedding matrix for the passages.

        llm_budget (int): Total number of LLM queries allowed.
        k_cold_start (int): Number of passages to sample during initial cold start.

        kernel (str): Kernel type to use in the GP (e.g., "rbf", "matern").
        acq_func (str): Acquisition function to use (e.g., "ucb", "ei", "pi").
        alpha (float): Noise level or regularization term for the GP.
        alpha_method (str): Strategy to set per-sample noise (alpha) based on LLM uncertainty.
        length_scale (float): Length scale parameter for the GP kernel.
        beta (float): Exploration weight for GP-UCB.
        nu (float): Smoothness parameter for the Matern kernel.
        xi (float): Exploration parameter used in GP-EI and GP-PI.

        use_query (Optional[int]): Optional override for query index when initializing the model.
        normalize_passage (bool): Whether to apply L2 normalization to passage embeddings.

        ard (bool): Whether to use Automatic Relevance Determination (per-dimension length scales).
        k_retrieval (int): Number of final passages to retrieve and return.
        batch_size (int): Number of passages per LLM batch during evaluation.
        return_score (bool): If True, return final GP scores along with passage IDs.
        cache (dict, optional): Dictionary cache to store or reuse LLM scores.
        update_cache (str, optional): Path to update CSV cache with new results.
        verbose (bool): If True, print debug and progress information.
    """
    if verbose:
        print(f"\n > Query {query_id}")
    assert llm_budget > 0, "llm_budget should be greater than 0"
    assert k_cold_start <= llm_budget, "k_cold_start should be less than or equal to llm_budget"

    if ard:
        length_scale = [length_scale] * passage_embeddings.shape[1]
    if acq_func == "ucb":
        bandit = GPUCB(beta=beta, kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu)
    elif acq_func == "thompson":
        # bandit = GPThompson(kernel=kernel, alpha=alpha, length_scale=length_scale, nu=nu)
        bandit = GPThompsonSub(kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu)
    elif acq_func == "random":
        bandit = GPRandom(kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu)
    elif acq_func == "greedy":
        k_cold_start = llm_budget
        bandit = GPUCB(beta=beta, kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu)
    elif acq_func == "ei":
        bandit = GPEI(kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu, xi=xi)
    elif acq_func == "pi":
        bandit = GPPI(kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu, xi=xi)
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

            batch_scores, batch_logit = llm.get_score(
                queries=[query],
                passages=batch_passages,
                query_ids=[query_id],
                passage_ids=batch,
                cache=cache,
                update_cache=update_cache,
                verbose=verbose
            )

            # store observations
            for pid, score, logit in zip(batch, batch_scores, batch_logit):
                score = float(score)
                cold_scores[pid] = score
                bandit.update(pid2emb[pid], score, logit)

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

        batch_scores, batch_logit = llm.get_score(
            queries=[query],
            passages=batch_passages,
            query_ids=[query_id],
            passage_ids=next_ids,
            cache=cache,
            update_cache=update_cache,
            verbose=verbose
        )

        for pid, score, logit in zip(next_ids, batch_scores, batch_logit):
            score = float(score)
            bandit_scores[pid] = score
            bandit.update(pid2emb[pid], score, logit)

    if verbose:  # debug print
        print(" >>> Bandit sbatch scores: ", bandit_scores)


    # return the top-k passages based on final GP predictions
    top_k_idx, top_k_scores = bandit.get_top_k(passage_embeddings, k_retrieval, return_scores=return_score)
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    if return_score:
        return top_k_ids, top_k_scores.tolist(), cold_scores|bandit_scores

    return top_k_ids
