from typing import Optional

import numpy as np
import torch

from src.GPBandit.basegp import BaseGP
from src.GPBandit.gpei import GPEI
from src.GPBandit.gppi import GPPI
from src.GPBandit.gprandom import GPRandom
from src.GPBandit.gpucb import GPUCB
from src.GPBandit.thompson import GPThompson
from src.utils import get_cache


def gp_bandit_retrieval_optimized(
        query: str,
        query_id: str,
        query_embedding: np.ndarray,
        passages: list[str],
        passage_ids: list[str],
        passage_embeddings: np.ndarray,

        llm,
        llm_budget: int,
        k_cold_start: int,
        score_type: str,

        kernel: str = "rbf",
        acq_func: str = "ucb",
        alpha: float = 1e-3,
        alpha_method: Optional[str] = None,
        train_alpha: bool = False,
        length_scale: float = 1,
        beta: float = 2.0,

        use_query: Optional[int] = None,
        ard: bool = False,

        k_retrieval: int = 1000,
        batch_size: int = 1,

        offline: bool = False,
        return_score: bool = False,
        cache: dict = None,
        update_cache: str = None,
        verbose: bool = False
):
    dtype = torch.float64

    if verbose:
        print(f"\n > Query {query_id}")
    assert llm_budget > 0
    assert k_cold_start <= llm_budget

    if acq_func == "ucb":
        bandit = GPUCB(beta=beta, alpha=alpha,
                       alpha_method=alpha_method, train_alpha=train_alpha,
                       ard=ard, length_scale=length_scale, verbose=verbose)
    elif acq_func == "ei":
        bandit = GPEI(alpha=alpha, alpha_method=alpha_method, train_alpha=train_alpha,
                        ard=ard, length_scale=length_scale, verbose=verbose)
    elif acq_func == "pi":
        bandit = GPPI(alpha=alpha, alpha_method=alpha_method, train_alpha=train_alpha,
                      ard=ard, length_scale=length_scale, verbose=verbose)
    elif acq_func == "thompson":
        bandit = GPThompson(alpha=alpha, alpha_method=alpha_method, train_alpha=train_alpha,
                            ard=ard, length_scale=length_scale, verbose=verbose)
    elif acq_func == "random":
        bandit = GPRandom(alpha=alpha, alpha_method=alpha_method, train_alpha=train_alpha,
                      ard=ard, length_scale=length_scale, verbose=verbose)
    elif acq_func == "greedy":
        bandit = BaseGP(alpha=alpha, alpha_method=alpha_method, train_alpha=train_alpha,
                      ard=ard, length_scale=length_scale, verbose=verbose)
        k_cold_start = llm_budget
    else:
        raise ValueError(f"Unknown acquisition function: {acq_func}")

    query_embedding_t = torch.tensor(query_embedding, dtype=dtype)
    passage_embeddings_t = torch.tensor(passage_embeddings, dtype=dtype)

    if use_query:
        bandit.update(query_embedding_t, 3, None)

    all_indices = torch.arange(passage_embeddings_t.size(0))
    cold_scores = torch.full((len(passages),), float('-inf'), dtype=dtype)
    bandit_scores = torch.full((len(passages),), float('-inf'), dtype=dtype)

    ########## Cold Start ##########
    if k_cold_start > 0:
        sims = cosine_similarity_torch(query_embedding_t, passage_embeddings_t)
        _, cold_start_idx = torch.topk(sims, k=k_cold_start)

        if verbose:
            print(f" >> Cold start phase: selecting top-{k_cold_start} passages")

        for i in range(0, k_cold_start, batch_size):
            batch_idx = cold_start_idx[i:i + batch_size] # list of passage indices
            batch_passages = [passages[j.item()] for j in batch_idx] # list of passages
            batch_ids = [passage_ids[j.item()] for j in batch_idx] # list of passage ids

            uncached_passages = []
            uncached_ids = []

            # Check cache for cold start passages
            for j, pid, passage in zip(batch_idx, batch_ids, batch_passages):
                cached = get_cache(cache, query_id, pid)
                if cached is not None:
                    score, logit = cached[score_type], cached["logit"]
                    cold_scores[j] = float(cached[score_type])
                    bandit.update(passage_embeddings_t[j], float(score), logit)
                else:
                    uncached_passages.append(passage)
                    uncached_ids.append(pid)

            # If there are uncached passages, get their scores
            if uncached_passages:
                if offline:
                    raise ValueError("Offline mode but uncached passages found.")
                scores, logits = llm.get_score(
                    [query], uncached_passages, [query_id], uncached_ids,
                    cache=cache, update_cache=update_cache, verbose=verbose
                )

                for j, s, logit in zip(batch_idx, scores, logits):
                    cold_scores[j] = float(s)
                    bandit.update(passage_embeddings_t[j], float(s), logit)

        avail_mask = torch.ones(len(passages), dtype=torch.bool)
        avail_mask[cold_start_idx] = False
    else:
        avail_mask = torch.ones(len(passages), dtype=torch.bool)

    if verbose:
        print(f" >> Cold start scores: {cold_scores[~avail_mask].tolist()}")

    ########## Exploration-Exploitation ##########
    if verbose:
        print(" >> Exploration-exploitation phase")

    budget = llm_budget - k_cold_start
    for _ in range(0, budget, 1): # assert sequential update
        available_idx = all_indices[avail_mask]
        if available_idx.numel() == 0:
            break

        emb = passage_embeddings_t[available_idx]
        selected_local = bandit.select(emb, 1)
        selected_idx = available_idx[selected_local]

        avail_mask[selected_idx] = False
        selected_passage = [passages[j.item()] for j in selected_idx]
        selected_id = [passage_ids[j.item()] for j in selected_idx]

        if verbose:
            print(f" >>> Selected passages: {selected_id}")
        # Check cache for cold start passages
        cached = get_cache(cache, query_id, selected_id[0]) # assert sequential update
        if cached is not None:
            selected_score, selected_logit = cached[score_type], cached["logit"]
            cold_scores[selected_idx] = float(cached[score_type])
            if verbose:
                print(f" >>>> Cached score for {selected_id[0]}: {selected_score}")
            bandit.update(passage_embeddings_t[selected_idx], float(selected_score), selected_logit)
        else:
            if offline:
                raise ValueError("Offline mode but uncached passages found.")
            scores, logits = llm.get_score(
                [query], selected_passage, [query_id], selected_id,
                cache=cache, update_cache=update_cache, verbose=verbose
            )
            for j, s, logit in zip(selected_idx, scores, logits):
                bandit_scores[j] = float(s)
                bandit.update(passage_embeddings_t[j], float(s), logit)

    ########## Ranking ##########
    top_k_idx, top_k_scores = bandit.get_top_k(passage_embeddings, k_retrieval, return_scores=return_score)
    top_k_ids = [passage_ids[idx] for idx in top_k_idx]

    if return_score:
        all_scores = {
            passage_ids[i]: float(score)
            for i, score in enumerate(torch.maximum(cold_scores, bandit_scores))
            if float(score) != float('-inf')
        }
        return top_k_ids, top_k_scores.tolist(), all_scores

    return top_k_ids

def cosine_similarity_torch(query: torch.Tensor, passages: torch.Tensor) -> torch.Tensor:
    query = query / query.norm(dim=-1, keepdim=True)
    passages = passages / passages.norm(dim=-1, keepdim=True)
    return passages @ query
