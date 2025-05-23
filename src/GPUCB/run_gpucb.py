import random
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import normalize

from src.GPUCB.gpucb import GPUCB
from src.LLM.llm import LLM
from src.utils import cosine_similarity


def bandit_retrieval(
        passage_ids,
        passage_embeddings,
        passages,
        llm: LLM,
        query,
        query_embedding,
        query_id,
        llm_budget,
        k_cold_start,
        use_query=None,
        alpha=1e-3,
        beta=2.0,
        length_scale=1,
        nu=2.5,
        acq_func="ucb",
        normalize_passage=False,
        ard=False,
        k_retrieval: int = 1000,
        kernel: str = "rbf",
        batch_size: int = 1,
        verbose: bool = False,
        return_score: bool = False,
        cache: dict = None,
        update_cache: str = None
):
    ############### Set up ################
    if llm_budget < 1:
        raise ValueError("LLM budget must be at least 1")

    k_cold_start = min(k_cold_start, llm_budget)
    if ard:
        length_scale = [length_scale] * passage_embeddings.shape[1]

    gpucb = GPUCB(beta=beta, kernel=kernel, alpha=alpha, length_scale=length_scale, acquisition_function=acq_func,
                  nu=nu)  # set up GP-UCB

    if normalize_passage:
        org_passage_embeddings = passage_embeddings.copy()
        passage_embeddings = normalize(passage_embeddings, norm='l2')
    else:
        org_passage_embeddings = passage_embeddings

    if use_query is not None:
        raise NotImplementedError("use_query with normalization is not implemented yet")
        gpucb.update(query_embedding, use_query)

    available_ids = passage_ids.copy()
    id_to_embedding = {pid: emb for pid, emb in zip(passage_ids, passage_embeddings)}

    observed_ids = []
    scores = {}

    ############### Cold start ################
    cosine_similairty_matrix = cosine_similarity(query_embedding, org_passage_embeddings)
    if query_embedding is not None and k_cold_start > 0:
        cold_start_idx = np.argsort(cosine_similairty_matrix)[::-1][:k_cold_start]
        cold_start_ids = [passage_ids[idx] for idx in cold_start_idx]

        random.shuffle(cold_start_ids)
        if verbose:
            print("There are ", len(cold_start_ids), " cold start ids")
            print("Cold start ids: ", cold_start_ids)
        cold_start_batches = [cold_start_ids[i:i + batch_size] for i in range(0, len(cold_start_ids), batch_size)]

        for batch in cold_start_batches:
            # remove batch items from available IDs
            for target_id in batch:
                available_ids.remove(target_id)

            # get passages corresponding to batch
            passage_idxs = [passage_ids.index(target_id) for target_id in batch]
            batch_passages = [passages[idx] for idx in passage_idxs]

            # get relevance scores for the batch using LLM
            batch_scores, _ = llm.get_score(query, batch_passages, query_ids=[query_id], passage_ids=batch, cache=cache,
                                         update_cache=update_cache)

            # store observations
            for target_id, score in zip(batch, batch_scores):
                scores[target_id] = score
                observed_ids.append(target_id)

                # update GP-UCB model using the passage embedding as feature
                gpucb.update(id_to_embedding[target_id], score)

            if verbose:  # debug print
                print("batch_scores: ", batch_scores)

                for score, passage in zip(batch_scores, batch_passages):
                    try:
                        print(f"Score: {score}, Passage: {passage}")
                    except UnicodeEncodeError:
                        print(f"Score: {score}, Passage: {passage.encode('ascii', 'replace').decode()}")

    ############### Exploration-exploitation ################
    num_iterations = (llm_budget - k_cold_start) // batch_size

    # use GP-UCB to select passages in batches
    for _ in tqdm(range(num_iterations), desc="Bandit Retrieval", disable=not verbose):
        if not available_ids:
            break

        available_embeddings = [id_to_embedding[pid] for pid in available_ids]

        next_embedding_idxs = gpucb.select(available_embeddings, batch_size)
        next_ids = [available_ids[idx] for idx in next_embedding_idxs]
        if verbose:
            print("there are ", len(available_ids), " available embeddings")
            print("Next IDs: ", next_ids)

        for next_id in next_ids:
            available_ids.remove(next_id)

        passage_idxs = [passage_ids.index(next_id) for next_id in next_ids]
        batch_passages = [passages[idx] for idx in passage_idxs]

        batch_scores, _ = llm.get_score(query, batch_passages, query_ids=[query_id], passage_ids=next_ids, cache=cache,
                                     update_cache=update_cache)

        for next_id, score in zip(next_ids, batch_scores):
            scores[next_id] = score
            observed_ids.append(next_id)
            gpucb.update(id_to_embedding[next_id], score)  # Update model

        if verbose:  # debug print
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
