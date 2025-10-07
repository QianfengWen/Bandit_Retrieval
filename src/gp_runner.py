from pathlib import Path
from collections import defaultdict
import json

import numpy as np
from tqdm import tqdm

from src.Retrieval.retrieval import gp_retrieval
from src.LLM.openrouter_llm import OpenRouterLLM
from src.Dataset.dataloader import handle_dataset
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score_gp, eval_rec, save_results


def main(
    dataset_name="travel_dest",
    top_k_passages=3,
    fusion_mode="mean",
    llm_model_name="openai/gpt-4o",
    embedder_name="all-MiniLM-L6-v2",
    kernel="rbf",
    llm_budget=100,
    sample_strategy="random",
    epsilon=0.1,
    batch_size=1,
    random_seed=42,
    normalize_y=True,
    alpha=1e-5,
    length_scale=1.0,
    tau=None,
    scoring_mode="expected_relevance",
):
    """Run the GP + LLM retrieval pipeline and store evaluation metrics."""
    dataset = handle_dataset(dataset_name)

    query_embeddings_path = f"data/{dataset_name}/{embedder_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{embedder_name}_passage_embeddings.pkl"

    (
        question_ids,
        queries,
        passage_ids,
        passages,
        relevance_map,
        passage_dict,
        _,
        _,
    ) = dataset.load_data()

    query_embeddings, passage_embeddings = handle_embeddings(
        embedder_name,
        query_embeddings_path,
        passage_embeddings_path,
        queries,
        passages,
    )

    output_root = Path("output") / "gpr_llm" / dataset_name / embedder_name
    evaluation_dir = output_root / "evaluation_results"
    retrieval_dir = output_root / "retrieval_results"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    llm = OpenRouterLLM(model_name=llm_model_name, score_mode=scoring_mode)
    cache = dataset.load_cache()
    update_cache = f"data/{dataset_name}/cache.csv"

    tau_value = tau if tau is not None else len(passage_ids)

    k_values = range(10, 51, 10)

    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)
    ndcg_k_dict = defaultdict(list)
    retrieval_cities = {}

    print("=== GP Retrieval with LLM Feedback ===")
    for idx, (query, query_id) in enumerate(
        tqdm(zip(queries, question_ids), total=len(queries), desc="Query")
    ):
        if idx % 10 == 0:
            print(f"Processing Query {idx + 1}/{len(queries)}...")

        gp = gp_retrieval(
            query=query,
            query_embedding=query_embeddings[idx],
            query_id=query_id,
            passage_ids=passage_ids.copy(),
            passage_embeddings=passage_embeddings,
            passages=passages,
            passage_dict=passage_dict,
            kernel=kernel,
            llm=llm,
            llm_budget=llm_budget,
            epsilon=epsilon,
            sample_strategy=sample_strategy,
            batch_size=batch_size,
            cache=cache,
            update_cache=update_cache,
            verbose=False,
            random_seed=random_seed,
            normalize_y=normalize_y,
            alpha=alpha,
            length_scale=length_scale,
            tau=tau_value,
        )

        bandit_cities = fusion_score_gp(
            gp=gp,
            passage_ids=passage_ids,
            passage_dict=passage_dict,
            passage_embeddings=passage_embeddings,
            top_k_passages=top_k_passages,
            k_retrieval=len(passage_ids),
            return_scores=False,
            fusion_method=fusion_mode,
        )
        retrieval_cities[query_id] = bandit_cities

        for k in k_values:
            prec_k, rec_k, map_k, ndcg_k = eval_rec(
                bandit_cities,
                list(relevance_map[query_id].keys()),
                k,
            )
            prec_k_dict[k].append(prec_k)
            rec_k_dict[k].append(rec_k)
            map_k_dict[k].append(map_k)
            ndcg_k_dict[k].append(ndcg_k)

    results = {}
    for k in k_values:
        mean_prec_k = np.mean(prec_k_dict[k]).round(4)
        mean_rec_k = np.mean(rec_k_dict[k]).round(4)
        mean_map_k = np.mean(map_k_dict[k]).round(4)
        mean_ndcg_k = np.mean(ndcg_k_dict[k]).round(4)

        print(f"Precision@{k}: {mean_prec_k}")
        print(f"Recall@{k}: {mean_rec_k}")
        print(f"MAP@{k}: {mean_map_k}")
        print(f"NDCG@{k}: {mean_ndcg_k}")

        results[f"precision@{k}"] = mean_prec_k
        results[f"recall@{k}"] = mean_rec_k
        results[f"map@{k}"] = mean_map_k
        results[f"ndcg@{k}"] = mean_ndcg_k

    configs = {
        "dataset": dataset_name,
        "embedder_name": embedder_name,
        "kernel": kernel,
        "llm_model_name": llm_model_name,
        "llm_budget": llm_budget,
        "scoring_mode": scoring_mode,
        "sample_strategy": sample_strategy,
        "epsilon": epsilon,
        "top_k_passages": top_k_passages,
        "fusion_mode": fusion_mode,
        "retrieval_cutoff": len(passage_ids),
        "batch_size": batch_size,
        "random_seed": random_seed,
        "normalize_y": normalize_y,
        "alpha": alpha,
        "length_scale": length_scale,
        "tau": tau_value,
    }

    config_str = "_".join([f"{k}={v}" for k, v in configs.items()])
    evaluation_path = evaluation_dir / f"{config_str}.csv"
    retrieval_results_path = retrieval_dir / f"{config_str}.json"

    with open(retrieval_results_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_cities, f, indent=4)

    assert (
        save_results(configs, results, str(evaluation_path)) is True
    ), "Results not saved"
    print(f"Results saved to {evaluation_path}")


if __name__ == "__main__":
    main()
