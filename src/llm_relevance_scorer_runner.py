from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.Dataset.dataloader import handle_dataset
from src.Retrieval.retrieval import llm_rerank
from src.Embedding.embedding import handle_embeddings
from src.RecUtils.rec_utils import fusion_score, eval_rec, save_results
from src.LLM.openrouter_llm import OpenRouterLLM


def main(
    dataset_name="travel_dest",
    top_k_passages=3,
    fusion_mode="mean",
    budget=100,
    embedder_name="all-MiniLM-L6-v2",
    llm_model_name="openai/gpt-4o",
    scoring_mode="expected_relevance",
    cross_encoder_reranking=False,
):
    """Run the LLM-based relevance scoring pipeline and store aggregated metrics."""
    dataset = handle_dataset(dataset_name)

    query_embeddings_path = f"data/{dataset_name}/{embedder_name}_query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset_name}/{embedder_name}_passage_embeddings.pkl"

    (
        question_ids,
        queries,
        passage_ids,
        passages,
        relevance_map,
        _,
        passage_city_map,
        cache,
    ) = dataset.load_data()

    query_embeddings, passage_embeddings = handle_embeddings(
        embedder_name,
        query_embeddings_path,
        passage_embeddings_path,
        queries,
        passages,
    )

    llm = OpenRouterLLM(model_name=llm_model_name, score_mode=scoring_mode)
    cache_path = f"data/{dataset_name}/cache.csv"

    k_values = range(10, 51, 10)
    fusion_mode_internal = "average" if fusion_mode == "mean" else fusion_mode

    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)
    ndcg_k_dict = defaultdict(list)

    if cross_encoder_reranking:
        print("Cross-encoder reranking is no longer supported and will be ignored.")

    print("=== LLM Relevance Scoring ===")
    for q_id, query_text, query_embedding in tqdm(
        zip(question_ids, queries, query_embeddings),
        desc="Query",
        total=len(question_ids),
    ):
        items, scores = llm_rerank(
            passage_ids=passage_ids,
            passage_embeddings=passage_embeddings,
            passages_text=passages,
            query_embedding=query_embedding,
            query_id=q_id,
            query_text=query_text,
            llm=llm,
            k_retrieval=budget,
            cache=cache,
            update_cache=cache_path,
            return_score=True,
        )

        bandit_cities = fusion_score(
            items,
            scores,
            passage_city_map,
            top_k_passages=top_k_passages,
            return_scores=False,
            fusion_mode=fusion_mode_internal,
        )

        for k in k_values:
            prec_k, rec_k, map_k, ndcg_k = eval_rec(
                bandit_cities,
                list(relevance_map[q_id].keys()),
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

    output_dir = Path("output") / "llm_score" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{embedder_name}_llm_relevance_results.csv"

    configs = {
        "dataset": dataset_name,
        "embedder_name": embedder_name,
        "llm_model_name": llm_model_name,
        "scoring_mode": scoring_mode,
        "budget": budget,
        "top_k_passages": top_k_passages,
        "fusion_mode": fusion_mode,
        "realtime_labeling": True,
    }

    assert (
        save_results(configs, results, str(result_path)) is True
    ), "Results not saved."
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
