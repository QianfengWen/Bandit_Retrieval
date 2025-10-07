from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .Dataset.dataloader import handle_dataset
from .Retrieval.retrieval import dense_retrieval
from .Embedding.embedding import handle_embeddings
from .RecUtils.rec_utils import fusion_score, eval_rec, save_results


def main(
    dataset_name="travel_dest",
    top_k_passages=3,
    fusion_mode="mean",
    embedder_name="all-MiniLM-L6-v2",
):
    """Run the dense retrieval baseline and persist evaluation metrics."""
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
        _,
    ) = dataset.load_data()

    query_embeddings, passage_embeddings = handle_embeddings(
        embedder_name,
        query_embeddings_path,
        passage_embeddings_path,
        queries,
        passages,
    )

    k_values = range(10, 51, 10)
    fusion_mode_internal = "average" if fusion_mode == "mean" else fusion_mode

    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)
    ndcg_k_dict = defaultdict(list)

    print("=== Dense Retrieval Baseline ===")
    for q_id, query_embedding in tqdm(
        zip(question_ids, query_embeddings),
        desc="Query",
        total=len(question_ids),
    ):
        items, scores = dense_retrieval(
            passage_ids,
            passage_embeddings,
            query_embedding,
            k_retrieval=len(passage_ids),
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

    output_dir = Path("output") / "baseline" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{embedder_name}_baseline_results.csv"

    configs = {
        "dataset": dataset_name,
        "embedder_name": embedder_name,
        "top_k_passages": top_k_passages,
        "fusion_mode": fusion_mode,
        "retrieval_cutoff": len(passage_ids),
    }

    assert (
        save_results(configs, results, str(result_path)) is True
    ), "Results not saved"
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
