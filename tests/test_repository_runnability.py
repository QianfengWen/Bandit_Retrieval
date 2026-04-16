import json
import zipfile


def test_dataset_package_imports_without_optional_missing_modules():
    import src.Dataset as dataset_package

    assert hasattr(dataset_package, "TravelDestDataset")
    assert hasattr(dataset_package, "PointRecUSDataset")


def test_handle_dataset_returns_known_dataset_and_rejects_unknown():
    import pytest

    from src.Dataset.dataloader import handle_dataset
    from src.Dataset.datasets import TravelDestDataset

    assert isinstance(handle_dataset("travel_dest"), TravelDestDataset)
    with pytest.raises(ValueError, match="Unknown dataset"):
        handle_dataset("missing_dataset")


def test_corpus_dataset_loads_from_bundled_data_zip_when_data_dir_is_missing(tmp_path, monkeypatch):
    dataset_root = "data/travel_dest"
    archive_path = tmp_path / "data.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{dataset_root}/corpus/City A.txt", "first passage\nsecond passage\n")
        archive.writestr(f"{dataset_root}/queries.txt", "find city a\n")
        archive.writestr(
            f"{dataset_root}/ground_truth.json",
            json.dumps({"find city a": ["City A"]}),
        )

    monkeypatch.chdir(tmp_path)

    from src.Dataset.datasets import TravelDestDataset

    (
        question_ids,
        queries,
        passage_ids,
        passages,
        relevance_map,
        passage_dict,
        passage_city_map,
        prelabel_relevance,
    ) = TravelDestDataset().load_data()

    assert queries == ["find city a"]
    assert passages == ["first passage", "second passage"]
    assert passage_ids == [0, 1]
    assert passage_dict == {0: [0, 1]}
    assert passage_city_map == {0: 0, 1: 0}
    assert relevance_map[question_ids[0]] == {0: 1.0}
    assert prelabel_relevance == {}


def test_hashing_embedder_creates_and_reloads_local_embeddings(tmp_path):
    import numpy as np

    from src.Embedding.embedding import handle_embeddings

    query_path = tmp_path / "hashing_query_embeddings.pkl"
    passage_path = tmp_path / "hashing_passage_embeddings.pkl"

    query_embeddings, passage_embeddings = handle_embeddings(
        "hashing",
        str(query_path),
        str(passage_path),
        ["quiet beach"],
        ["quiet beach with sand", "busy museum"],
    )

    assert query_embeddings.shape == (1, 384)
    assert passage_embeddings.shape == (2, 384)
    assert query_embeddings.dtype == np.float32
    assert passage_embeddings.dtype == np.float32
    assert query_path.exists()
    assert passage_path.exists()

    reloaded_query_embeddings, reloaded_passage_embeddings = handle_embeddings(
        "hashing",
        str(query_path),
        str(passage_path),
        ["different query text should not be used"],
        ["different passage text should not be used"],
    )

    np.testing.assert_array_equal(reloaded_query_embeddings, query_embeddings)
    np.testing.assert_array_equal(reloaded_passage_embeddings, passage_embeddings)


def test_default_runners_use_local_hashing_embedder():
    import inspect

    from src import baseline_runner, gp_runner, llm_relevance_scorer_runner

    assert inspect.signature(baseline_runner.main).parameters["embedder_name"].default == "hashing"
    assert inspect.signature(gp_runner.main).parameters["embedder_name"].default == "hashing"
    assert (
        inspect.signature(llm_relevance_scorer_runner.main)
        .parameters["embedder_name"]
        .default
        == "hashing"
    )


def test_cosine_similarity_handles_zero_vectors_without_nan():
    import numpy as np

    from src.Retrieval.retrieval import calculate_cosine_similarity

    scores = calculate_cosine_similarity(
        np.array([1.0, 0.0]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    np.testing.assert_allclose(scores, np.array([0.0, 1.0]))
    assert np.isfinite(scores).all()


def test_openrouter_llm_can_use_partial_cache_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from src.LLM.openrouter_llm import OpenRouterLLM

    llm = OpenRouterLLM(require_api_key=False)
    scores = llm.get_score(
        query="quiet beach",
        passages=["cached passage", "uncached passage"],
        query_id=123,
        passage_ids=[10, 11],
        cache={123: {10: 2.5}},
    )

    assert scores == [2.5, -1.0]


def test_fusion_score_gp_accepts_numpy_score_arrays():
    import numpy as np

    from src.RecUtils.rec_utils import fusion_score_gp

    class FakeGP:
        def get_top_k(self, candidates, k, return_scores=False):
            scores = np.linspace(1.0, 0.5, len(candidates))
            indices = np.arange(len(candidates))[:k]
            if return_scores:
                return indices, scores[:k]
            return indices

    ranked_cities = fusion_score_gp(
        gp=FakeGP(),
        passage_ids=[0, 1],
        passage_dict={0: [0], 1: [1]},
        passage_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        top_k_passages=1,
        k_retrieval=2,
        return_scores=False,
        fusion_method="mean",
    )

    assert ranked_cities == [0, 1]


def test_gp_runner_config_filename_sanitizes_path_separators():
    from src.gp_runner import _format_config_filename

    filename = _format_config_filename(
        {
            "dataset": "travel_dest",
            "llm_model_name": "openai/gpt-4o",
            "alpha": 1e-5,
        }
    )

    assert filename == "dataset=travel_dest_llm_model_name=openai-gpt-4o_alpha=1e-05"
    assert "/" not in filename


def test_gp_runner_config_filename_is_bounded():
    from src.gp_runner import _format_config_filename

    filename = _format_config_filename({"very_long": "x" * 400})

    assert len(filename) <= 180
    assert filename.startswith("very_long=")
