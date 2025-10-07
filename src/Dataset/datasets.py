from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd

from .dataloader import Dataloader


class CorpusDataset(Dataloader):
    """Generic corpus-backed dataset.

    Expects the following layout under ``data/<slug>``::

        corpus/                # One ``.txt`` file per city
        queries.txt            # Newline separated queries
        ground_truth.json      # Mapping query -> list of relevant cities
        cache.csv              # Optional cached LLM scores
    """

    def __init__(self, slug: str, encoding: str = "utf-8", errors: str = "strict"):
        self.slug = slug
        self.encoding = encoding
        self.errors = errors
        self.cache_filename = "cache.csv"

    @property
    def root(self) -> Path:
        return Path("data") / self.slug

    def _open(self, path: Path):
        return path.open("r", encoding=self.encoding, errors=self.errors)

    def load_dataset(self, cache_path: str | None = None):
        self.cache_path = cache_path or str(self.root / self.cache_filename)

        passages: List[str] = []
        passage_dict: Dict[int, List[int]] = {}
        cities: List[str] = []
        passage_id = 0

        corpus_dir = self.root / "corpus"
        for city_idx, file in enumerate(sorted(corpus_dir.glob("*.txt"))):
            city_name = unicodedata.normalize("NFC", file.stem)
            cities.append(city_name)
            passage_dict[city_idx] = []

            with self._open(file) as fh:
                for passage in fh.read().strip().split("\n"):
                    passages.append(passage)
                    passage_dict[city_idx].append(passage_id)
                    passage_id += 1

        with self._open(self.root / "queries.txt") as fh:
            queries = fh.read().strip().split("\n")

        with self._open(self.root / "ground_truth.json") as fh:
            qrels_iter = json.load(fh)

        prelabel_relevance = self.load_cache()
        return queries, passages, qrels_iter, passage_dict, cities, prelabel_relevance

    def load_questions(self):
        question_map = {
            query: query_id + len(self.passages) + 100
            for query_id, query in enumerate(self.queries)
        }
        question_ids = list(question_map.values())
        question_texts = list(question_map.keys())
        self.question_sub = {
            qid: question_map[query]
            for query, qid in question_map.items()
        }
        return question_ids, question_texts

    @staticmethod
    def load_passages_from_list(passages: List[str]):
        passage_map = {idx: passage for idx, passage in enumerate(passages)}
        passage_ids = list(passage_map.keys())
        passage_texts = list(passage_map.values())
        return passage_ids, passage_texts

    def load_passages(self):
        return self.load_passages_from_list(self.passages)

    def create_relevance_map(self, question_texts):
        query_to_id = {q: idx for idx, q in enumerate(question_texts)}
        relevance_map: Dict[int, Dict[int, float]] = defaultdict(dict)

        for query, relevant_doc_names in self.qrels_iter.items():
            idx = query_to_id.get(query)
            if idx is None:
                continue
            query_id = idx + len(self.passages) + 100
            for doc_name in relevant_doc_names:
                if doc_name in self.cities:
                    doc_id = self.cities.index(doc_name)
                    relevance_map[query_id][doc_id] = 1.0
        return relevance_map

    @staticmethod
    def create_passage_ids_to_city_map(passage_dict):
        return {
            passage_id: city_id
            for city_id, passage_ids in passage_dict.items()
            for passage_id in passage_ids
        }

    def load_cache(self):
        try:
            df = pd.read_csv(self.cache_path, skipinitialspace=True)
        except FileNotFoundError:
            return defaultdict(dict)

        prelabel_relevance: Dict[int, Dict[int, float]] = defaultdict(dict)
        for query_id, doc_id, score in zip(df["query_id"], df["passage_id"], df["score"]):
            prelabel_relevance[int(query_id)][int(doc_id)] = float(score)
        return prelabel_relevance

    def load_data(self):
        (
            self.queries,
            self.passages,
            self.qrels_iter,
            self.passage_dict,
            self.cities,
            prelabel_relevance,
        ) = self.load_dataset()

        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()

        relevance_map = self.create_relevance_map(question_texts)
        passage_city_map = self.create_passage_ids_to_city_map(self.passage_dict)

        return (
            question_ids,
            question_texts,
            passage_ids,
            passage_texts,
            relevance_map,
            self.passage_dict,
            passage_city_map,
            prelabel_relevance,
        )


class TravelDestDataset(CorpusDataset):
    def __init__(self):
        super().__init__("travel_dest")


class RestaurantPhiDataset(CorpusDataset):
    def __init__(self):
        super().__init__("restaurant_phi")


class RestaurantNorDataset(CorpusDataset):
    def __init__(self):
        super().__init__("restaurant_nor")


class HotelChicagoDataset(CorpusDataset):
    def __init__(self):
        super().__init__("hotel_chicago", encoding="cp1252", errors="ignore")


class HotelLondonDataset(CorpusDataset):
    def __init__(self):
        super().__init__("hotel_london", encoding="cp1252", errors="ignore")


class HotelMontrealDataset(CorpusDataset):
    def __init__(self):
        super().__init__("hotel_montreal", encoding="cp1252", errors="ignore")


class HotelNYCDataset(CorpusDataset):
    def __init__(self):
        super().__init__("hotel_nyc", encoding="cp1252", errors="ignore")


class PointRecUSDataset(Dataloader):
    """Dataset loader for the POI-style ``point_rec_us`` corpus."""

    def __init__(self):
        self.root = Path("data") / "point_rec_us"
        self.cache_path = self.root / "cache.csv"

    def load_dataset(self):
        passages: List[str] = []
        passage_dict: Dict[int, List[int]] = {}
        cities: List[str] = []
        pois: List[str] = []

        passage_id = 0
        poi_idx = 0

        for file in sorted((self.root / "corpus").glob("*.json")):
            city_name = file.stem
            cities.append(city_name)
            with file.open("r", encoding="utf-8") as fh:
                poi_data = json.load(fh)
            for poi_id, poi in poi_data.items():
                snippets = poi.get("snippets") or []
                if not snippets:
                    continue
                passage_dict[poi_idx] = []
                for snippet in snippets:
                    passages.append(snippet["snippet"])
                    passage_dict[poi_idx].append(passage_id)
                    passage_id += 1
                pois.append(poi_id)
                poi_idx += 1

        with (self.root / "infoneeds.json").open("r", encoding="utf-8") as fh:
            self.query_metadata = json.load(fh)

        queries: Dict[str, str] = {}
        for query_id, meta in self.query_metadata.items():
            if "US" in meta.get("Country", []) and meta.get("Description") not in queries.values():
                queries[query_id] = meta["Description"]

        qrels_iter = []
        with (self.root / "relevance" / "qrels.trec").open("r", encoding="utf-8") as fh:
            for line in fh:
                query_id, _, poi_id, relevance = line.strip().split(" ")
                description = self.query_metadata[query_id]["Description"]
                qrels_iter.append((query_id, description, poi_id, relevance))

        prelabel_relevance = self.load_cache()
        return queries, passages, qrels_iter, passage_dict, cities, pois, prelabel_relevance

    def load_questions(self):
        question_map = {
            query: query_idx + len(self.passages) + 100
            for query_idx, query in enumerate(self.queries.values())
        }
        question_ids = list(question_map.values())
        question_texts = list(question_map.keys())
        self.question_sub = {
            qid: question_map[query]
            for query, qid in question_map.items()
        }
        return question_ids, question_texts

    def load_passages(self):
        return CorpusDataset.load_passages_from_list(self.passages)

    def create_relevance_map(self, question_texts):
        query_to_id = {q: idx for idx, q in enumerate(question_texts)}
        relevance_map: Dict[int, Dict[int, float]] = defaultdict(dict)

        for idx in query_to_id.values():
            relevance_map[idx + len(self.passages) + 100] = {}

        for raw_qid, query, poi_id, relevance in self.qrels_iter:
            if query not in query_to_id:
                continue
            if int(relevance) < 2:
                continue
            if poi_id not in self.pois:
                continue
            query_id = query_to_id[query] + len(self.passages) + 100
            relevance_map[query_id][self.pois.index(poi_id)] = 1.0

        return dict(sorted(relevance_map.items()))

    @staticmethod
    def create_passage_ids_to_city_map(passage_dict):
        return CorpusDataset.create_passage_ids_to_city_map(passage_dict)

    def load_cache(self):
        try:
            df = pd.read_csv(self.cache_path, skipinitialspace=True)
        except FileNotFoundError:
            return defaultdict(dict)

        prelabel_relevance: Dict[int, Dict[int, float]] = defaultdict(dict)
        for query_id, doc_id, score in zip(df["query_id"], df["passage_id"], df["score"]):
            prelabel_relevance[int(query_id)][int(doc_id)] = float(score)
        return prelabel_relevance

    def load_data(self):
        (
            self.queries,
            self.passages,
            self.qrels_iter,
            self.passage_dict,
            self.cities,
            self.pois,
            prelabel_relevance,
        ) = self.load_dataset()

        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()

        relevance_map = self.create_relevance_map(question_texts)
        passage_city_map = self.create_passage_ids_to_city_map(self.passage_dict)

        return (
            question_ids,
            question_texts,
            passage_ids,
            passage_texts,
            relevance_map,
            self.passage_dict,
            passage_city_map,
            prelabel_relevance,
        )
