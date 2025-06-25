import csv
import json
from pathlib import Path

from tqdm import tqdm

from src.Dataset.dataset import IRDataset


class TravelDest(IRDataset):
    def __init__(self, cache_path=None):
        super().__init__(cache_path)
        self.data_path = (Path(__file__).parent / '..' / '..' / 'data' / 'traveldest').resolve()


    def load_queries(self):
        query_path = self.data_path / 'queries_with_ids.txt'
        question_map = dict()

        with open(query_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Queries"):
                parts = line.strip().split('\t')
                query_id, text = parts[0], parts[1]
                question_map[text] = query_id

        question_ids = list(map(str, question_map.values()))
        question_texts = list(question_map.keys())

        print(f" >>> Loaded {len(question_ids)} queries from {query_path}")

        return question_ids, question_texts

    def load_passages(self):
        passage_path = self.data_path / 'unique_passages.csv'
        passage_map = dict()

        with open(passage_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Loading Passages"):
                doc_id = row['passage_id']
                text = row['passage']
                passage_map[text] = doc_id

        passage_ids = list(passage_map.values())
        passage_texts = list(passage_map.keys())

        print(f" >>> Loaded {len(passage_ids)} passages from {passage_path}")

        return passage_ids, passage_texts


    def load_relevance_map(self):
        relevance_map_path = self.data_path / 'gt_passages.json'
        relevance_map = json.load(open(relevance_map_path, 'r', encoding='utf-8'))

        return relevance_map

    def load_data(self):
        passage_ids, passage_texts = self.load_passages()
        question_ids, question_texts = self.load_queries()
        relevance_map = self.load_relevance_map()

        return question_ids, question_texts, passage_ids, passage_texts, relevance_map





