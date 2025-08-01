import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.Dataset.dataset import IRDataset


class Signal(IRDataset):
    def __init__(self, cache_path=None):
        super().__init__(cache_path=cache_path)
        self.data_path = (Path(__file__).parent / '..' / '..' / 'data' / 'signal').resolve()

    def load_queries(self):
        query_path = self.data_path / 'queries.jsonl'
        question_map = dict()

        with open(query_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Queries" ):
                data = json.loads(line.strip())
                query_id = data['_id']
                query_text = data['text']

                question_map[query_text] = query_id

        query_ids = list(map(str, question_map.values()))
        query_texts = list(question_map.keys())

        return query_ids, query_texts

    def load_passages(self):
        passage_path = self.data_path / 'corpus.jsonl'
        passage_map = dict()

        with open(passage_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Passages"):
                data = json.loads(line.strip())
                doc_id = data['_id']
                text = data['text']

                passage_map[text] = doc_id

        passage_ids = list(map(str, passage_map.values()))
        passage_texts = list(passage_map.keys())

        return passage_ids, passage_texts

    def create_relevance_map(self):
        qrel_path = self.data_path / 'test.tsv'
        relevance_map = defaultdict(dict)

        with open(qrel_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in tqdm(f, desc="Loading Qrels"):
                parts = line.strip().split('\t')
                query_id = parts[0]
                doc_id = parts[1]
                relevance_score = int(parts[2])

                relevance_map[query_id][doc_id] = relevance_score

        return relevance_map

    def load_data(self):
        passage_ids, passage_texts = self.load_passages()
        question_ids, question_texts = self.load_queries()
        relevance_map = self.create_relevance_map()

        print(f" >>> Loaded {len(passage_ids)} passages and {len(question_ids)} queries")

        return question_ids, question_texts, passage_ids, passage_texts, relevance_map