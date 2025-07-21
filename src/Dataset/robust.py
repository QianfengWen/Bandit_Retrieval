from tqdm import tqdm

from src.Dataset.dataset import IRDataset


class Robust04(IRDataset):
    def __init__(self, data_name="disks45/nocr/trec-robust-2004", cache_path=None):
        super().__init__(data_name, cache_path)

    def load_queries(self):
        question_map = dict()
        sub = dict()

        for query in tqdm(self.dataset.queries_iter(), desc="Loading Queries", total=self.dataset.queries_count()):
            if query.description in question_map:
                sub[query.query_id] = question_map[query.description]
            else:
                question_map[query.description] = query.query_id

        self.question_sub = sub
        question_ids = list(map(str, question_map.values()))
        question_texts = list(question_map.keys())

        return question_ids, question_texts

    def load_passages(self):
        passage_map = dict()
        sub = dict()

        for passage in tqdm(self.dataset.docs_iter(), desc="Loading Passages", total=self.dataset.docs_count()):
            if passage.body in passage_map:
                sub[passage.doc_id] = passage_map[passage.body]
            else:
                passage_map[passage.body] = passage.doc_id

        self.passage_sub = sub
        passage_ids = list(map(str, passage_map.values()))
        passage_texts = list(passage_map.keys())

        return passage_ids, passage_texts