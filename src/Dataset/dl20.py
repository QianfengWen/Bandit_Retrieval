from src.Dataset.dataset import IRDataset


class DL20(IRDataset):
    def __init__(self, data_name="msmarco-document/trec-dl-2020/judged", cache_path=None):
        super().__init__(data_name, cache_path)
        self.relevance_threshold = 2

    def load_passages(self):
        passage_map = dict()
        sub = dict()

        for passage in self.dataset.docs_iter():
            if passage.body in passage_map:
                pass
            else:
                passage_map[passage.body] = passage.doc_id

        self.passage_sub = sub
        passage_ids = list(passage_map.values())
        passage_texts = list(passage_map.keys())

        return passage_ids, passage_texts