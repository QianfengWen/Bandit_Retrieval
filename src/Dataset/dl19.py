from src.Dataset.ir_dataset import IRDataset


class Covid(IRDataset):
    def __init__(self, data_name="msmarco-document/trec-dl-2019/judged", cache_path=None):
        super().__init__(data_name, cache_path)
        self.relevance_threshold = 2

