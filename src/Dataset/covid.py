from src.Dataset.ir_dataset import IRDataset


class Covid(IRDataset):
    def __init__(self, data_name="beir/trec-covid", cache_path="src/data/covid/cache.csv"):
        super().__init__(data_name, cache_path)
        self.relevance_threshold = 2

