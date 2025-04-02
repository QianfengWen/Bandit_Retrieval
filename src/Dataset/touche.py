from src.Dataset.ir_dataset import IRDataset


class Touche(IRDataset):
    def __init__(self, data_name="beir/webis-touche2020/v2", cache_path="src/data/touche/cache.csv"):
        super().__init__(data_name, cache_path)
        self.relevance_threshold = 2
