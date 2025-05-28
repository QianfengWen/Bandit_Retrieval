from src.Dataset.dataset import IRDataset


class Scifact(IRDataset):
    def __init__(self, data_name="beir/scifact/test", cache_path=None):
        super().__init__(data_name, cache_path)
