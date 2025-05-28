from src.Dataset.dataset import IRDataset

class Antique(IRDataset):
    def __init__(self, data_name="antique/test/non-offensive", cache_path=None):
        super().__init__(data_name, cache_path)
        self.relevance_threshold = 2

