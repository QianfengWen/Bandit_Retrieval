from src.Dataset.dataset import IRDataset

class FiQA(IRDataset):
    def __init__(self, data_name="beir/fiqa/test", cache_path=None):
        super().__init__(data_name, cache_path)
