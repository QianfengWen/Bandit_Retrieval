from src.Dataset.dataset import IRDataset


class NfCorpus(IRDataset):
    def __init__(self, data_name="beir/nfcorpus/test", cache_path=None):
        super().__init__(data_name, cache_path)
        self.relevance_threshold = 2

