from src.Dataset.dataset import IRDataset

class DBPedia(IRDataset):
    def __init__(self, data_name="beir/dbpedia-entity/test", cache_path=None):
        super().__init__(data_name, cache_path)

