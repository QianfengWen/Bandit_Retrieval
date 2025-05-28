from src.Dataset.dataset import IRDataset


class Touche(IRDataset):
    def __init__(self, data_name="beir/webis-touche2020/v2", cache_path=None):
        super().__init__(data_name, cache_path)