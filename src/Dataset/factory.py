from src.Dataset.antique import Antique
from src.Dataset.covid import Covid
from src.Dataset.dbpedia import DBPedia
from src.Dataset.dl19 import DL19
from src.Dataset.dl20 import DL20
from src.Dataset.fiqa import FiQA
from src.Dataset.news import News
from src.Dataset.nfcorpus import NfCorpus
from src.Dataset.robust import Robust04
from src.Dataset.scidocs import Scidocs
from src.Dataset.scifact import Scifact
from src.Dataset.touche import Touche
from src.Dataset.traveldest import TravelDest


def handle_dataset(dataset_name, cache_path=None):
    if dataset_name == "covid":
        dataset = Covid(cache_path=cache_path)
    elif dataset_name == "touche":
        dataset = Touche(cache_path=cache_path)
    elif dataset_name == 'dl19':
        dataset = DL19(cache_path=cache_path)
    elif dataset_name == 'dl20':
        dataset = DL20(cache_path=cache_path)
    elif dataset_name == 'nfcorpus':
        dataset = NfCorpus(cache_path=cache_path)
    elif dataset_name == 'scidocs':
        dataset = Scidocs(cache_path=cache_path)
    elif dataset_name == 'scifact':
        dataset = Scifact(cache_path=cache_path)
    elif dataset_name == 'fiqa':
        dataset = FiQA(cache_path=cache_path)
    elif dataset_name == "dbpedia":
        dataset = DBPedia(cache_path=cache_path)
    elif dataset_name == "antique":
        dataset = Antique(cache_path=cache_path)
    elif dataset_name == "news":
        dataset = News(cache_path=cache_path)
    elif dataset_name == "traveldest":
        dataset = TravelDest(cache_path=cache_path)
    elif dataset_name == "robust":
        dataset = Robust04(cache_path=cache_path)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return dataset
