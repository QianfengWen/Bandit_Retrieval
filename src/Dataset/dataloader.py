from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
import json

from src.Dataset.covid import Covid
from src.Dataset.fiqa import FiQA
from src.Dataset.scidocs import Scidocs
from src.Dataset.scifact import Scifact
from src.Dataset.touche import Touche
from src.Dataset.nfcorpus import NfCorpus
from src.Dataset.dl19 import DL19
from src.Dataset.dl20 import DL20


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
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return dataset

class Dataloader(ABC):    
    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_questions(self):
        pass

    @abstractmethod
    def load_passages(self):
        pass

    @abstractmethod
    def create_relevance_map(self):
        pass

    def load_data(self):
        self.dataset = self.load_dataset()
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        relevance_map = self.create_relevance_map()
        
        return question_ids, question_texts, passage_ids, passage_texts, relevance_map

    def visualize_relevance(self, fig_path):
        relevance = [qrel.relevance for qrel in self.dataset.qrels_iter()]
        plt.hist(relevance, bins=len(set(relevance)))
        plt.xlabel('Relevance')
        plt.ylabel('Frequency')
        plt.title('Relevance Distribution')
        plt.savefig(fig_path)
        return
    
    def save_relevance_map(self, data_dir, relevance_map):
        data_path = os.path.join(data_dir, "relevance_map.json")
        with open(data_path, 'w') as json_file:
            json.dump(relevance_map, json_file, indent=4)
        return
    



        