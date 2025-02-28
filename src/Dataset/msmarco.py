from collections import defaultdict
from datasets import load_dataset
from src.Dataset.dataloader import Dataloader

class MSMARCO(Dataloader):

    def load_dataset(self):
        dataset = load_dataset("microsoft/ms_marco", 'v1.1', cache_dir="data", split="test")

        dataset = dataset.shuffle(seed=42).select(range(3000))
        # dataset = dataset.shuffle(seed=42)
        return dataset
    
    def load_questions(self):
        return list(self.dataset["query_id"]), list(self.dataset["query"])
    
    def load_passages(self):
        passage_ids = []
        passage_texts = []
        for example in self.dataset:
            question_id = example["query_id"]
            for i, passage_text in enumerate(example["passages"]["passage_text"]):
                passage_id = f"{question_id}-{i}"
                passage_ids.append(passage_id)
                passage_texts.append(passage_text)
        return passage_ids, passage_texts
    
    def create_relevance_map(self):
        relevance_map = defaultdict(dict)
        [example['passages']['is_selected'] for example in self.dataset]
        for example in self.dataset:
            query_id = example["query_id"]
            for i, relevance in enumerate(example["passages"]["is_selected"]):
                doc_id = f"{query_id}-{i}"
                if relevance <= 0:
                    continue
                relevance_map[query_id][doc_id] = relevance
        return relevance_map
    
    def load_data(self):
        self.dataset = self.load_dataset()
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        relevance_map = self.create_relevance_map()
        return question_ids, question_texts, passage_ids, passage_texts, relevance_map