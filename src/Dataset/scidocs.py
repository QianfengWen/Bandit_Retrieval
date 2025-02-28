from src.Dataset.dataloader import Dataloader
from collections import defaultdict
import ir_datasets

class Scidocs(Dataloader):
    def load_dataset(self):
        dataset = ir_datasets.load("beir/scidocs")
        return dataset
    
    def load_questions(self):
        # avoid duplicate questions
        question_map = dict()
        sub = dict()

        for query in self.dataset.queries_iter():
            if query.text in question_map:
                sub[query.query_id] = question_map[query.text]
            else:
                question_map[query.text] = query.query_id
            
        self.question_sub = sub

        question_ids = list(question_map.values())
        question_texts = list(question_map.keys())

        return question_ids, question_texts


    def load_passages(self):
        # avoid duplicate passages
        passage_map = dict()
        sub = dict()

        for passage in self.dataset.docs_iter():
            if passage.text in passage_map:
                sub[passage.doc_id] = passage_map[passage.text]
            else:
                passage_map[passage.text] = passage.doc_id

        self.passage_sub = sub

        passage_ids = list(passage_map.values())
        passage_texts = list(passage_map.keys())

        return passage_ids, passage_texts
    

    def create_relevance_map(self):
        relevance_map = defaultdict(dict)
        for qrel in self.dataset.qrels_iter():
            query_id = qrel.query_id
            doc_id = qrel.doc_id
            relevance = qrel.relevance
            
            if relevance <= 0:
                continue

            if query_id in self.question_sub:
                query_id = self.question_sub[query_id]
            
            if doc_id in self.passage_sub:
                doc_id = self.passage_sub[doc_id]
            
            relevance_map[query_id][doc_id] = relevance
        
        return relevance_map