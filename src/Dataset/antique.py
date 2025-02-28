from src.Dataset.dataloader import Dataloader
from collections import defaultdict
import ir_datasets
import random

class Antique(Dataloader):
    def load_dataset(self):
        dataset = ir_datasets.load("antique/train")
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
    
    def load_passages(self, sample_limit=30000):
        # avoid duplicate passages
        passage_map = dict()
        sub = dict()
        
        # First collect all relevant passages from relevance map
        relevant_passages = set()
        for qrel in self.dataset.qrels_iter():
            if qrel.relevance >= 0:  # Using same relevance criteria as in create_relevance_map
                relevant_passages.add(qrel.doc_id)
        
       # Process relevant passages first
        for passage in self.dataset.docs_iter():
            if passage.doc_id in relevant_passages:
                if passage.text in passage_map:
                    sub[passage.doc_id] = passage_map[passage.text]
                else:
                    passage_map[passage.text] = passage.doc_id
        
        # If we haven't reached the sample limit, add random passages
        random.seed(42)  # Set seed for reproducibility
        remaining_slots = sample_limit - len(passage_map)
        if remaining_slots > 0:
            non_relevant_passages = []
            for passage in self.dataset.docs_iter():
                if passage.doc_id not in relevant_passages:
                    non_relevant_passages.append(passage)
            
            # Randomly sample from non-relevant passages
            sampled_passages = random.sample(non_relevant_passages, 
                                        min(remaining_slots, len(non_relevant_passages)))
            
            for passage in sampled_passages:
                if passage.text in passage_map:
                    sub[passage.doc_id] = passage_map[passage.text]
                else:
                    passage_map[passage.text] = passage.doc_id


        self.passage_sub = sub

        passage_ids = list(passage_map.values())
        passage_texts = list(passage_map.keys())

        return passage_ids, passage_texts
    
    # def create_relevance_map(self):
    #     relevance_map = defaultdict(dict)
    #     for qrel in self.dataset.qrels_iter():
    #         query_id = qrel.query_id
    #         doc_id = qrel.doc_id
    #         relevance = qrel.relevance
            
    #         if relevance <= 0:
    #             continue

    #         if query_id in self.question_sub:
    #             query_id = self.question_sub[query_id]
            
    #         relevance_map[query_id][doc_id] = relevance
        
    #     return relevance_map
    def create_relevance_map(self):
        relevance_map = defaultdict(dict)
        for qrel in self.dataset.qrels_iter():
            query_id = qrel.query_id
            doc_id = qrel.doc_id
            relevance = qrel.relevance
            
            if relevance <= 2:
                continue

            if query_id in self.question_sub:
                query_id = self.question_sub[query_id]
            
            if doc_id in self.passage_sub:
                doc_id = self.passage_sub[doc_id]
            
            relevance_map[query_id][doc_id] = relevance
        
        return relevance_map