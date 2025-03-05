from collections import defaultdict
import os
import json
import unicodedata

class TravelDest():
    def load_dataset(self):
        file_names = os.listdir("../../data/travel_dest/corpus")
        file_names = sorted(file_names)

        passages = []
        queries = []
        passage_dict = {}
        cities = []
        for file in file_names:
            city_name = file.split(".txt")[0]
            city_name = unicodedata.normalize("NFC", city_name)
            cities.append(city_name)
            passage_dict[city_name] = []
            with open(f"../../data/travel_dest/corpus/{file}", "r") as f:
                for line in f:
                    line = line.rstrip('\n')
                    passages.append(line)
                    passage_dict[city_name].append(line)

        with open(f"../../data/travel_dest/queries.txt", "r") as f:
            for line in f:
                line = line.rstrip('\n')
                queries.append(line)

        with open(f"../../data/travel_dest/ground_truth.json", "r", encoding='utf-8') as f:
            qrels_iter = json.load(f)
            return queries, passages, qrels_iter, passage_dict, cities
                   
    def load_questions(self):
        question_map = dict()
        sub = dict()

        for query_id, query in enumerate(self.queries):
            if query in question_map:
                sub[query_id + len(self.passages) + 100] = question_map[query]
            else:
                question_map[query] = query_id + len(self.passages) + 100
            
        self.question_sub = sub
        question_ids = list(question_map.values())
        question_texts = list(question_map.keys())
        return question_ids, question_texts

    def load_passages(self):
        passage_map = dict()
        sub = dict()

        for passage_id, passage in enumerate(self.passages):
            if passage in passage_map:
                sub[passage_id] = passage_map[passage]
            else:
                passage_map[passage] = passage_id

        self.passage_sub = sub
        passage_ids = list(passage_map.values())
        passage_texts = list(passage_map.keys())
        return passage_ids, passage_texts
    
    def create_relevance_map(self, question_texts):
        relevance_map = defaultdict(dict)
        for query, relevant_doc_names in self.qrels_iter.items():
            query_id = question_texts.index(query)
            for doc_name in relevant_doc_names:
                doc_id = self.cities.index(doc_name)
                relevance_map[query_id][doc_id] = 1.0
        return relevance_map
            
    def load_data(self):
        self.queries, self.passages, self.qrels_iter, self.passage_dict, self.cities = self.load_dataset()
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        relevance_map = self.create_relevance_map(question_texts)
        
        return question_ids, question_texts, passage_ids, passage_texts, relevance_map