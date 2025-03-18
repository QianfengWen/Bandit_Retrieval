from collections import defaultdict
import os
import json
import unicodedata
import pdb
class TravelDest():
    def load_dataset(self, cache_path="data/travel_dest/cache.csv"):
        self.cache_path = cache_path
        
        file_names = os.listdir("data/travel_dest/corpus")
        file_names = sorted(file_names)

        passages = []
        queries = []
        passage_dict = {}
        cities = []
        for city_id, file in enumerate(file_names):
            city_name = file.split(".txt")[0]
            city_name = unicodedata.normalize("NFC", city_name)
            cities.append(city_name)
            passage_dict[city_id] = []
            with open(f"data/travel_dest/corpus/{file}", "r", encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    passages.append(line)
                    passage_dict[city_id].append(line)

        with open(f"data/travel_dest/queries.txt", "r", encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                queries.append(line)

        with open(f"data/travel_dest/ground_truth.json", "r", encoding='utf-8') as f:
            qrels_iter = json.load(f)

        prelabel_relevance = self.load_cache()
        
        return queries, passages, qrels_iter, passage_dict, cities, prelabel_relevance
                   
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
        for passage_id, passage in enumerate(self.passages):
            passage_map[passage_id] = passage
        passage_ids = list(passage_map.keys())
        passage_texts = list(passage_map.values())
        return passage_ids, passage_texts
    
    def create_relevance_map(self, question_texts):
        relevance_map = defaultdict(dict)
        for query, relevant_doc_names in self.qrels_iter.items():
            query_id = question_texts.index(query)
            for doc_name in relevant_doc_names:
                doc_id = self.cities.index(doc_name)
                relevance_map[query_id + len(self.passages) + 100][doc_id] = 1.0
        return relevance_map        
            
    def create_passage_ids_to_city_map(self, passage_dict, passage_ids, passage_texts):
        passage_id_to_city = {}
        for passage_id, passage_text in zip(passage_ids, passage_texts):
            for city, passages in passage_dict.items():
                if passage_text in passages:
                    passage_id_to_city[passage_id] = city
                    break
        
        return passage_id_to_city

    def load_data(self):
        self.queries, self.passages, self.qrels_iter, self.passage_dict, self.cities, prelabel_relevance = self.load_dataset()
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        relevance_map = self.create_relevance_map(question_texts)
        passage_city_map = self.create_passage_ids_to_city_map(self.passage_dict, passage_ids, passage_texts)
        
        return question_ids, question_texts, passage_ids, passage_texts, relevance_map, passage_city_map, prelabel_relevance
    
    def load_cache(self):
        with open(self.cache_path, "r", encoding='utf-8') as f:
            prelabel_relevance = defaultdict(dict)
            f.readline()
            for line in f:
                query_id, doc_id, score = line.split(',')
                query_id = int(query_id.strip(" \n"))
                doc_id = int(doc_id.strip(" \n"))
                score = int(score.strip(" \n"))
                prelabel_relevance[query_id][doc_id] = score
        return prelabel_relevance

if __name__ == "__main__":
    dataset = TravelDest()
    question_ids, question_texts, passage_ids, passage_texts, relevance_map, passage_city_map, prelabel_relevance = dataset.load_data()
    pdb.set_trace()
