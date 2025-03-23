# from collections import defaultdict
# import os
# import json
# import unicodedata
# import pdb
# class TravelDest():
#     def load_dataset(self, cache_path="data/travel_dest/cache.csv"):
#         self.cache_path = cache_path
        
#         file_names = os.listdir("data/travel_dest/corpus")
#         file_names = sorted(file_names)

#         passages = []
#         queries = []
#         passage_dict = {}
#         cities = []
#         for city_id, file in enumerate(file_names):
#             city_name = file.split(".txt")[0]
#             city_name = unicodedata.normalize("NFC", city_name)
#             cities.append(city_name)
#             passage_dict[city_id] = []
#             with open(f"data/travel_dest/corpus/{file}", "r", encoding='utf-8') as f:
#                 for line in f:
#                     line = line.rstrip('\n')
#                     passages.append(line)
#                     passage_dict[city_id].append(line)

#         with open(f"data/travel_dest/queries.txt", "r", encoding='utf-8') as f:
#             for line in f:
#                 line = line.rstrip('\n')
#                 queries.append(line)

#         with open(f"data/travel_dest/ground_truth.json", "r", encoding='utf-8') as f:
#             qrels_iter = json.load(f)

#         prelabel_relevance = self.load_cache()
        
#         return queries, passages, qrels_iter, passage_dict, cities, prelabel_relevance
                   
#     def load_questions(self):
#         question_map = dict()
#         sub = dict()

#         for query_id, query in enumerate(self.queries):
#             if query in question_map:
#                 sub[query_id + len(self.passages) + 100] = question_map[query]
#             else:
#                 question_map[query] = query_id + len(self.passages) + 100
            
#         self.question_sub = sub
#         question_ids = list(question_map.values())
#         question_texts = list(question_map.keys())
#         return question_ids, question_texts

#     def load_passages(self):
#         passage_map = dict()
#         for passage_id, passage in enumerate(self.passages):
#             passage_map[passage_id] = passage
#         passage_ids = list(passage_map.keys())
#         passage_texts = list(passage_map.values())
#         return passage_ids, passage_texts
    
#     def create_relevance_map(self, question_texts):
#         relevance_map = defaultdict(dict)
#         for query, relevant_doc_names in self.qrels_iter.items():
#             query_id = question_texts.index(query)
#             for doc_name in relevant_doc_names:
#                 doc_id = self.cities.index(doc_name)
#                 relevance_map[query_id + len(self.passages) + 100][doc_id] = 1.0
#         return relevance_map        
            
#     def create_passage_ids_to_city_map(self, passage_dict, passage_ids, passage_texts):
#         passage_id_to_city = {}
#         for passage_id, passage_text in zip(passage_ids, passage_texts):
#             for city, passages in passage_dict.items():
#                 if passage_text in passages:
#                     passage_id_to_city[passage_id] = city
#                     break
        
#         return passage_id_to_city

#     def load_data(self):
#         self.queries, self.passages, self.qrels_iter, self.passage_dict, self.cities, prelabel_relevance = self.load_dataset()
#         question_ids, question_texts = self.load_questions()
#         passage_ids, passage_texts = self.load_passages()
#         relevance_map = self.create_relevance_map(question_texts)
#         passage_city_map = self.create_passage_ids_to_city_map(self.passage_dict, passage_ids, passage_texts)
        
#         return question_ids, question_texts, passage_ids, passage_texts, relevance_map, self.passage_dict, passage_city_map, prelabel_relevance

    
#     def load_cache(self):
#         with open(self.cache_path, "r", encoding='utf-8') as f:
#             prelabel_relevance = defaultdict(dict)
#             f.readline()
#             for line in f:
#                 query_id, doc_id, score = line.split(',')
#                 query_id = int(query_id.strip(" \n"))
#                 doc_id = int(doc_id.strip(" \n"))
#                 score = int(score.strip(" \n"))
#                 prelabel_relevance[query_id][doc_id] = score
#         return prelabel_relevance


from collections import defaultdict
import os
import json
import unicodedata
import pandas as pd

class TravelDest:
    def load_dataset(self, cache_path="data/travel_dest/cache.csv"):
        self.cache_path = cache_path

        # Step 1: Read file names and sort them
        file_names = sorted(os.listdir("data/travel_dest/corpus"))

        # Step 2: Initialize containers
        passages = []
        passage_dict = {}
        cities = []

        passage_id = 0  # Use numeric passage ID for faster lookup

        # Step 3: Load corpus data more efficiently
        for city_id, file in enumerate(file_names):
            city_name = file.split(".txt")[0]
            city_name = unicodedata.normalize("NFC", city_name)
            cities.append(city_name)

            # Store city_id instead of string
            passage_dict[city_id] = []  

            # Read file content at once for faster I/O
            with open(f"data/travel_dest/corpus/{file}", "r", encoding='utf-8') as f:
                city_passages = f.read().strip().split('\n')
                for passage in city_passages:
                    passages.append(passage)  
                    passage_dict[city_id].append(passage_id)  # Save passage ID only
                    passage_id += 1

        # Step 4: Load queries
        with open(f"data/travel_dest/queries.txt", "r", encoding='utf-8') as f:
            queries = f.read().strip().split('\n')

        # Step 5: Load ground truth directly from JSON
        with open(f"data/travel_dest/ground_truth.json", "r", encoding='utf-8') as f:
            qrels_iter = json.load(f)

        # Step 6: Load cache using pandas (faster parsing)
        prelabel_relevance = self.load_cache()

        return queries, passages, qrels_iter, passage_dict, cities, prelabel_relevance
    
    def load_questions(self):
        # Direct dictionary comprehension for fast mapping
        question_map = {query: query_id + len(self.passages) + 100 
                        for query_id, query in enumerate(self.queries)}

        question_ids = list(question_map.values())
        question_texts = list(question_map.keys())

        # Create reverse mapping for fast lookup
        self.question_sub = {
            query_id + len(self.passages) + 100: question_map[query]
            for query_id, query in enumerate(self.queries) if query in question_map
        }

        return question_ids, question_texts

    def load_passages(self):
        # Fast dictionary comprehension to create mapping
        passage_map = {passage_id: passage for passage_id, passage in enumerate(self.passages)}
        passage_ids = list(passage_map.keys())
        passage_texts = list(passage_map.values())

        return passage_ids, passage_texts
    
    def create_relevance_map(self, question_texts):
        # Precompute query-to-id mapping
        query_to_id = {q: idx for idx, q in enumerate(question_texts)}
        relevance_map = defaultdict(dict)

        # Direct mapping using dictionary lookup
        for query, relevant_doc_names in self.qrels_iter.items():
            if query in query_to_id:
                query_id = query_to_id[query] + len(self.passages) + 100
                for doc_name in relevant_doc_names:
                    if doc_name in self.cities:
                        doc_id = self.cities.index(doc_name)
                        relevance_map[query_id][doc_id] = 1.0
        
        return relevance_map
    
    def create_passage_ids_to_city_map(self, passage_dict):
        # Create reverse mapping directly using IDs (O(1) lookup)
        passage_id_to_city = {
            passage_id: city_id
            for city_id, passage_ids in passage_dict.items()
            for passage_id in passage_ids
        }
        return passage_id_to_city

    def load_cache(self):
        try:
            # Use pandas for fast parsing and direct mapping
            df = pd.read_csv(self.cache_path, skipinitialspace=True)
            prelabel_relevance = defaultdict(dict)

            # Efficient row-wise mapping using zip
            for query_id, doc_id, score in zip(df['query_id'], df['passage_id'], df['score']):
                prelabel_relevance[int(query_id)][int(doc_id)] = float(score)

            return prelabel_relevance
        except FileNotFoundError:
            return defaultdict(dict)


    def load_data(self):
        # Step 1: Load dataset components
        self.queries, self.passages, self.qrels_iter, self.passage_dict, self.cities, prelabel_relevance = self.load_dataset()
        
        # Step 2: Load questions and passages using fast mapping
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        
        # Step 3: Create fast relevance map and passage-to-city map
        relevance_map = self.create_relevance_map(question_texts)
        passage_city_map = self.create_passage_ids_to_city_map(self.passage_dict)

        # Step 4: Return pre-processed data
        return (
            question_ids, question_texts, 
            passage_ids, passage_texts, 
            relevance_map, self.passage_dict, 
            passage_city_map, prelabel_relevance
        )
