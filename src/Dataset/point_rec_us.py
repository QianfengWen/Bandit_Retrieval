from collections import defaultdict
import os
import json
import unicodedata
import pandas as pd

class PointRecUS:
    def load_dataset(self):
        self.cache_path = f"data/point_rec_us/cache.csv"

        # Step 1: Read file names and sort them
        file_names = sorted(os.listdir(f"data/point_rec_us/corpus"))

        # Step 2: Initialize containers
        passages = []
        passage_dict = {}
        cities = []
        pois = []

        passage_id = 0
        poi_count = 0

        # Step 3: Load corpus data more efficiently
        for city_id, file in enumerate(file_names):
            city_name = file.split(".json")[0]
            cities.append(city_name)

            # Read file content at once for faster I/O
            with open(f"data/point_rec_us/corpus/{file}", "r", encoding='utf-8') as f:
                poi_data = json.load(f)
                # print(f"For city {city_name}")
                for poi_id in poi_data.keys():
                    poi = poi_data[poi_id]
                    if poi["snippets"] != []:
                        passage_dict[poi_count] = []
                        for snippet in poi["snippets"]:
                            passages.append(snippet['snippet'])
                            passage_dict[poi_count].append(passage_id)
                            passage_id += 1
                        poi_count += 1
                        pois.append(poi_id)

        # Step 4: Load queries
        queries = {}
        with open(f"data/point_rec_us/infoneeds.json", "r", encoding='utf-8') as f:
            query_data = json.load(f)
            for query_id in query_data.keys():
                if "US" in query_data[query_id]["Country"] and query_data[query_id]["Description"] not in queries.values():
                    queries[query_id] = query_data[query_id]["Description"]
        
        # Step 5: Load ground truth directly from JSON
        with open(f"data/point_rec_us/relevance/qrels.trec", "r", encoding='utf-8') as f:
            qrels_iter_data = f.readlines()
            qrels_iter = []
            for raw_qrels_iter in qrels_iter_data:
                query_id, _, poi_id, relevance = raw_qrels_iter.strip().split(" ")
                qrels_iter.append((query_id, query_data[query_id]["Description"], poi_id, relevance))
        
        prelabel_relevance = self.load_cache()
        return queries, passages, qrels_iter, passage_dict, cities, pois, prelabel_relevance

    def load_questions(self):
        # Direct dictionary comprehension for fast mapping
        question_map = {query: query_id + len(self.passages) + 100 
                        for query_id, query in enumerate(self.queries.values())}

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

        for query_id in query_to_id.values():
            relevance_map[query_id + len(self.passages) + 100] = {}

        # Direct mapping using dictionary lookup
        for raw_query_id, query, poi_id, relevance in self.qrels_iter:
            if query in query_to_id:
                query_id = query_to_id[query] + len(self.passages) + 100
                if relevance == "3" or relevance == "2":
                    # print(f"raw_query_id: {raw_query_id}, query_id: {query_id}, POI: {poi_id}, Relevance: {relevance}")
                    if poi_id in self.pois:
                        relevance_map[query_id][self.pois.index(poi_id)] = 1.0
        
        # sort the relevance map by keys
        relevance_map = dict(sorted(relevance_map.items()))
        
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
            print(f"Loading cache from {self.cache_path}")
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
        self.queries, self.passages, self.qrels_iter, self.passage_dict, self.cities, self.pois, prelabel_relevance = self.load_dataset()
        
        # Step 2: Load questions and passages using fast mapping
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        
        # # Step 3: Create fast relevance map and passage-to-city map
        relevance_map = self.create_relevance_map(question_texts)
        passage_city_map = self.create_passage_ids_to_city_map(self.passage_dict)

        # Step 4: Return pre-processed data
        return (
            question_ids, question_texts, 
            passage_ids, passage_texts, 
            relevance_map, self.passage_dict,
            passage_city_map, prelabel_relevance
        )