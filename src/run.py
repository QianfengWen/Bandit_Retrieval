from src.Dataset.msmarco import MSMARCO
from src.Dataset.antique import Antique
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.scidocs import Scidocs
from src.Embedding.embedding import create_embeddings, load_embeddings
from src.GPUCB.gpucb import GPUCB
import numpy as np

# pipeline kwargs
model_name = "all-MiniLM-L6-v2"
dataset_name = "scidocs"
query_embeddings_path = f"data/{dataset_name}/query_embeddings.pkl"
passage_embeddings_path = f"data/{dataset_name}/passage_embeddings.pkl"

"""
1. X-axis choice:
    a) Passage index as X-axis
    b) Score (e.g. Dense Retrieval distance) as X-axis

2. GP-UCB kernel choice:
    a) RBF
    b) Matern
    c) Rational Quadratic
    d) Exponential
    e) Dot Product

3. GP-UCB beta choice:
    a) 100
    b) 1000
    c) 10000

4. GP-UCB initialization choice (cold-start):
    a) Random
    b) Sampling methods (MCMC, Thompson Sampling, etc.)
    c) Dense Retrieval (choose top-k passages for each query)

5. Baseline choice:
"""

def handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, query_texts, passage_texts):
    if model_name:
        return create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path)
    else:
        return load_embeddings(query_embeddings_path, passage_embeddings_path)

def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)



if __name__ == "__main__":
    dataset = Scidocs()
    question_ids, question_texts, passage_ids, passage_texts, relevance_map = dataset.load_data()

    # load embeddings
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, question_texts, passage_texts)

    # create mapping
    query_idx_to_id = {i: id for i, id in enumerate(question_ids)}
    passage_idx_to_id = {i: id for i, id in enumerate(passage_ids)}
    query_id_to_idx = {id: i for i, id in enumerate(question_ids)}
    passage_id_to_idx = {id: i for i, id in enumerate(passage_ids)}

    # run gp-ucb
    gpucb = GPUCB(query_embeddings, passage_embeddings, relevance_map)

    cosine_metrics = calculate_cosine_similarity(query_embeddings, passage_embeddings)

    
        





    

    
    
