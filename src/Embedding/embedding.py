import pickle
import numpy as np
import os


LOCAL_EMBEDDERS = {"hashing", "local-hashing", "hashing-vectorizer"}

def create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path):
    if model_name in LOCAL_EMBEDDERS:
        return create_hashing_embeddings(
            query_texts,
            passage_texts,
            query_embeddings_path,
            passage_embeddings_path,
        )

    import torch
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder.to(device)

    query_embeddings = embedder.encode(query_texts, convert_to_tensor=False, show_progress_bar=True)
    passages_embeddings = embedder.encode(passage_texts, convert_to_tensor=False, show_progress_bar=True)

    save_embeddings(query_embeddings, passages_embeddings, query_embeddings_path, passage_embeddings_path)
   
    return query_embeddings, passages_embeddings


def create_hashing_embeddings(query_texts, passage_texts, query_embeddings_path, passage_embeddings_path):
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=384,
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )
    query_embeddings = vectorizer.transform(query_texts).toarray().astype(np.float32, copy=False)
    passage_embeddings = vectorizer.transform(passage_texts).toarray().astype(np.float32, copy=False)

    save_embeddings(query_embeddings, passage_embeddings, query_embeddings_path, passage_embeddings_path)

    return query_embeddings, passage_embeddings

def save_embeddings(query_embeddings, passage_embeddings=None, query_embeddings_path=None, passage_embeddings_path=None):
    os.makedirs(os.path.dirname(query_embeddings_path), exist_ok=True)
    if passage_embeddings_path is not None:
        os.makedirs(os.path.dirname(passage_embeddings_path), exist_ok=True)
    with open(query_embeddings_path, "wb") as f:
        pickle.dump(query_embeddings, f)

    if passage_embeddings is not None:
        with open(passage_embeddings_path, "wb") as f:
            pickle.dump(passage_embeddings, f)
    
    return

def load_embeddings(query_embeddings_path, passage_embeddings_path):
    with open(query_embeddings_path, "rb") as f:
        query_embeddings = pickle.load(f)
        assert isinstance(query_embeddings, np.ndarray), "query_embeddings should be a numpy array"

    with open(passage_embeddings_path, "rb") as f:
        passage_embeddings = pickle.load(f)
        assert isinstance(passage_embeddings, np.ndarray), "passage_embeddings should be a numpy array"
    return query_embeddings, passage_embeddings

def handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, query_texts, passage_texts):
    if model_name and os.path.exists(query_embeddings_path) and os.path.exists(passage_embeddings_path):
        print(f"Loading embeddings from {query_embeddings_path} and {passage_embeddings_path}")
        return load_embeddings(query_embeddings_path, passage_embeddings_path)
    else:
        print(f"Creating embeddings for {model_name}")
        return create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path)
