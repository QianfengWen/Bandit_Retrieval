from src.LLM.ChatGPT import ChatGPT
from src.Dataset.travel_dest import TravelDest
from src.Embedding.embedding import create_embeddings, load_embeddings

import numpy as np
import pandas as pd
import os, json
from tqdm import tqdm
from collections import defaultdict

def calculate_cosine_similarity(query_embeddings, passage_embeddings):
    return np.dot(query_embeddings, passage_embeddings.T)

def handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, query_texts, passage_texts):
    if model_name and os.path.exists(query_embeddings_path) and os.path.exists(passage_embeddings_path):
        print(f"Loading embeddings from {query_embeddings_path} and {passage_embeddings_path}")
        return load_embeddings(query_embeddings_path, passage_embeddings_path)
    else:
        print(f"Creating embeddings for {model_name}")
        return create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path)

def main():
    # Configurations
    dataset = TravelDest()
    dataset_name = "travel_dest"
    model_name = "all-MiniLM-L6-v2"
    batch_size = 1
    num = 1000

    output_path = "../data/travel_dest/cache.csv"
    query_embeddings_path = f"../data/{dataset_name}/{model_name}_query_embeddings.pkl"
    passage_embeddings_path = f"../data/{dataset_name}/{model_name}_passage_embeddings.pkl"


    # Load dataset
    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city, prelabel_relevance = dataset.load_data()
    query_embeddings, passage_embeddings = handle_embeddings(model_name, query_embeddings_path, passage_embeddings_path, queries, passages)

    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize rating results
    rating_results = defaultdict(dict)


    # Loop over queries
    for q_id, query, query_embedding in tqdm(zip(question_ids, queries, query_embeddings), desc="Query", total=len(queries)):
        try:
            q_id = int(q_id)  # Ensure query_id is an integer

            # Compute similarity matrix
            similarity_matrix = calculate_cosine_similarity(query_embedding, passage_embeddings)
            sorted_indices = np.argsort(similarity_matrix)[::-1][:num]

            # Create sorted list of (passage_id, passage) pairs
            sorted_passage = [(int(passage_ids[i]), passages[i]) for i in sorted_indices]

            # Batch processing
            sorted_batches = [sorted_passage[i:i + batch_size] for i in range(0, len(sorted_passage), batch_size)]

            for batch in tqdm(sorted_batches, desc="Passage", total=len(sorted_batches)):
                try:
                    # Separate IDs and passages
                    p_ids = [p_id for p_id, _ in batch]
                    batch_passages = [passage for _, passage in batch]

                    # Get scores from LLM
                    scores = llm.get_score(query, batch_passages)

                    # Store results
                    for (p_id, passage), score in zip(batch, scores):
                        rating_results[q_id][p_id] = score

                except Exception as e:
                    print(f"Failed to process passage batch for query {q_id}: {e}")
                    continue

        except Exception as e:
            print(f"Failed to process query {q_id}: {e}")
            continue

    # Convert to DataFrame for CSV export
    rows = []
    for q_id, passages in rating_results.items():
        for p_id, score in passages.items():
            rows.append({"query_id": q_id, "passage_id": p_id, "score": score})

    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
