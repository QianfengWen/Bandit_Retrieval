import os
import sys

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

from src.Dataset.factory import handle_dataset
from src.evaluate import evaluate
from src.utils import load_dataset

# dataset_name = sys.argv[1]
dataset_name = "nfcorpus"

dataset = handle_dataset(dataset_name)
query_ids, queries, passage_ids, passages, relevance_map = dataset.load_data()


searcher = LuceneSearcher(f"data/{dataset_name}/bm25/indexes")

results = {}
for qid, q in tqdm(zip(query_ids, queries), desc=" > BM25 Search", total=len(query_ids)):
    hits = searcher.search(q, k=50)

    results[qid] = {
        "pred": [hit.docid for hit in hits],
        "score": [hit.score for hit in hits]
    }

metric, results = evaluate(results, relevance_map, cutoff=[1, 5, 10, 50], threshold=dataset.relevance_threshold)
