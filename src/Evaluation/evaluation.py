import numpy as np
import math

def evaluate(question_ids: list, passage_ids: list, retrieval_results: np.array, relevance_map: dict, evaluation_func: callable, k: int):
    """
    Evaluate retrieval results
    :param question_ids: list of question ids
    :param passage_ids: list of passage ids
    :param retrieval_results: numpy array of retrieval results
    :param relevance_map: dictionary of relevance mapping
    :param evaluation_func: evaluation function
    :param k: number of top items to consider
    :return: evaluation score
    """
    scores = []
    for i, qid in enumerate(question_ids):
        # q_result = retrieval_results[i].tolist()
        # q_result = [passage_ids.index(j) for j in q_result]
        q_result = retrieval_results[i]
        truth = relevance_map[qid]
        scores.append(evaluation_func(q_result, truth, k))
    return np.mean(scores)


def recall_k(items, truth, k):
    """
    Compute recall@k
    :param items: list of retrieved items
    :param truth: list of ground truth items
    :param k: number of top items to consider
    :return: recall@k
    """
    if len(truth) == 0:
        return 0
    items = items[:k]
    return len(set(items) & set(truth)) / len(truth)


def precision_k(items, truth, k):
    """
    Compute precision@k
    :param items: list of retrieved items
    :param truth: list of ground truth items
    :param k: number of top items to consider
    :return: precision@k
    """
    items = items[:k]
    return len(set(items) & set(truth)) / len(items)


def mean_average_precision_k(items, truth, k):
    """
    Compute mean average precision@k
    :param items: list of retrieved items
    :param truth: list of ground truth items
    :param k: number of top items to consider
    :return: mean average precision@k
    """
    if len(truth) == 0:
        return 0
    running_sum = 0
    num_correct = 0
    for i, item in enumerate(items[:k]):
        if item in truth:
            num_correct += 1
            running_sum += num_correct / (i + 1)
    return running_sum / len(truth)

def normalized_dcg_k(items, relevance_map, k):
    """
    Compute normalized discounted cumulative gain@k with graded relevance
    :param items: list of retrieved items
    :param relevance_map: dict of {item_id: relevance score}, i.e., ground truth
    :param k: number of top items to consider
    :return: normalized discounted cumulative gain@k
    """

    def dcg(recommended, relevance_map, k):
        return sum((2 ** relevance_map.get(item, 0) - 1) / math.log2(i + 2)
                   for i, item in enumerate(recommended[:k]))

    def idcg(relevance_map, k):
        ideal_rels = sorted(relevance_map.values(), reverse=True)
        return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))

    actual_dcg = dcg(items, relevance_map, k)
    ideal_dcg = idcg(relevance_map, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg
