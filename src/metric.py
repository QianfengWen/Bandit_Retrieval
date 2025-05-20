import numpy as np
import math

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
