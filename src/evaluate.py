import math
from collections import defaultdict


def evaluate(results:dict, relevance_map, cutoff:list[int], threshold:int):
    """
    Evaluate the performance of the model using various metrics.
    :param total_items: list of retrieved items
    :param total_truth: list of ground truth items
    :param cutoff: list of cutoff values for evaluation
    :return: dictionary with evaluation results
    """
    ndcg_k_dict = defaultdict(list)
    prec_k_dict = defaultdict(list)
    rec_k_dict = defaultdict(list)
    map_k_dict = defaultdict(list)
    mrr_k_dict = defaultdict(list)

    for q_id, result in results.items():
        pred = result['pred']
        relevance = relevance_map[q_id]
        gt = set([p_id for p_id, rel in relevance.items() if rel >= threshold])
        results[q_id]['gt'] = list(gt)

        for k in cutoff:
            if len(gt) == 0:
                continue
            prec_k_dict[k].append(precision_k(pred, gt, k))
            rec_k_dict[k].append(recall_k(pred, gt, k))
            map_k_dict[k].append(mean_average_precision_k(pred, gt, k))
            ndcg_k_dict[k].append(normalized_dcg_k(pred, relevance, k))
            mrr_k_dict[k].append(reciprocal_rank_k(pred, gt, k))

    print("\n\n > Evaluation Results")
    metric = {}
    for k in cutoff:
        print("\n >> Cutoff at ", k)
        metric[f"Precision@{k}"] = sum(prec_k_dict[k]) / len(prec_k_dict[k])
        metric[f"Recall@{k}"] = sum(rec_k_dict[k]) / len(rec_k_dict[k])
        metric[f"MAP@{k}"] = sum(map_k_dict[k]) / len(map_k_dict[k])
        metric[f"NDCG@{k}"] = sum(ndcg_k_dict[k]) / len(ndcg_k_dict[k])
        metric[f"MRR@{k}"] = sum(mrr_k_dict[k]) / len(mrr_k_dict[k])

        print(f"Precision@{k}: {sum(prec_k_dict[k]) / len(prec_k_dict[k]):.4f}")
        print(f"Recall@{k}: {sum(rec_k_dict[k]) / len(rec_k_dict[k]):.4f}")
        print(f"MAP@{k}: {sum(map_k_dict[k]) / len(map_k_dict[k]):.4f}")
        print(f"NDCG@{k}: {sum(ndcg_k_dict[k]) / len(ndcg_k_dict[k]):.4f}")
        print(f"MRR@{k}: {sum(mrr_k_dict[k]) / len(mrr_k_dict[k]):.4f}")

    return metric, results


def recall_k(items, truth, k):
    """
    Compute recall@k
    :param items: list of retrieved items
    :param truth: list of ground truth items
    :param k: number of top items to consider
    :return: recall@k
    """
    assert len(items)>= k, "Number of items should be greater than or equal to k"
    assert len(truth)>= 0, "Number of truth items should be greater than or equal to 0"
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
    assert len(items)>= k, "Number of items should be greater than or equal to k"
    assert len(truth)>= 0, "Number of truth items should be greater than or equal to 0"
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

def reciprocal_rank_k(items, truth, k):
    """
    Compute mean reciprocal rank@k
    :param items: list of retrieved items
    :param truth: list of ground truth items
    :param k: number of top items to consider
    :return: mean reciprocal rank@k
    """
    for i, item in enumerate(items[:k]):
        if item in truth:
            return 1 / (i + 1)
    return 0.0