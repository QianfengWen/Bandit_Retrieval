import numpy as np

def visualize(question_ids: list, passage_ids: list, retrieval_results: np.array, relevance_map: dict, k: int):
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
    
