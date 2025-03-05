# import pdb

# class LLM:
#     """
#     Simple LLM interface for scoring passages based on relevance to a query.
#     In a real implementation, this would call an actual LLM API.
#     """
    
#     def __init__(self, relevance_map=None):
#         """
#         Initialize the LLM interface
        
#         Args:
#             relevance_map: Optional dictionary mapping from (query_id, passage_id) to relevance score.
#                            If provided, we'll use the ground truth relevance instead of calling the LLM.
#         """
#         self.relevance_map = relevance_map
        
#     def get_score(self, query, passage, query_id=None, passage_id=None):
#         """
#         Get the relevance score of a passage for a query
        
#         Args:
#             query: The query text
#             passage: The passage text
#             query_id: Optional query ID for ground truth lookups
#             passage_id: Optional passage ID for ground truth lookups
            
#         Returns:
#             A relevance score between 0 and 1
#         """
#         # If ground truth is provided and we have IDs, use it
#         if self.relevance_map and query_id is not None and passage_id is not None:
#             if query_id in self.relevance_map and passage_id in self.relevance_map[query_id]:
#                 return self.relevance_map[query_id][passage_id]
#             elif query_id not in self.relevance_map:
#                 raise ValueError(f"No relevance map found for query_id: {query_id}")
       
#         return 0.0
    

import abc

class LLM(abc.ABC):
    """
    Abstract class for Language Model (LLM)
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int = 16000) -> str:
        """
        Generate text from the prompt
        prompt (str): The input text prompt.
        max_tokens (int,): The maximum number of tokens to generate.
        """
        pass

    def get_score(self, query, passages):
        """
        Get the relevance score of a passage for a query
        
        Args:
            query: The query text
            passage: The passage text
            query_id: Optional query ID for ground truth lookups
            passage_id: Optional passage ID for ground truth lookups
            
        Returns:
            A relevance score between 0 and 1
        """
        pass