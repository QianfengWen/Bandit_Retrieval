
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

    def get_score(self, query, passages, query_ids, passage_ids, cache=None, update_cache=False):
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