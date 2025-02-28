class LLM:
    """
    Simple LLM interface for scoring passages based on relevance to a query.
    In a real implementation, this would call an actual LLM API.
    """
    
    def __init__(self, relevance_map=None):
        """
        Initialize the LLM interface
        
        Args:
            relevance_map: Optional dictionary mapping from (query_id, passage_id) to relevance score.
                           If provided, we'll use the ground truth relevance instead of calling the LLM.
        """
        self.relevance_map = relevance_map
        
    def get_score(self, query, passage, query_id=None, passage_id=None):
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
        # If ground truth is provided and we have IDs, use it
        if self.relevance_map and query_id is not None and passage_id is not None:
            if (query_id, passage_id) in self.relevance_map:
                return self.relevance_map[(query_id, passage_id)]
        
        # In a real implementation, we would call an actual LLM API here
        # For now, implement a simple similarity measure based on term overlap
        query_terms = set(query.lower().split())
        passage_terms = set(passage.lower().split())
        
        # Simple Jaccard similarity
        if not query_terms or not passage_terms:
            return 0.0
        
        intersection = len(query_terms.intersection(passage_terms))
        union = len(query_terms.union(passage_terms))
        
        return intersection / union 