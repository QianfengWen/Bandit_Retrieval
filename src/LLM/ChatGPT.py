from openai import OpenAI
from src.LLM.llm import LLM
from typing import Optional, Union
from pydantic import BaseModel
import os
import json

class Scores(BaseModel):
    scores: dict[str, float]

response_format = {
    "type": "json_schema",
    "json_schema":{
    "name": "output_schema",
    "schema": Scores.model_json_schema()
    }
}

class ChatGPT(LLM):
    """
    GPT Chat Completion using OpenAI API
    """
    def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18", api_key: str = "API_KEY"):
    # def __init__(self, model_name: str = "gpt-4o-2024-11-20", api_key: str = "API_KEY"):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate(self, message: list[dict], temperature: float = 0.0, response_format: Optional[dict] = None) -> Union[str, None]:
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=message,
                temperature=temperature,
                response_format=response_format
            )

        except Exception as e:
            print(e)
            return None
        return response.choices[0].message.content
    
    def get_score(self, query: str, passages: list[str]) -> list[float]:
        """
        Get the relevance score of each passage for a given query using a single LLM call.

        Args:
            query: The query text.
            passages: A list of passage texts.
        
        Returns:
            A list of relevance scores between 0 and 1.
        """
        formatted_passages = "\n".join([f"{i+1}. {p}" for i, p in enumerate(passages)])
        prompt_template = """
        Given a query and a list of passages, you must provide a score on an integer scale of 0 to 3 with the following meanings:
        0 = represent that the passage has nothing to do with the query, 
        1 = represents that the passage seems related to the query but does not answer it, 
        2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
        3 = represents that the passage is dedicated to the query and contains the exact answer.

        Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

        Query: {query}
        Passages: {passages}

        Split this problem into steps:
        Consider the underlying intent of the search.
        Measure how well the content matches a likely intent of the query (M).
        Measure how trustworthy the passage is (T).
        Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
        Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
        Return the results in JSON format.
        """

        prompt = [
            {"role": "system", "content": "You are an assistant that evaluates the relevance of passages to a given query. "},
            {"role": "user", "content": prompt_template.format(query=query, passages=formatted_passages)}
        ]

        response = self.generate(prompt, response_format=response_format)

        try:
            scores = json.loads(response)["scores"]
            if len(scores) == 1:
                return [scores.get("final score", 0.1)]
            else:
                return [scores.get(f"passage{i+1}", 0.1) for i in range(len(passages))] 
        except Exception as e:
            print("Failed to parse LLM response:", e)
            return [0.1] * len(passages) 
        

if __name__ == "__main__":
    # Load the API key from the environment variable
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    query = "What is the capital of France?"
    passages = ["Paris is the capital of France.", "France is a country in Europe.", "Paris is known for its Eiffel Tower."]
    # for passage in passages:
    #     print(passage)
    #     print(llm.get_score(query, [passage]))
    print(llm.get_score(query, passages))
