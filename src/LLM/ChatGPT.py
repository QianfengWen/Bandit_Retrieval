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
        prompt = [
            {"role": "system", "content": "You are an assistant that evaluates the relevance of passages to a given query. " 
                                          "Provide a score between 0 and 1 for each passage, in JSON format."},
            {"role": "user", "content": f"Query: {query}\nPassages:\n{formatted_passages}\n\n"
                                        "Return a JSON object mapping passage index to its relevance score, e.g., {\"1\": 0.8, \"2\": 0.5}."}
        ]

        response = self.generate(prompt, response_format=response_format)
        try:
            scores = json.loads(response)["scores"]
            return [scores.get(str(i + 1), 0.1) for i in range(len(passages))] 
        except Exception as e:
            print("Failed to parse LLM response:", e)
            return [0.1] * len(passages) 
        

if __name__ == "__main__":
    # Load the API key from the environment variable
    llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    query = "What is the capital of France?"
    passages = ["Paris is the capital of France.", "France is a country in Europe.", "Paris is known for its Eiffel Tower."]
    for passage in passages:
        print(passage)
        print(llm.get_score(query, [passage]))
