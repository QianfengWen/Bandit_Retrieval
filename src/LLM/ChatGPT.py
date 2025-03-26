from openai import OpenAI
from src.LLM.llm import LLM
from typing import Optional, Union
from pydantic import BaseModel
import os, json, csv
from collections import defaultdict

class Scores(BaseModel):
    scores: dict[str, int]

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
    # def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18", api_key: str = "API_KEY"):
    def __init__(self, model_name: str = "gpt-4o-2024-11-20", api_key: str = "API_KEY"):
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
    
    def get_score(
            self, 
            query: str, 
            passages: list[str], 
            query_id: int = None, 
            passage_ids: list[int] = None, 
            cache: str = None, 
            update_cache: bool = False
        ) -> list[float]:
        """
        Get the relevance score of each passage for a given query using a single LLM call.

        Args:
            query: The query text.
            passages: A list of passage texts.
        
        Returns:
            A list of relevance scores between 0 and 1.
        """
        if cache:
            try:
                assert query_id is not None and passage_ids is not None
                all_ratings = cache[query_id]
                target_ratings = [all_ratings[p_id] for p_id in passage_ids]
                print(f"Cache hit for query {query_id}, using precomputed ratings ...")
                print(f"Ratings: {target_ratings}")
                return target_ratings
            
            except KeyError:
                print(f"Cache miss for query {query_id}, using LLM ...")
                pass

            except AssertionError:
                # print("Cache is enabled but query_id and passage_ids are not provided, using LLM ...")
                pass
            
            except Exception as e:
                # print(f"Caching error: {e}, using LLM ...")
                pass

        # cache miss for some passages
        cache = cache or defaultdict(dict)
        cache.setdefault(query_id, {})

        few_shot_examples = {
            passage: cache[query_id][pid] 
            for pid, passage in zip(passage_ids, passages) 
            if pid in cache[query_id]
        }

        if few_shot_examples:
            few_shot_texts = f"Attached are example ratings:\n{json.dumps(few_shot_examples, indent=2)}"
        else:
            few_shot_texts = ""
        
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
        Do not provide any code in result. Return a JSON object mapping passage index to its relevance score, the index starts from 0.
        
        {few_shot_texts}
        """
        prompt = [
            {"role": "system", "content": "You are an assistant that evaluates the relevance of passages to a given query. "},
            {"role": "user", "content": prompt_template.format(query=query, passages=formatted_passages, few_shot_texts=few_shot_texts)}
        ]

        response = self.generate(prompt, response_format=response_format)

        try:
            scores = json.loads(response)["scores"]
            scores = [scores.get(str(i), -1) for i in range(len(passages))] 
            if update_cache and query_id is not None and passage_ids is not None:
                new_entries = []
                
                for p_id, score in zip(passage_ids, scores):
                    if p_id not in cache[query_id]:
                        new_entries.append([query_id, p_id, score])

                if new_entries:
                    # print(f"Updating cache with {len(new_entries)} new entries ...")
                    file_exists = os.path.exists(update_cache)
                    
                    with open(update_cache, mode='a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        if not file_exists:
                            writer.writerow(["query_id", "passage_id", "score"])
                        
                        writer.writerows(new_entries)
            
            return scores
        
        except Exception as e:
            print("Failed to parse LLM response:", e)
            return [-1] * len(passages) 
        

if __name__ == "__main__":
    from src.Dataset.travel_dest import TravelDest
    import os
    import json
    from tqdm import tqdm

    # Config
    batch_size = 5
    rating_results = dict()
    lm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))

    # Load dataset
    dataset = TravelDest()
    question_ids, queries, passage_ids, passages, relevance_map, passage_to_city = dataset.load_data()

    for q_id, query in tqdm(zip(question_ids, queries), desc="Query", total=len(queries)):
        rating_results[q_id] = dict()
        ground_truth = relevance_map[q_id]

        # Create list of (p_id, passage) pairs to retain p_id info
        passage_to_eval = [
            (p_id, passage) for p_id, passage in zip(passage_ids, passages) 
            if passage_to_city[p_id] in ground_truth
        ]
        
        # Batch processing
        passage_batches = [
            passage_to_eval[i:i + batch_size] 
            for i in range(0, len(passage_to_eval), batch_size)
        ]

        for batch in tqdm(passage_batches, desc="Batch", total=len(passage_batches)):
            # Get the passage texts only for LLM input
            passage_texts = [passage for _, passage in batch]
            try:
                scores = lm.get_score(query, passage_texts)
            except Exception as e:
                print(f"Failed to process query {q_id} and passages {passage_batches}: {e}")
                continue
            # Store the scores using original p_id
            for (p_id, _), score in zip(batch, scores):
                rating_results[q_id][p_id] = score
        
        # Optional logging for debugging
        print(f"Processed query {q_id}: {rating_results[q_id]}")
        
    # Save results to JSON
    with open("data/travel_dest/rating_results.json", "w", encoding="utf-8") as f:
        json.dump(rating_results, f, indent=4)

            
