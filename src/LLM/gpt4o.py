import math
import os
from math import inf
from pathlib import Path

from openai import OpenAI
from typing import Optional, Union
import csv

class gpt4o:
    def __init__(self, prompt_type, score_type):
        self.model_name = "gpt-4o-2024-08-06"
        self.prompt_type = prompt_type
        self.score_type = score_type
        self.client = OpenAI()

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open(Path(cur_dir) / 'prompt' / f'{prompt_type}.txt', 'r', encoding='utf-8') as file:
            self.prompt_template = file.read()

    def get_score(
            self,
            queries: list[str],
            passages: list[str],
            query_ids: list[str],
            passage_ids: list[str],
            cache: Optional[dict] = None,
            update_cache: str = False,
            verbose: bool = False
    ):
        """
        Get scores for the given queries and passages using OpenAI API.
        Args:
            queries: List of query texts.
            passages: List of passage texts.
            query_ids: List of query IDs.
            passage_ids: List of passage IDs.
            cache: Optional dictionary to cache results.
            update_cache: Path to a CSV file to update with new results. If provided, the results will be written to the CSV file.
            verbose: If True, print debug information.
        """
        batch_qps = [(q, p) for q, p in zip(queries, passages)]
        batch_score, batch_prob = [], []
        for query, passage in batch_qps:
            messages = [
                {"role": "system",
                 "content": "You are an assistant that evaluates the relevance of passages to a given query."},
                {"role": "user", "content": self.prompt_template.format(query=query, passage=passage)}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
                temperature=0.0,
            )
            score = [-inf] * 4
            for t in response.choices[0].logprobs.content[0].top_logprobs:
                try:
                    idx = int(t.token)
                    if 0 <= idx < 4:
                        score[idx] = t.logprob
                except:
                    pass
            score = [math.exp(s) for s in score]
            score = [s / sum(score) for s in score]

            er = sum([i*s for i, s in enumerate(score)])
            pr = score.index(max(score))
            if self.score_type == "er":
                batch_score.append(er)
            elif self.score_type == "pr":
                batch_score.append(pr)
            else:
                raise ValueError(f"Invalid score type: {self.score_type}")
            batch_prob.append(score)

            if update_cache:
                os.makedirs(os.path.dirname(update_cache), exist_ok=True)
                file_exists = os.path.exists(update_cache)

                with open(update_cache, mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)

                    if not file_exists:
                        writer.writerow(["query_id", "passage_id", "er", "pr", "logit"])

                    writer.writerow([query_ids[0], passage_ids[0], er, pr, score])

            return batch_score, batch_prob


