import csv
import json
import math
import os
import time
from collections import defaultdict
from typing import List, Optional

import requests

from src.LLM.llm import LLM


class OpenRouterLLM(LLM):
    """
    LLM client that scores passages via the OpenRouter chat-completion API.
    Supports expected relevance (ER) and pointwise relevance (PR) scoring modes.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        max_retries: int = 3,
        max_tokens: int = 16_000,
        score_mode: str = "expected_relevance",
        label_values: Optional[List[int]] = None,
    ):
        super().__init__(model_name=model_name)
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.score_mode = score_mode
        self.label_values = label_values if label_values is not None else [0, 1, 2, 3]

        if self.score_mode not in {"expected_relevance", "pointwise"}:
            raise ValueError("score_mode must be 'expected_relevance' or 'pointwise'")

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set."
            )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Call OpenRouter chat completion endpoint and return the response text.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"OpenRouter request failed after {self.max_retries} attempts."
                    ) from exc
                time.sleep(2 ** attempt)

        raise RuntimeError("OpenRouter request failed unexpectedly.")

    def _build_prompt(self, query: str, passages: List[str]) -> str:
        formatted_passages = "\n".join(
            [f"{idx + 1}. {text}" for idx, text in enumerate(passages)]
        )

        if self.score_mode == "expected_relevance":
            prompt_template = """
You are an assistant that evaluates the relevance of passages to a given query.
Return logits for each relevance label r_k ∈ {0, 1, 2, 3}. The output must be a JSON object
mapping each passage index (starting at 0) to an array of four logits ordered by the label value.

Query: {query}
Passages:
{passages}

Example format:
{{"logits": {{"0": [0.1, -0.2, 0.3, 0.0], "1": [0.5, 0.2, -0.1, -1.0]}}}}
"""
        else:
            prompt_template = """
You are an assistant that evaluates the relevance of passages to a given query.
Provide an integer relevance score on the scale:
0 = no relevance, 1 = somewhat related, 2 = partially answers, 3 = fully answers the query.
Return a JSON object mapping the passage index (starting at 0) to its relevance score.

Query: {query}
Passages:
{passages}

Example format: {{"scores": {{"0": 3, "1": 2, "2": 0}}}}
"""
        return prompt_template.format(query=query, passages=formatted_passages)

    def _parse_expected_relevance(self, response: str, num_passages: int) -> List[float]:
        try:
            parsed = json.loads(response)
            logits_map = parsed["logits"]
        except Exception as exc:
            print(f"Failed to parse logits from LLM response: {exc}")
            return [-1.0] * num_passages

        scores = []
        num_labels = len(self.label_values)
        for idx in range(num_passages):
            raw = logits_map.get(str(idx), [])
            if isinstance(raw, dict):
                logits = [float(raw.get(str(k), 0.0)) for k in range(num_labels)]
            else:
                logits = [float(raw[i]) if i < len(raw) else 0.0 for i in range(num_labels)]

            if not logits:
                scores.append(-1.0)
                continue

            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            denom = sum(exp_logits) if exp_logits else 1.0
            probs = [val / denom for val in exp_logits]
            expected = sum(prob * label for prob, label in zip(probs, self.label_values))
            scores.append(expected)

        return scores

    def _parse_pointwise(self, response: str, num_passages: int) -> List[float]:
        try:
            parsed = json.loads(response)
            scores_map = parsed["scores"]
            return [scores_map.get(str(i), -1) for i in range(num_passages)]
        except Exception as exc:
            print(f"Failed to parse pointwise scores from LLM response: {exc}")
            return [-1.0] * num_passages

    def get_score(
        self,
        query: str,
        passages: List[str],
        query_id: Optional[int] = None,
        passage_ids: Optional[List[int]] = None,
        cache: Optional[dict] = None,
        update_cache: Optional[str] = None,
    ) -> List[float]:
        """
        Score passages for a query, optionally using cached results and persisting updates.
        """
        cache_dict = cache if cache is not None else defaultdict(dict)

        if cache_dict and query_id is not None and passage_ids is not None:
            try:
                cached_scores = cache_dict[query_id]
                target_scores = [cached_scores[p_id] for p_id in passage_ids]
                return target_scores
            except KeyError:
                pass

        if query_id is not None and query_id not in cache_dict:
            cache_dict[query_id] = {}

        prompt = self._build_prompt(query, passages)
        response = self.generate(prompt)

        if self.score_mode == "expected_relevance":
            scores = self._parse_expected_relevance(response, len(passages))
        else:
            scores = self._parse_pointwise(response, len(passages))

        if (
            update_cache
            and query_id is not None
            and passage_ids is not None
            and len(passage_ids) == len(scores)
        ):
            cache_dict.setdefault(query_id, {})
            new_entries = []
            for p_id, score in zip(passage_ids, scores):
                previous = cache_dict[query_id].get(p_id)
                cache_dict[query_id][p_id] = score
                if previous is None:
                    new_entries.append([query_id, p_id, score])

            if new_entries:
                file_exists = os.path.exists(update_cache)
                with open(update_cache, mode="a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(["query_id", "passage_id", "score"])
                    writer.writerows(new_entries)

        return scores
