import abc
import csv
import os
from pathlib import Path

import torch
from filelock import FileLock
from unsloth import FastLanguageModel

from src.utils import logit2entropy


class LLM(abc.ABC):
    """
    Abstract class for Language Model (LLM)
    """
    def __init__(self, model_name, prompt_type=None, max_seq_length=None):
        self.label2idx = None
        self.label2weight = torch.tensor([0, 1, 2, 3])

        if max_seq_length is None:
            if prompt_type == 'fewshot':
                self.max_seq_length = 3072
            elif prompt_type == 'zeroshot':
                self.max_seq_length = 2048
            elif prompt_type == 'fewshotv2':
                self.max_seq_length = 2048
            elif prompt_type == 'rg4l':
                self.max_seq_length = 1024
            else:
                raise ValueError("prompt_type should be either 'fewshot' or 'zeroshot'")
        else:
            self.max_seq_length = max_seq_length

        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map='auto',
            token=os.environ['HF_TOKEN'],
            # token = "hf_...", # use onde if using gated models like meta-llama/Llama-2-7b-hf
        )
        FastLanguageModel.for_inference(self.model)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open(Path(cur_dir) / 'prompt' / f'{prompt_type}.txt', 'r', encoding='utf-8') as file:
            self.prompt_template = file.read()


    def get_score(
            self,
            queries: list[str],
            passages: list[str],
            query_ids: list[int],
            passage_ids: list[int],
            cache:dict=None,
            update_cache:str=False,
            score_type:str='er'):

        """
        Get the relevance score of each passage for a given query using a single LLM call.
        Args:
            queries: The query text.
            passages: A list of passage texts.
            query_ids: The ID of the query.
            passage_ids: A list of IDs for the passages.
            cache: A dictionary of cached results.
            update_cache: Path to the CSV file to update with new results. If provided, the results will be written to the CSV file.
            score_type: The type of score to return. Options are 'er' (expected relevance) or 'pr' (peack relevance).
        """
        if cache and (len(queries) == 1) and (len(passages) == 1):
            qid, pid = query_ids[0], passage_ids[0]
            if pid in cache.get(qid, {}):
                return [cache[qid][pid][score_type]], [logit2entropy(cache[qid][pid]['logit'])]

        batch_qps = [(q, p) for q, p in zip(queries, passages)]
        logit = self.get_logit(batch_qps)
        prob = torch.softmax(logit, dim=-1)
        entropy = torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
        er_scores = torch.sum(prob * self.label2weight, dim=-1).cpu().numpy()
        pr_scores = torch.argmax(prob, dim=-1).cpu().numpy()

        new_entries = [
            (query_ids[i], passage_ids[i], er_scores[i], pr_scores[i], logit[i].cpu().numpy().tolist())
            for i in range(len(batch_qps))
        ]

        if update_cache:
            os.makedirs(os.path.dirname(update_cache), exist_ok=True)
            file_exists = os.path.exists(update_cache)

            lock_path = update_cache + ".lock"
            with FileLock(lock_path):
                with open(update_cache, mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)

                    if not file_exists:
                        writer.writerow(["query_id", "passage_id", "er", "pr", "logit"])

                    writer.writerows(new_entries)

        if score_type=='er':
            return er_scores, entropy
        elif score_type=='pr':
            return pr_scores, entropy
        else:
            raise ValueError("score_type should be either 'er' or 'pr'")

    def get_logit(self, batch_qps):

        batch_prompts = []
        for query, passage in batch_qps:
            messages = [
                {"role": "system",
                 "content": "You are an assistant that evaluates the relevance of passages to a given query."},
                {"role": "user", "content": self.prompt_template.format(query=query, passage=passage)}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(prompt)

        batch_inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
        with torch.no_grad():
            model_outputs = self.model(**batch_inputs)
        logit = model_outputs.logits[:, -1, [self.label2idx[0], self.label2idx[1], self.label2idx[2], self.label2idx[3]]].cpu()
        return logit