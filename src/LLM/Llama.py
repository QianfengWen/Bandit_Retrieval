import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import csv
import os
import re
from collections import defaultdict
from filelock import FileLock

from src.LLM.llm import LLM


class Llama(LLM):
    def __init__(self, model_name):
        self.max_seq_length = 1024  # Choose any! We auto support RoPE Scaling internally!
        self.max_gen_length = 32
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map='auto',
            token=os.environ['HF_TOKEN']
            # token = "hf_...", # use onde if using gated models like meta-llama/Llama-2-7b-hf
        )
        FastLanguageModel.for_inference(self.model)
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="llama-3.1")
        # self.tokenizer.padding_side = 'left'

        self.prompt_template = """
Given a query and a list of passages, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query,
1 = represents that the passage seems related to the query but does not answer it,
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passages}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code or in result. Respond only with the final score in the following format:
## Final score (O): X
""".strip()

    ## Likely intent (M): X
    ## Trustworthy (T): X

    def generate(self):
        pass

    def get_score(self, queries, passages, query_ids=None, passage_ids=None, cache=None, update_cache=False):
        generated_score = []
        if cache:
            filtered_query, filtered_passages, filtered_query_ids, filtered_passages_ids = [], [], [], []
            for q, p, qid, pid in zip(queries, passages, query_ids, passage_ids):
                if pid in cache.get(qid, {}):
                  if len(passages) > 1:
                      raise ValueError ("Batched score should not contain cached output")
                  else:
                      return [cache[qid][pid]]
                else:
                    filtered_query.append(q)
                    filtered_passages.append(p)
                    filtered_query_ids.append(qid)
                    filtered_passages_ids.append(pid)

            queries, passages, query_ids, passage_ids = filtered_query, filtered_passages, filtered_query_ids, filtered_passages_ids
            if len(queries) == 0:
                return generated_score
        else:
            cache = defaultdict(dict)

        batch_qps = [(q, p) for q, p in zip(queries, passages)]
        batch_prompts = []

        for query, passage in batch_qps:
            messages = [
                {"role": "system",
                 "content": "You are an assistant that evaluates the relevance of passages to a given query."},
                {"role": "user", "content": self.prompt_template.format(query=query, passages=passage)}
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
            max_length=self.max_seq_length-self.max_gen_length,
        )

        batch_inputs = {k: v.to("cuda") for k, v in batch_inputs.items()}

        batch_outputs = self.model.generate(
            **batch_inputs,
            max_new_tokens = self.max_gen_length,
            use_cache = False,
            do_sample=False
            )

        batch_decoded = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

        new_entries = []
        for i, decoded in enumerate(batch_decoded):
            match = re.search(r"(?:##\s*)?final score\s*\(O\)\s*:\s*(\d+)", decoded, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                new_entries.append((query_ids[i], passage_ids[i], score))
            else:
                matches = list(re.finditer(r'(?<!\d)([0-3])(?!\d)', decoded))
                if matches:
                    score = int(matches[-1].group(1))
                    new_entries.append((query_ids[i], passage_ids[i], score))
                else:
                    score = None
                    print(f"\n\n\n\n>> The score was not parsed correctly for query {query_ids[i]} and passage {passage_ids[i]}. ")
                    print(f">>> input len: {len(batch_inputs['input_ids'][i])}")
                    print(f">>> The total output is {decoded}.")

            generated_score.append(score)

        if update_cache:
            os.makedirs(os.path.dirname(update_cache), exist_ok=True)
            file_exists = os.path.exists(update_cache)

            lock_path = update_cache + ".lock"
            with FileLock(lock_path):
                with open(update_cache, mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)

                    if not file_exists:
                        writer.writerow(["query_id", "passage_id", "score"])

                    writer.writerows(new_entries)

        return generated_score

