import torch

from src.LLM.llm import LLM

class Qwen(LLM):

    def __init__(self, model_name, prompt_type, score_type):
        super().__init__(model_name, prompt_type, score_type)

        self.label2idx = {
            0: 15,
            1: 16,
            2: 17,
            3: 18
        }

    def get_logit(self, batch_qps):

        batch_prompts = []
        for query, passage in batch_qps:
            messages = [
                {"role": "system",
                 "content": "You are an assistant that evaluates the relevance of passages to a given query."},
                {"role": "user", "content": self.prompt_template.format(query=query, passage=passage)}
            ]
            # Note: Disabling thinking for Qwen
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
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