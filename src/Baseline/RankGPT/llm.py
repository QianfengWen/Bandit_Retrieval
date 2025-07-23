import os

from unsloth import FastLanguageModel
import torch


class RankQwen():
    def __init__(self, model_name='unsloth/Qwen3-14B-unsloth-bnb-4bit'):
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map='auto',
            token=os.environ['HF_TOKEN'],
            # token = "hf_...", # use onde if using gated models like meta-llama/Llama-2-7b-hf
        )
        FastLanguageModel.for_inference(self.model)

    def chat(self, message):
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        batch_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model_outputs = self.model.generate(
                **batch_inputs,
                temperature=0,
                max_new_tokens=512,
                do_sample=False)

        seq = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0].split('</think>\n\n')[-1]

        return seq
