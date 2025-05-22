from pathlib import Path

import torch
import unsloth
from unsloth.chat_templates import get_chat_template

from src.LLM.llm import LLM

class Llama3(LLM):

    def __init__(self, model_name, prompt_type):
        super().__init__(model_name, prompt_type)

        self.tokenizer = get_chat_template(self.tokenizer, chat_template="llama-3.1")
        self.label2idx = {
            0: 15,
            1: 16,
            2: 17,
            3: 18
        }