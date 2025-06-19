from src.LLM.llm import LLM

class Llama4(LLM):

    def __init__(self, model_name, prompt_type, score_type):
        super().__init__(model_name, prompt_type, score_type)

        self.label2idx = {
            0: 28,
            1: 29,
            2: 30,
            3: 31
        }