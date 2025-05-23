from src.LLM.llm import LLM

class Mistral(LLM):

    def __init__(self, model_name, prompt_type, score_type):
        super().__init__(model_name, prompt_type, score_type)

        self.label2idx = {
            0: 1048,
            1: 1049,
            2: 1050,
            3: 1051
        }
