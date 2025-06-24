
def handle_llm(llm_name, prompt_type=None, score_type=None):
    if llm_name is None:
        raise NotImplementedError ("No LLM name provided")

    elif "chatgpt" in llm_name.lower():
        raise NotImplementedError ("ChatGPT is not supported in this version")

    elif llm_name == 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit':
        from .llama3 import Llama3
        llm = Llama3(model_name=llm_name, prompt_type=prompt_type, score_type=score_type)

    elif llm_name == "unsloth/Qwen3-14B-unsloth-bnb-4bit":
        from .qwen import Qwen
        llm = Qwen(model_name=llm_name, prompt_type=prompt_type, score_type=score_type)

    elif llm_name == "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit":
        from .mistral import Mistral
        llm = Mistral(model_name=llm_name, prompt_type=prompt_type, score_type=score_type)

    elif llm_name == "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit":
        from .llama4 import Llama4
        llm = Llama4(model_name=llm_name, prompt_type=prompt_type, score_type=score_type)
    else:
        raise Exception(f"Unknown llm name: {llm_name}")
    return llm
