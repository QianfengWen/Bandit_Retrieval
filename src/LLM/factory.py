import os

def handle_llm(llm_name, prompt_type=None):
    if llm_name is None:
        raise NotImplementedError ("No LLM name provided")
        from .chatgpt import ChatGPT
        llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    elif "chatgpt" in llm_name.lower():
        raise NotImplementedError ("ChatGPT is not supported in this version")
        from .chatgpt import ChatGPT
        llm = ChatGPT(api_key=os.getenv("CHATGPT_API_KEY"))
    elif "llama" in llm_name.lower():
        from .llama import Llama
        llm = Llama(model_name=llm_name, prompt_type=prompt_type)
    else:
        raise Exception(f"Unknown llm name: {llm_name}")
    return llm
