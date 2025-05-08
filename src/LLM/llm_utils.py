import os

from src.LLM.ChatGPT import ChatGPT
from src.LLM.Llama import Llama

def handle_llm(llm_name):
    if llm_name is None:
        llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
    elif "chatgpt" in llm_name.lower():
        llm = ChatGPT(api_key=os.getenv("CHATGPT_API_KEY"))
    elif "llama" in llm_name.lower():
         llm = Llama(model_name=llm_name)
    else:
        raise Exception(f"Unknown llm name: {llm_name}")
    return llm