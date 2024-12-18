from enum import Enum

class Models(Enum):
    GPT35TURBO = ('gpt', 'gpt-3.5-turbo')
    GPT4 = ('gpt', 'gpt-4')
    LLAMA3 = ('llama', 'llama3')
    LLAMA32 = ('llama', 'llama3.2')
    TINYLLAMA = ('llama', 'tinyllama')


def get_model(model: Models):
    if model.value[0] == 'gpt':
        # requires 'OPENAI_API_KEY' Environment variable to be set
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model.value[1],
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm
    if model.value[0] == 'llama':
        # requires local llama instance with chosen version
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=model.value[1],
            temperature=0
        )
        return llm