from langchain_groq import ChatGroq

from config import GROQ_API_KEY, LLM_MODEL


def load_llm():

    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )