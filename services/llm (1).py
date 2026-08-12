from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME


class LLM:

    @staticmethod
    def load():
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

        return ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0
        )