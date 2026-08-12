from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


class EmbeddingModel:

    @staticmethod
    def load_embeddings():

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )

        return embeddings