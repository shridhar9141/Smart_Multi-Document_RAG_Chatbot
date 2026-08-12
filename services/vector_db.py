from langchain_chroma import Chroma
from services.embeddings import EmbeddingModel
from langchain_core.embeddings import Embeddings
from config import DB_DIRECTORY
import os


class VectorDB:

    @staticmethod
    def load(embedding_function: Embeddings):

        print("=" * 60)
        print("Current Working Directory :", os.getcwd())
        print("Database Path :", os.path.abspath(DB_DIRECTORY))
        print("Database Exists :", os.path.exists(DB_DIRECTORY))
        print("=" * 60)

        db = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=embedding_function
        )

        print("Vector Database Loaded Successfully")

        return db

    @staticmethod
    def create(documents, embedding_function: Embeddings, persist_directory=None):

        print("=" * 60)
        print("Creating Vector Database from documents")
        print("=" * 60)

        if persist_directory:
            db = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                persist_directory=persist_directory
            )
        else:
            db = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function
            )

        print("Vector Database Created Successfully")

        return db

    