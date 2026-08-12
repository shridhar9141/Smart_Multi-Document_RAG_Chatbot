import os

from loaders.pdf_loader import load_pdf

from vectorstore.vector_store import create_vectorstore

from config import DB_PATH


def initialize_database():

    if not os.path.exists(DB_PATH):

        docs = load_pdf("E:\RAG\Data")

        create_vectorstore(docs)