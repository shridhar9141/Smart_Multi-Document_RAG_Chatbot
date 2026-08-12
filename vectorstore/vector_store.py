from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import DB_PATH

from embeddings.embedding import get_embedding


def create_vectorstore(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding(),
        persist_directory=DB_PATH
    )

    return vector_db


def load_vectorstore():

    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=get_embedding()
    )