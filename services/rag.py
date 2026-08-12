from chains.rag_chain import Level3RAGPipeline
from config import TOP_K_RETRIEVAL, TOP_N_RERANK


class RAGService:

    @staticmethod
    def create_chain(documents, vector_db, top_k=TOP_K_RETRIEVAL, top_n=TOP_N_RERANK):
        """
        Creates a Level 3 RAG Chain integrating Conversational Memory,
        Hybrid Retrieval (BM25 + Vector Search), and Cross-Encoder Reranking.
        """
        return Level3RAGPipeline(
            documents=documents,
            vector_db=vector_db,
            top_k=top_k,
            top_n=top_n
        )
