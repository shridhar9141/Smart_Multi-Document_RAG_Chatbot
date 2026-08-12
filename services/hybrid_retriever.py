from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config import TOP_K_RETRIEVAL, HYBRID_WEIGHTS


class HybridRetrieverFactory:

    @staticmethod
    def create_hybrid_retriever(
        documents: List[Document],
        vector_db,
        top_k: int = TOP_K_RETRIEVAL,
        weights: Optional[List[float]] = None
    ) -> EnsembleRetriever:
        """
        Creates a Hybrid Retriever combining BM25 keyword search and Vector search.
        
        Args:
            documents: List of chunked Document objects.
            vector_db: VectorDB / Chroma instance.
            top_k: Number of documents to retrieve per retriever.
            weights: List of weights [bm25_weight, vector_weight].
            
        Returns:
            EnsembleRetriever combining BM25 and Vector search.
        """
        if weights is None:
            weights = HYBRID_WEIGHTS

        # 1. BM25 Keyword Retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = top_k

        # 2. Vector Store Retriever
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

        # 3. Ensemble Retriever combining Keyword + Vector
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )

        return ensemble_retriever
