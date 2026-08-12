from typing import List, Sequence, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, TOP_N_RERANK


class CrossEncoderReranker:
    """
    Reranker service that uses sentence_transformers.CrossEncoder to re-score
    candidate documents and pick top_n most relevant documents.
    """

    _instance = None
    _model = None

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    @classmethod
    def get_instance(cls, model_name: str = RERANKER_MODEL):
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = TOP_N_RERANK
    ) -> List[Document]:
        """
        Re-ranks a list of candidate documents against a query using CrossEncoder.
        
        Args:
            query: The user input query string.
            documents: Candidate documents (e.g. Top 10 retrieved).
            top_n: Number of top documents to return after reranking.
            
        Returns:
            Top_n documents sorted by cross-encoder relevance score.
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = []
        for doc, score in doc_score_pairs[:top_n]:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)}
            )
            reranked_docs.append(doc_copy)

        return reranked_docs
