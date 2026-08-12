from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from prompts.prompt import RAG_PROMPT, CONTEXTUALIZE_Q_PROMPT
from services.llm import LLM
from services.hybrid_retriever import HybridRetrieverFactory
from services.reranker import CrossEncoderReranker
from config import TOP_K_RETRIEVAL, TOP_N_RERANK


class Level3RAGPipeline:
    """
    Level 3 RAG Pipeline integrating:
    1. Conversational Memory (Contextualizing query using chat history)
    2. Hybrid Retrieval (BM25 Keyword + Chroma Vector Search)
    3. Cross-Encoder Reranking (Top 10 candidate docs -> Top 3 re-scored docs)
    4. LLM Generation
    """

    def __init__(
        self,
        documents: List[Document],
        vector_db,
        top_k: int = TOP_K_RETRIEVAL,
        top_n: int = TOP_N_RERANK
    ):
        self.documents = documents
        self.vector_db = vector_db
        self.top_k = top_k
        self.top_n = top_n

        # Hybrid Retriever (BM25 + Vector)
        self.hybrid_retriever = HybridRetrieverFactory.create_hybrid_retriever(
            documents=documents,
            vector_db=vector_db,
            top_k=top_k
        )

        # Cross-Encoder Reranker
        self.reranker = CrossEncoderReranker.get_instance()

        # LLM
        self.llm = LLM.load()

    def _contextualize_query(self, query: str, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return query

        contextualize_chain = CONTEXTUALIZE_Q_PROMPT | self.llm | StrOutputParser()
        standalone_query = contextualize_chain.invoke({
            "chat_history": chat_history,
            "input": query
        })
        return standalone_query

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes Level 3 pipeline:
        Question -> Contextualization -> Hybrid Retrieval (Top 10) -> Reranker (Top 3) -> LLM Answer
        """
        query = inputs.get("question") or inputs.get("input") or ""
        chat_history = inputs.get("chat_history", [])

        # Step 1: Conversational Memory - Query Reformulation
        standalone_query = self._contextualize_query(query, chat_history)

        # Step 2: Hybrid Retrieval (Retrieves Top K, default 10)
        candidate_docs = self.hybrid_retriever.invoke(standalone_query)

        # Step 3: Cross-Encoder Reranking (Scores & selects Top N, default 3)
        reranked_docs = self.reranker.rerank(standalone_query, candidate_docs, top_n=self.top_n)

        # Format context string
        context_str = "\n\n".join([doc.page_content for doc in reranked_docs])

        # Step 4: LLM Answer Generation
        rag_chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = rag_chain.invoke({
            "context": context_str,
            "chat_history": chat_history,
            "input": query
        })

        return {
            "answer": answer,
            "query": query,
            "standalone_query": standalone_query,
            "source_documents": reranked_docs,
            "retrieved_documents": candidate_docs
        }


def get_rag_chain(documents, vector_db, top_k=TOP_K_RETRIEVAL, top_n=TOP_N_RERANK):
    return Level3RAGPipeline(documents=documents, vector_db=vector_db, top_k=top_k, top_n=top_n)