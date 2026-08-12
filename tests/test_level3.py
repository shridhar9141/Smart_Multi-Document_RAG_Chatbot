import os
import pytest
from langchain_core.documents import Document
from services.embeddings import EmbeddingModel
from services.vector_db import VectorDB
from services.hybrid_retriever import HybridRetrieverFactory
from services.reranker import CrossEncoderReranker
from chains.rag_chain import Level3RAGPipeline
from services.evaluator import RAGEvaluator


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Shridhar is a software developer with experience in React and Node.js.", metadata={"source": "resume.pdf"}),
        Document(page_content="Shridhar studied Electronics and Communication Engineering at KIT, CGPA 8.35.", metadata={"source": "resume.pdf"}),
        Document(page_content="Shridhar solved 100+ DSA problems on GeeksforGeeks and has a SQL certification.", metadata={"source": "resume.pdf"}),
        Document(page_content="The weather in Bangalore is pleasant during evening times.", metadata={"source": "random.pdf"}),
        Document(page_content="Python Essentials and AI certification completed by Shridhar from Cisco.", metadata={"source": "resume.pdf"}),
    ]


def test_hybrid_retriever(sample_docs):
    embeddings = EmbeddingModel.load_embeddings()
    vector_db = VectorDB.create(sample_docs, embedding_function=embeddings)
    
    hybrid = HybridRetrieverFactory.create_hybrid_retriever(sample_docs, vector_db, top_k=4)
    results = hybrid.invoke("Shridhar DSA problems")
    
    assert len(results) > 0
    contents = [d.page_content for d in results]
    assert any("GeeksforGeeks" in c for c in contents)


def test_cross_encoder_reranker(sample_docs):
    reranker = CrossEncoderReranker.get_instance()
    reranked = reranker.rerank("What degree did Shridhar pursue?", sample_docs, top_n=2)
    
    assert len(reranked) == 2
    assert "Electronics and Communication" in reranked[0].page_content
    assert "rerank_score" in reranked[0].metadata


def test_rag_evaluator_metrics(sample_docs):
    embeddings = EmbeddingModel.load_embeddings()
    vector_db = VectorDB.create(sample_docs, embedding_function=embeddings)
    pipeline = Level3RAGPipeline(sample_docs, vector_db, top_k=5, top_n=3)

    test_dataset = [
        {
            "id": 1,
            "question": "Where did Shridhar study?",
            "expected_answer": "Kalpataru Institute of Technology (KIT)",
            "ground_truth_keywords": ["KIT", "Electronics"]
        }
    ]

    eval_result = RAGEvaluator.evaluate(pipeline, test_dataset)
    assert eval_result["total_questions"] == 1
    assert "retrieval_accuracy" in eval_result
    assert "average_precision" in eval_result
    assert "average_recall" in eval_result
    assert "average_answer_correctness" in eval_result
