import os
import json
from loaders.pdf_loader import PDFLoader
from services.embeddings import EmbeddingModel
from services.vector_db import VectorDB
from services.rag import RAGService
from services.evaluator import RAGEvaluator, EVALUATION_DATASET


def main():
    print("=" * 80)
    print("LEVEL 3 RAG EVALUATION BENCHMARK")
    print("=" * 80)

    pdf_path = os.path.join(os.getcwd(), "Data", "shridhar_resume_ (4).pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: Default PDF not found at {pdf_path}")
        return

    print(f"Loading document: {pdf_path}")
    documents = PDFLoader.load_pdf(pdf_path)
    for doc in documents:
        doc.metadata["source"] = os.path.basename(pdf_path)

    print(f"Loaded {len(documents)} document chunk(s).")

    print("Initializing Embedding Model and Vector Database...")
    embedding_model = EmbeddingModel.load_embeddings()
    vector_db = VectorDB.create(documents, embedding_function=embedding_model, persist_directory=None)

    print("Building Level 3 RAG Pipeline (Hybrid Retrieval + Reranker)...")
    rag_pipeline = RAGService.create_chain(
        documents=documents,
        vector_db=vector_db,
        top_k=10,
        top_n=3
    )

    print("Running evaluation on 10 benchmark questions...")
    eval_results = RAGEvaluator.evaluate(rag_pipeline, EVALUATION_DATASET)

    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    print(f"Total Questions           : {eval_results['total_questions']}")
    print(f"Retrieval Accuracy (Hit%) : {eval_results['retrieval_accuracy']}%")
    print(f"Average Precision         : {eval_results['average_precision']}")
    print(f"Average Recall            : {eval_results['average_recall']}")
    print(f"Average Answer Correctness: {eval_results['average_answer_correctness']}")
    print("=" * 80)

    print("\nPER-QUESTION DETAILED RESULTS:")
    print("-" * 100)
    header = f"{'ID':<4} | {'Question':<45} | {'Hit':<5} | {'Prec':<6} | {'Recall':<6} | {'Ans Score':<9}"
    print(header)
    print("-" * 100)

    for item in eval_results["per_question_results"]:
        q_text = item['question'][:42] + "..." if len(item['question']) > 45 else item['question']
        hit_str = "YES" if item['hit'] else "NO"
        print(f"{item['id']:<4} | {q_text:<45} | {hit_str:<5} | {item['precision']:<6} | {item['recall']:<6} | {item['answer_correctness']:<9}")

    print("-" * 100)

    # Save output JSON artifact
    output_path = os.path.join(os.getcwd(), "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nFull evaluation report saved to {output_path}")


if __name__ == "__main__":
    main()
