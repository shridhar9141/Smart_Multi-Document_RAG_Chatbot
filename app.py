import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from loaders.pdf_loader import PDFLoader
from services.embeddings import EmbeddingModel
from services.vector_db import VectorDB
from services.rag import RAGService
from services.evaluator import RAGEvaluator, EVALUATION_DATASET
from config import GROQ_API_KEY, TOP_K_RETRIEVAL, TOP_N_RERANK

UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Level 3 PDF RAG System",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

def save_uploaded_pdf(uploaded_file):
    output_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path


def load_documents_from_uploads(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        pdf_path = save_uploaded_pdf(uploaded_file)
        loaded_docs = PDFLoader.load_pdf(pdf_path)
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
        documents.extend(loaded_docs)
    return documents


def get_langchain_chat_history():
    history = []
    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:
    st.title("⚡ Level 3 PDF RAG")
    st.markdown("**Features:** Memory | Hybrid Search | Reranker | Evaluation")
    st.markdown("---")

    st.write("### Settings")

    if st.button("🧹 Clear uploaded PDFs"):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        st.session_state.uploaded_files = None
        st.success("Uploaded files cleared.")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploaded_files"
    )

    top_k = st.slider(
        "Top-K Hybrid Retrieval (BM25 + Vector)",
        min_value=2,
        max_value=20,
        value=TOP_K_RETRIEVAL,
        step=1
    )

    top_n = st.slider(
        "Top-N Cross-Encoder Reranking",
        min_value=1,
        max_value=top_k,
        value=min(TOP_N_RERANK, top_k),
        step=1
    )

    if st.button("🗑️ Clear Chat & Memory"):
        st.session_state.messages = []
        st.success("Conversation history cleared.")

    st.markdown("---")

    if uploaded_files:
        st.success(f"{len(uploaded_files)} PDF(s) uploaded")
    else:
        st.info("Upload PDFs or default dataset will be used.")

    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY is missing. Set it in your .env file.")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# Prepare Retrieval Chain & Document Loading
# ---------------------------------------------------

pipeline = None
documents = []
vector_db = None

embedding_model = EmbeddingModel.load_embeddings()

if uploaded_files:
    documents = load_documents_from_uploads(uploaded_files)
else:
    # Fallback to default PDF if available
    default_pdf = os.path.join(os.getcwd(), "Data", "shridhar_resume_ (4).pdf")
    if os.path.exists(default_pdf):
        loaded_docs = PDFLoader.load_pdf(default_pdf)
        for doc in loaded_docs:
            doc.metadata["source"] = os.path.basename(default_pdf)
        documents.extend(loaded_docs)

if documents:
    vector_db = VectorDB.create(documents, embedding_function=embedding_model, persist_directory=None)
    pipeline = RAGService.create_chain(
        documents=documents,
        vector_db=vector_db,
        top_k=top_k,
        top_n=top_n
    )

# ---------------------------------------------------
# Main UI Tabs
# ---------------------------------------------------

tab_chat, tab_eval = st.tabs(["💬 Level 3 RAG Chatbot", "📊 Retrieval & Answer Evaluation"])

# ---------------------------------------------------
# Tab 1: Chatbot Interface
# ---------------------------------------------------

with tab_chat:
    st.title("📄 PDF Question Answering with Memory & Reranking")
    st.caption("Ask questions about your uploaded PDFs. Chatbot remembers context and reranks documents.")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "retrieved_details" in message:
                with st.expander("🔍 Retrieval & Reranker Diagnostics"):
                    st.markdown(message["retrieved_details"])

    # User Input
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving (Hybrid BM25 + Vector) & Reranking (Cross-Encoder)..."):
                try:
                    if pipeline is None:
                        answer = "Please upload at least one PDF file or ensure GROQ_API_KEY is set."
                        retrieved_details = ""
                    else:
                        chat_history = get_langchain_chat_history()[:-1]  # Exclude current question
                        response = pipeline.invoke({
                            "question": question,
                            "chat_history": chat_history
                        })

                        answer = response.get("answer", "No answer generated.")
                        reranked_docs = response.get("source_documents", [])
                        retrieved_candidates = response.get("retrieved_documents", [])
                        standalone_q = response.get("standalone_query", question)

                        # Diagnostics string
                        retrieved_details = f"**Standalone Query (Contextualized):** {standalone_q}\n\n"
                        retrieved_details += f"**Step 1: Hybrid Retrieval ({len(retrieved_candidates)} Candidates):**\n"
                        for idx, doc in enumerate(retrieved_candidates, 1):
                            retrieved_details += f"- Candidate {idx} [{doc.metadata.get('source', 'Doc')}]: `{doc.page_content[:100]}...`\n"

                        retrieved_details += f"\n**Step 2: Cross-Encoder Reranking (Top {len(reranked_docs)} Selected):**\n"
                        for idx, doc in enumerate(reranked_docs, 1):
                            score = doc.metadata.get("rerank_score", "N/A")
                            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                            retrieved_details += f"- Rank {idx} (Score: `{score_str}`): `{doc.page_content[:150]}...`\n"

                    st.markdown(answer)

                    if reranked_docs:
                        source_files = sorted({doc.metadata.get("source", "Unknown") for doc in reranked_docs})
                        st.markdown(f"**Sources:** {', '.join(source_files)}")

                    with st.expander("🔍 Retrieval & Reranker Diagnostics"):
                        st.markdown(retrieved_details)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "retrieved_details": retrieved_details
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    print(e)

# ---------------------------------------------------
# Tab 2: Evaluation Benchmark Interface
# ---------------------------------------------------

with tab_eval:
    st.header("📊 Level 3 Retrieval Evaluation Benchmark")
    st.markdown("Measures **Retrieval Accuracy (Hit%)**, **Precision**, **Recall**, and **Answer Correctness** across 10 benchmark questions.")

    if pipeline is None:
        st.warning("Please upload a PDF or ensure default documents are present to run evaluation.")
    else:
        if st.button("🚀 Run 10-Question Evaluation Benchmark"):
            with st.spinner("Evaluating 10 benchmark questions across Level 3 RAG Pipeline..."):
                eval_results = RAGEvaluator.evaluate(pipeline, EVALUATION_DATASET)

                # Metrics Summary Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval Accuracy", f"{eval_results['retrieval_accuracy']}%")
                col2.metric("Average Precision", f"{eval_results['average_precision']:.4f}")
                col3.metric("Average Recall", f"{eval_results['average_recall']:.4f}")
                col4.metric("Answer Correctness", f"{eval_results['average_answer_correctness']:.4f}")

                st.markdown("---")
                st.subheader("Detailed Evaluation Results")

                table_data = []
                for res in eval_results["per_question_results"]:
                    table_data.append({
                        "ID": res["id"],
                        "Question": res["question"],
                        "Hit": "✅ Yes" if res["hit"] else "❌ No",
                        "Precision": res["precision"],
                        "Recall": res["recall"],
                        "Answer Score": res["answer_correctness"],
                        "Generated Answer": res["generated_answer"]
                    })

                st.dataframe(table_data, use_container_width=True)