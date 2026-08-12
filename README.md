# ⚡ Smart Multi-Document RAG Chatbot

An AI-powered RAG chatbot built with Python, Streamlit, LangChain, ChromaDB, and Groq. It retrieves relevant information from multiple PDF documents using hybrid search (keyword BM25 + dense vector embeddings) and cross-encoder reranking to deliver accurate, contextual answers.

---

## 🌟 Key Features

- 📑 **Multi-Document Ingestion**: Upload and index multiple PDF documents dynamically with PyPDF loader and character/token text splitters.
- 🔍 **Hybrid Retrieval**: Combines semantic dense vector search (ChromaDB + Sentence Transformers) with sparse keyword search (BM25) for high recall and precision.
- 🎯 **Cross-Encoder Reranking**: Re-ranks initial retrieved candidate chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2` to deliver top relevant context to the LLM.
- 💬 **Interactive Streamlit UI**: User-friendly chat interface for querying documents, viewing reference context sources, and inspecting evaluation benchmarks.
- ⚡ **Groq LLM Acceleration**: Supercharged inference speed utilizing Groq's high-speed API.
- 📊 **Evaluation Framework**: Built-in automated evaluator to measure context precision, recall, and answer relevance across synthetic query datasets.

---

## 🏗️ Architecture & Pipeline

```
[ PDF Documents ] ──► [ PyPDF Loader & Text Splitter ]
                                │
                                ▼
               ┌─────────────────────────────────┐
               │         Hybrid Retriever        │
               │  ┌───────────────────────────┐  │
               │  │ ChromaDB Vector Search    │  │
               │  │ (Dense Semantic Search)   │  │
               │  └─────────────┬─────────────┘  │
               │                │                │
               │  ┌─────────────┴─────────────┐  │
               │  │ BM25 Sparse Search        │  │
               │  │ (Keyword Matching)        │  │
               │  └─────────────┬─────────────┘  │
               └────────────────┼────────────────┘
                                │ Candidate Chunks
                                ▼
                  ┌───────────────────────────┐
                  │   Cross-Encoder Reranker  │
                  └─────────────┬─────────────┘
                                │ Top-N Reranked Context
                                ▼
                  ┌───────────────────────────┐
                  │    Groq LLM RAG Chain     │
                  └─────────────┬─────────────┘
                                │
                                ▼
                        [ Final Answer ]
```

---

## 🛠️ Project Structure

```
Smart_Multi-Document_RAG_Chatbot/
├── app.py                      # Streamlit Web Application entry point
├── config.py                   # Central configuration & environment variables
├── rag.py                      # Core RAG orchestration logic
├── run_evaluation.py           # Evaluation pipeline executor
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── chains/                     # LangChain LCEL chain definitions
├── llm/                        # LLM provider initialization (Groq)
├── loaders/                    # PDF document loader and preprocessors
├── prompts/                    # System prompts and prompt templates
├── services/                   # Core RAG services (retrieval, reranker, evaluator)
├── tests/                      # Unit and integration tests
├── utils/                      # Helper utilities
└── vectorstore/                # ChromaDB vector database manager
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API Key](https://console.groq.com/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shridhar9141/Smart_Multi-Document_RAG_Chatbot.git
   cd Smart_Multi-Document_RAG_Chatbot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables**:
   Copy `.env.example` to `.env` and fill in your Groq API Key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```ini
   GROQ_API_KEY=your_actual_groq_api_key
   MODEL_NAME=llama-3.3-70b-versatile
   ```

---

## 🖥️ Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

1. Upload one or more PDF files via the sidebar.
2. Click **Process Documents** to chunk, embed, and build the hybrid retriever index.
3. Start asking questions in the chat interface!

---

## 📈 Running Benchmarks & Evaluation

To run the automated RAG evaluation metrics:

```bash
python run_evaluation.py
```

This will run evaluation queries and save detailed metric results to `evaluation_results.json`.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.

