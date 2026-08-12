from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DB_DIRECTORY = os.getenv("DB_DIRECTORY", "chroma_db")

# Level 3 Configurations
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "3"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
HYBRID_WEIGHTS = [0.5, 0.5]