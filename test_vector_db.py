from loaders.pdf_loader import PDFLoader
from services.vector_db import VectorDB

docs = PDFLoader.load_pdf(r"E:\RAG\Data\shridhar_resume_ (4).pdf")

db = VectorDB.create(docs)

print("Database Created Successfully")