from services.embeddings import EmbeddingModel

embedding = EmbeddingModel.load_embeddings()

vector = embedding.embed_query("What is Artificial Intelligence?")

print(len(vector))