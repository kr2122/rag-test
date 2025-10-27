from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
sentence_embeddings = model.encode("KNN是什么算法")
print(sentence_embeddings)
