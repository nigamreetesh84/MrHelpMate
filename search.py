import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from cache import SimpleCache

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Initialize Chroma client (new API)
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("policy_chunks")

# Initialize models and cache
embed_model = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(CROSS_ENCODER)
cache = SimpleCache()


# load cross-encoder
from sentence_transformers import CrossEncoder
reranker = CrossEncoder(CROSS_ENCODER)

def retrieve(query, top_k=20):
    cached = cache.get(query)
    if cached:
        return cached
    q_emb = embed_model.encode(query)
    # results = col.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)

    # results contains documents, metadatas, ids, distances
    items = []
    for doc, meta, id_ in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
        items.append({"id": id_, "text": doc, "meta": meta})
    cache.set(query, items)
    return items

def rerank(query, items, top_n=3):
    pairs = [(query, it["text"]) for it in items]
    scores = reranker.predict([p[1] for p in pairs], [p[0] for p in pairs]) if False else reranker.predict(pairs)
    # CrossEncoder from sentence_transformers accepts (query, doc) pairs
    scored = []
    for it, s in zip(items, scores):
        scored.append({**it, "score": float(s)})
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

if __name__ == "__main__":
    q = "What is the scheduled benefit for all members?"
    items = retrieve(q)
    top3 = rerank(q, items)
    for i,r in enumerate(top3,1):
        print(i, r["id"], r["score"])
        print(r["text"][:400])
        print("----")
