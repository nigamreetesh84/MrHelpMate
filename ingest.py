import os, re, json, argparse
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

# Configuration
PDF_PATH = os.path.join(os.getcwd(), "data", "Principal-Sample-Life-Insurance-Policy.pdf")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHROMA_DIR = "chroma_db"

def clean_text(s: str) -> str:
    """Basic cleaning to remove headers, footers, and normalize whitespace."""
    s = s.replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'This page left blank intentionally', '', s, flags=re.I)
    return s.strip()

def extract_pdf_chunks(pdf_path):
    """
    Loads PDF, splits into pages using LangChain loader, 
    and further splits into smaller text chunks.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # returns a list of Document objects (page-level)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ Successfully loaded {len(documents)} pages.")
    print(f"✅ Generated {len(chunks)} text chunks for embedding.")

    # Clean and extract text + metadata
    texts = [clean_text(chunk.page_content) for chunk in chunks]
    metadata = []
    for idx, chunk in enumerate(chunks):
        m = chunk.metadata.copy()
        m["chunk_id"] = f"chunk_{idx}"
        metadata.append(m)

    return texts, metadata

def build_embeddings(texts, metadata, model_name=EMBED_MODEL):
    """Encodes the text chunks into dense embeddings."""
    print(f"🔹 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"🔹 Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def upsert_chroma(texts, metadata, embeddings):
    """Stores embeddings in a persistent ChromaDB collection."""
    print("🔹 Initializing ChromaDB collection...")
    from chromadb import PersistentClient

    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("policy_chunks")

    ids = [m["chunk_id"] for m in metadata]
    collection.add(
        documents=texts,
        metadatas=metadata,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    print(f"✅ Stored {len(texts)} chunks into ChromaDB ({CHROMA_DIR})")


if __name__ == "__main__":
    texts, metadata = extract_pdf_chunks(PDF_PATH)
    embeddings = build_embeddings(texts, metadata)
    upsert_chroma(texts, metadata, embeddings)
    print("🎯 Ingest complete. Total chunks embedded:", len(texts))
