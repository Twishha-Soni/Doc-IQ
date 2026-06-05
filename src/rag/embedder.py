import chromadb
from sentence_transformers import SentenceTransformer
from .chunker import Chunk
from .config import EMBEDDING_MODEL

# Load the embedding model once at module level.
# This is like a Spring @Bean singleton — created once, reused everywhere.
# The first call downloads the model; subsequent calls load from local cache.
_model = SentenceTransformer(EMBEDDING_MODEL)

# Create a local ChromaDB client.
# "persistent" means it saves to disk at the given path — data survives restarts.
# This is like configuring a DataSource in Spring — one client, reused everywhere.
_client = chromadb.PersistentClient(path="./chroma_store")

def _get_collection():
    """
    Get or create the ChromaDB collection.
    A 'collection' is like a database table — it holds all our chunk vectors.
    get_or_create means it's safe to call repeatedly without duplicating data.
    """
    return _client.get_or_create_collection(name="documents")

def embed_and_store(chunks: list[Chunk]) -> int:
    """
    Convert chunks to vectors and store them in ChromaDB.
    Returns the number of chunks stored.
    """
    collection = _get_collection()

    # Extract just the text from each chunk for embedding
    texts = [chunk.text for chunk in chunks]

    # Convert all texts to vectors in one batch call.
    # .tolist() converts the numpy array the model returns into a plain Python list,
    # which is what ChromaDB expects.
    vectors = _model.encode(texts).tolist()

    # Build the three parallel lists ChromaDB needs:
    # ids       → unique string ID for each chunk
    # documents → the raw text (returned during retrieval)
    # metadatas → extra info per chunk (used for citations later)
    ids = [f"{chunk.filename}::chunk_{chunk.index}" for chunk in chunks]
    metadatas = [{"filename": chunk.filename, "index": chunk.index} for chunk in chunks]
    
    # upsert = insert if new, update if ID already exists.
    # Safe to call multiple times on the same document — won't create duplicates.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas
    )

    return len(chunks)

def clear_collection() -> None:
    """
    Delete and recreate the collection — wipes all stored chunks.
    Useful when re-ingesting documents from scratch.
    """
    try:
        _client.delete_collection(name="documents")
    except Exception:
        pass # collection didn't exist yet — that's fine