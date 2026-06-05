from dataclasses import dataclass
from .embedder import _get_collection, _model
from .config import TOP_K_RESULTS

@dataclass
class RetrievedChunk:
    """
    A chunk returned from vector search, with its similarity score.
    This is what gets passed to the AI as context for answering a question.
    """
    text: str
    filename: str
    chunk_index: int
    similarity: float   # 1.0 = identical meaning, 0.0 = unrelated

def retrieve(query: str, top_k: int = TOP_K_RESULTS) -> list[RetrievedChunk]:
    """
    Find the most semantically relevant chunks for a given query.

    Steps:
    1. Embed the query using the same model used during ingest
    2. Ask ChromaDB for the top_k closest chunk vectors
    3. Unpack the parallel result lists into RetrievedChunk objects
    """
    collection = _get_collection()

    # Check if the collection has anything in it
    if collection.count() == 0:
        return []
    
    # Embed the query — same model as ingest, so vectors are in the same space
    query_vector = _model.encode(query).tolist()

    # Query ChromaDB for the closest chunks.
    # include= controls what gets returned alongside the IDs.
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns parallel lists wrapped in an outer batch list.
    # [0] unwraps the batch layer — we only sent one query, so we want index 0.
    documents = results["documents"][0] # list of chunk texts
    metadatas = results["metadatas"][0] # list of metadata dicts
    distances = results["distances"][0] # list of distance scores

    # ChromaDB returns "distance" (lower = more similar).
    # Convert to similarity (higher = more similar) so it reads more naturally.
    # For cosine distance: similarity = 1 - distance
    retrieved = []
    for doc, meta, dis in zip(documents, metadatas, distances):
        retrieved.append(RetrievedChunk(
            text=doc,
            filename=meta["filename"],
            chunk_index=meta["index"],
            similarity=round(1-dis,4)
        ))

    # Sort by similarity descending — most relevant first
    retrieved.sort(key=lambda chunk: chunk.similarity, reverse=True)

    return retrieved