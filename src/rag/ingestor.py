from pathlib import Path
from rag.chunker import chunk_text
from rag.embedder import embed_and_store, clear_collection

def ingest_documents(documents_dir: Path) -> dict:
    """
    Full ingest pipeline: read all .txt files, chunk them, embed them, store them.
    Returns a summary of what was processed.
    """
    if not documents_dir.exists():
        return {
            "status": "error",
            "message": f"Directory not found: {documents_dir}"
        }
    
    txt_files = [f for f in documents_dir.iterdir() if f.suffix == ".txt"]

    if not txt_files:
        return {
            "status": "error",
            "message": f"No .txt files found in {documents_dir}"
        }
    
    # Wipe existing vectors so re-ingesting is always a clean slate
    clear_collection()

    total_chunks = 0
    processed_files = []

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, filename=file_path.name)
        count = embed_and_store(chunks)

        total_chunks += count
        processed_files.append({"filename": file_path.name, "chunks": count})

        print(f" Ingested {file_path.name} -> {count} chunks")

    return {
        "status": "ok",
        "files_processed": len(processed_files),
        "total_chunks": total_chunks,
        "details": processed_files
    }