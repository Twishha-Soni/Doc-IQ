from pathlib import Path
import fitz
from .chunker import chunk_text
from .embedder import embed_and_store, clear_collection

def _read_file(file_path: Path) -> str:
    """
    Extract text from a file, handling both .txt and .pdf formats.
    
    This is a 'dispatch' pattern — one function that routes to the
    right implementation based on the file type. Common in Python
    when you want callers to not care about the underlying format.
    """
    if file_path.suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    elif file_path.suffix == ".pdf":
        doc = fitz.open(str(file_path))            # open the PDF
        pages = [page.get_text() for page in doc]   # extract text per page
        doc.close()                    # always close file handles
        return "\n".join(pages)       # join pages into one string
    
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

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
    
    # Collect both .txt and .pdf files
    supported_extensions = {".txt", ".pdf"}
    files = [
        f for f in documents_dir.iterdir()
        if f.suffix in supported_extensions
    ]

    if not files:
        return {
            "status": "error",
            "message": f"No .txt or .pdf files found in {documents_dir}"
        }
    
    # Wipe existing vectors so re-ingesting is always a clean slate
    clear_collection()

    total_chunks = 0
    processed_files = []

    for file_path in files:
        try:
            text = _read_file(file_path)
            chunks = chunk_text(text, filename=file_path.name)
            count = embed_and_store(chunks)

            total_chunks += count
            processed_files.append({"filename": file_path.name, "chunks": count})

            print(f" Ingested {file_path.name} -> {count} chunks")

        except Exception as e:
            # Don't let one bad file stop the whole ingest
            print(f"Failed to ingest {file_path.name}: {e}")
            processed_files.append({"filename": file_path.name, "error": str(e)})
    

    return {
        "status": "ok",
        "files_processed": len(processed_files),
        "total_chunks": total_chunks,
        "details": processed_files
    }