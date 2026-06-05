import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.rag.ingestor import ingest_documents
from src.rag.retriever import retrieve
from src.rag.generator import generate_answer, Message

app = FastAPI(title="Document Intelligence Assistant")

# Where uploaded documents get saved — same folder your RAG pipeline reads from
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)    # create it if it doesn't exist yet

# Mount the frontend folder so FastAPI can serve index.html
# We'll create this folder in Level 3 — for now it just needs to exist
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)


# ── Pydantic models ──────────────────────────────────────────────────────────
# These define the shape of JSON request bodies.
# FastAPI validates incoming JSON against these automatically —
# wrong field type or missing field returns a 422 error with a clear message.
# Think of them like Java records or DTOs.

class AskRequest(BaseModel):
    question: str
    history: list[Message] = []     # default to empty list if not provided


# ---- Endpoints ----

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a .txt or .pdf file and save it to the documents folder.
    The '...' inside File(...) means this field is required — FastAPI convention.
    'async def' is used here because file I/O benefits from async handling in FastAPI.
    """
    allowed = {".txt", ".pdf"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        return {
            "status": "error",
            "message": f"Only .txt and .pdf files are supported."
        }
    
    dest = DOCUMENTS_DIR / file.filename

    # Save the uploaded file to disk
    # shutil.copyfileobj copies file content in chunks — memory-efficient for large files
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "ok",
        "filname": file.filename
    }

@app.post("/ingest")
def ingest():
    """
    Trigger the full ingest pipeline over all documents in the documents folder.
    Same logic as the MCP ingest tool — just exposed over HTTP.
    """
    return ingest_documents(DOCUMENTS_DIR)

@app.post("/ask")
def ask(request: AskRequest):
    """
    Run the full RAG pipeline: retrieve relevant chunks, generate a cited answer.
    Accepts optional conversation history for follow-up questions.
    """
    if not request.question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty."
        }
    
    chunks = retrieve(request.question)

    if not chunks:
        return {
            "status": "error",
            "message": "No results found.Try ingesting documents first."
        }
    
    result = generate_answer(request.question, chunks, request.history)

    if result["status"] == "ok":
        result["sources"] = [
            {"filename": chunk.filename, "similarity": chunk.similarity}
            for chunk in chunks
        ]

    return result

@app.get("/documents")
def list_documents():
    """List all documents currently in the documents folder."""
    files = [f.name for f in DOCUMENTS_DIR.iterdir() if f.is_file()]
    return {
        "status": "ok",
        "documents": files
    }

# Serve the frontend — this must come last so API routes take priority
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
