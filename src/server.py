from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from rag.ingestor import ingest_documents
from rag.retriever import retrieve
from rag.generator import generate_answer, Message

# Load environment variables before anything else
load_dotenv()

# This creates your MCP server — think of it like creating a Spring ApplicationContext.
# The string "document-assistant" is the server's name, which MCP clients can read.
mcp = FastMCP("document-assistant")

# Define where our documents live — relative to the project root.
# Path(__file__) is the path to this file (server.py).
# .parent goes up one level to src/, .parent again goes up to the project root.
# Then we append "documents" to get the documents/ folder.
DOCUMENTS_DIR = Path(__file__).parent.parent/"documents"

@mcp.tool()
def list_documents() -> dict:
    """
    List all available documents in the document store.
    Returns a list of filenames that can be read with read_document.
    Call this first to discover what documents are available.
    """
    if not DOCUMENTS_DIR.exists():
        return {
            "status": "error",
            "message": f"Documents directory not found at {DOCUMENTS_DIR}"
        }
    
    # List comprehension: build a list of filenames for every file in the folder
    files = [f.name for f in DOCUMENTS_DIR.iterdir() if f.is_file()]

    return {
        "status": "ok",
        "documents": files,
        "count": len(files)
    }

@mcp.tool()
def read_document(filename: str) -> dict:
    """
    Read the full contents of a document by filename.
    Use list_documents first to see available filenames.
    Returns the text content of the file.
    """
    # Build the full path by joining the documents directory with the filename
    file_path = DOCUMENTS_DIR / filename
    
    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "status": "ok",
            "filename": filename,
            "content": content,
            "character_count": len(content)
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"File '{filename}' not found. Use list_documents to see available files."
        }
    except PermissionError:
        return {
            "status": "error",
            "message": f"Permission denied reading '{filename}'"
        }

@mcp.tool()
def ingest() -> dict:
    """
    Ingest all documents into the RAG vector store.
    Reads every .txt file in the documents folder, splits them into chunks,
    converts each chunk to a vector embedding, and stores them for semantic search.
    Call this once after adding or updating documents, before using ask_question.
    """
    return ingest_documents(DOCUMENTS_DIR)

@mcp.tool()
def search(query: str) -> dict:
    """
    Search the document store for chunks relevant to a query.
    Uses semantic similarity — finds meaning matches, not just keyword matches.
    Returns the top matching chunks with their source filenames and similarity scores.
    Call ingest first to populate the vector store before searching.
    """
    if not query.strip():
        return {
            "status": "error",
            "message": "Query cannot be empty"
        }
    
    chunks = retrieve(query)

    if not chunks:
        return {
            "status": "error",
            "message": "No results found. The vector store may be empty - try calling ingest first."
        }
    
    return {
        "status": "ok",
        "query": query,
        "results_count": len(chunks),
        "results": [
            {
                "text": chunk.text,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "similarity": chunk.similarity
            }
            for chunk in chunks     # ← list comprehension over RetrievedChunk objects
        ]
    }

@mcp.tool()
def ask_question(question: str, history: list[Message] | None = None) -> dict:
    """
    Ask a question and get a grounded answer from your documents.
    Runs the full RAG pipeline: retrieve relevant chunks, generate a cited answer.

    Supports conversation history for follow-up questions.
    Pass history as a list of {role, content} dicts representing prior exchanges:
      [
        {"role": "user", "content": "what is RAG?"},
        {"role": "assistant", "content": "RAG stands for..."}
      ]

    Call ingest first to populate the vector store.
    """
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty"
        }
    
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question)

    if not chunks:
        return {
            "status": "error",
            "message": "No relevant content found. Try calling ingest first."
        }
    
    # Step 2: Generate a grounded answer using those chunks
    result = generate_answer(question, chunks, history)

    # Step 3: Attach retrieval metadata to the response
    if result["status"] == "ok":
        result["retrieved_chunks"] = [
            {
                "text": chunk.text,
                "filename": chunk.filename,
                "similarity": chunk.similarity
            }
            for chunk in chunks
        ]

    return result
    


if __name__ == "__main__":
    print("Starting Document Assistant MCP server...")
    print(f"Serving doucments from: {DOCUMENTS_DIR}")
    mcp.run()