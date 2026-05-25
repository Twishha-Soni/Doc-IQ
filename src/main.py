import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_environment() -> list[str]:
    """
    Validate that everything needed to run is in place.
    Returns a list of error messages — empty list means all good.
    """
    errors = []

    if not os.getenv("GEMINI_API_KEY"):
        errors.append("GEMINI_API_KEY not set - add it to your .env file")

    document_dir = Path(__file__).parent.parent / "documents"
    if not document_dir.exists():
        errors.append(f"documents/ folder not found at {document_dir}")
    else:
        txt_files = list(document_dir.glob("*.txt"))
        if not txt_files:
            errors.append("No .txt files found in documents/ folder at {document_dir}")

    return errors

def main():
    print("=" * 50)
    print("  Document Intelligence Assistant")
    print("=" * 50)

    # Validate environment before starting
    errors = check_environment()
    if errors:
        print("\nStartup checks failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("\nStart checks passed:")
    print(" - GEMINI_API_KEY found")
    print(" - documents/ folder found")

    print("\nAvailable MCP tools:")
    tools = [
        ("ingest()", "Ingest documents into the vector store"),
        ("list_documents()", "List available documents"),
        ("read_document(filename)", "Read a document's full content"),
        ("search(query)", "Semantic search over document chunks"),
        ("ask_question(question, history?)", "RAG-powered Q&A with citations"),
    ]

    for name, description in tools:
        print(f" # {name:40s} {description}")

    print("\nStarting MCP server...")
    print("Run with: mcp dev src/server.py")
    print("=" * 50)

if __name__ == "__main__":
    main()