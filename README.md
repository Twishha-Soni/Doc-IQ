# DocIQ — Document Intelligence Assistant

A full-stack RAG (Retrieval-Augmented Generation) application that lets you upload documents, index them into a vector store, and ask natural-language questions — with cited, grounded answers powered by Gemini.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-f97316?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-3.1_Flash_Lite-4285f4?style=flat-square&logo=google&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-FastMCP-8b5cf6?style=flat-square)

---

## What it does

- **Upload** `.txt` and `.pdf` documents through a clean web UI
- **Ingest** — chunks documents with a sliding-window splitter, embeds them with `sentence-transformers`, and stores vectors in ChromaDB
- **Ask** natural-language questions — the system retrieves the most semantically relevant chunks and generates a cited answer via Gemini
- **Conversation history** — follow-up questions maintain context across turns
- **Two interfaces** — a minimal web frontend (FastAPI) and a full MCP server exposing all RAG tools for AI client integration

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Interfaces                        │
│                                                          │
│   Browser      →  FastAPI (api.py)                      │
│   Claude Desktop / MCP Inspector  →  server.py          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                     RAG Pipeline                         │
│                                                          │
│  ingestor.py  →  chunker.py  →  embedder.py             │
│                                    │                     │
│                               ChromaDB                   │
│                                    │                     │
│  retriever.py  ←──────────────────┘                     │
│       │                                                  │
│  generator.py  →  Gemini API  →  cited answer           │
└─────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web API | FastAPI + Uvicorn |
| AI Tool Interface | FastMCP (Model Context Protocol) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (local persistence) |
| LLM | Google Gemini 3.1 Flash Lite via `google-genai` SDK |
| PDF parsing | PyMuPDF (`fitz`) |
| Frontend | HTML + CSS + Vanilla JS |

---

## Project structure

```
Doc-IQ/
├── documents/          # Drop .txt or .pdf files here
├── frontend/
│   ├── index.html      # Web UI markup
│   └── style.css       # Styles
├── chroma_store/       # Auto-created on first ingest
├── .env                # GEMINI_API_KEY goes here
├── requirements.txt
└── src/
    ├── main.py         # Startup validator
    ├── api.py          # FastAPI server (web interface)
    ├── server.py       # MCP server (Claude Desktop / Inspector)
    └── rag/
        ├── config.py   # CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
        ├── chunker.py  # Sliding-window text splitter
        ├── embedder.py # Embedding + ChromaDB store/retrieve
        ├── ingestor.py # Full ingest pipeline (.txt + .pdf)
        ├── retriever.py# Cosine similarity search
        └── generator.py# Gemini prompt builder + API call
```

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/Twishha-Soni/Doc-IQ.git
cd Doc-IQ
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env
# Open .env and paste your Gemini API key
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 3. Run the web interface

```bash
uvicorn src.api:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 4. Connect to Claude Desktop (optional)

The MCP server integrates directly with Claude Desktop, making all five
RAG tools available as native AI tools in any conversation.

Edit Claude Desktop's config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "doc-iq": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "mcp.run", "/absolute/path/to/Doc-IQ/src/server.py"],
      "env": {
        "GEMINI_API_KEY": "your_gemini_api_key_here"
      }
    }
  }
}
```

To find your absolute paths, run these from the project root with your venv activated:

```bash
which python          # → paste as "command"
realpath src/server.py  # → paste as the path in "args"
```

Restart Claude Desktop. A hammer icon (🔨) in the chat input confirms the tools are live.

### 5. Run the MCP Inspector (development)

For tool-by-tool testing during development:

```bash
mcp dev src/server.py
```

Opens the MCP Inspector at [http://localhost:5173](http://localhost:5173).

---

## How to use

1. **Upload** — drag and drop `.txt` or `.pdf` files onto the upload zone
2. **Ingest** — click "Ingest documents" to chunk, embed, and index them into ChromaDB
3. **Ask** — type any question in the chat input and press Enter
4. **Follow up** — conversation history is maintained automatically across turns, so follow-up questions like "can you elaborate?" work correctly

---

## MCP tools

The MCP server exposes five tools that mirror the full RAG pipeline:

| Tool | Description |
|---|---|
| `list_documents` | List all documents in the store |
| `read_document` | Read the full content of a document by filename |
| `ingest` | Chunk, embed, and index all documents into ChromaDB |
| `search` | Semantic search — returns top matching chunks with similarity scores |
| `ask_question` | Full RAG pipeline — retrieve + generate a cited answer |

---

## Configuration

All RAG parameters live in `src/rag/config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | 3 | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |

---

## Environment variables

```env
GEMINI_API_KEY=your_key_here
```

---

