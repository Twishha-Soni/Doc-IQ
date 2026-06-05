# DocIQ — Document Intelligence Assistant

A full-stack RAG (Retrieval-Augmented Generation) application that lets you upload documents, index them into a vector store, and ask natural-language questions — with cited, grounded answers powered by Gemini.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-f97316?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-4285f4?style=flat-square&logo=google&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-FastMCP-8b5cf6?style=flat-square)

---

## What it does

- **Upload** `.txt` and `.pdf` documents through a web UI or MCP tools
- **Ingest** — chunks documents with a sliding-window splitter, embeds them with `sentence-transformers`, and stores vectors in ChromaDB
- **Ask** natural-language questions — the system retrieves the most semantically relevant chunks and generates a cited answer via Gemini
- **Conversation history** — follow-up questions maintain context across turns
- **Two interfaces** — a minimal web frontend (FastAPI) and a full MCP server for AI client integration

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Interfaces                        │
│                                                          │
│   Browser → FastAPI (api.py)                            │
│   Claude Desktop → MCP Server (server.py)               │
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
| AI Interface | FastMCP (Model Context Protocol) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (local persistence) |
| LLM | Google Gemini 2.0 Flash via `google-genai` SDK |
| PDF parsing | PyMuPDF (`fitz`) |
| Frontend | Vanilla HTML/CSS/JS (zero dependencies) |

---

## Project structure

```
document-assistant/
├── documents/          # Drop .txt or .pdf files here
├── frontend/
│   └── index.html      # Single-file web UI
├── chroma_store/       # Auto-created — persisted vector DB
├── .env                # GEMINI_API_KEY goes here
├── requirements.txt
└── src/
    ├── main.py         # Startup validator
    ├── api.py          # FastAPI server (web interface)
    ├── server.py       # MCP server (AI client interface)
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
git clone https://github.com/YOUR_USERNAME/document-assistant.git
cd document-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 3. Run the web interface

```bash
uvicorn src.api:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 4. Run the MCP server (optional)

```bash
mcp dev src/server.py
```

Opens the MCP Inspector at [http://localhost:5173](http://localhost:5173) for tool-by-tool testing.

---

## How to use

1. **Upload** — drag and drop `.txt` or `.pdf` files onto the upload zone
2. **Ingest** — click "Ingest documents" to chunk, embed, and index them
3. **Ask** — type any question in the chat input and press Enter
4. **Follow up** — conversation history is maintained automatically across turns

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

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

---

## .env.example

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## License

MIT
