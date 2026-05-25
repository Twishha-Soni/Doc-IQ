import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rag.retriever import RetrievedChunk
from typing_extensions import TypedDict

load_dotenv()

# Create a client instance rather than configuring globally
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class Message(TypedDict):
    role: str      # "user" or "assistant"
    content: str   # message text


def _build_prompt(question: str, chunks: list[RetrievedChunk], history: list[Message]) -> str:

    """
    Build the full RAG prompt including optional conversation history.
    History gives the model context about what was already discussed,
    so follow-up questions like 'can you elaborate?' make sense.
    """

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk.filename}\n{chunk.text}]"
        )

    context = "\n\n".join(context_blocks)

    # Build the history section only if there are prior exchanges
    history_section = ""
    if history:
        history_lines = []
        for msg in history:
            # Capitalise the role label for readability in the prompt
            role_label = msg["role"].capitalize()
            history_lines.append(f"{role_label}: {msg['content']}")
        history_section = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"


    return f"""You are a helpful Document Intelligence Assistant.
Answer questions based strictly on the provided context.

Rules:
- Answer using ONLY the information in the context below
- If the context does not contain enough information, say so clearly
- Always cite which source(s) you used by referencing the filename e.g. [python_notes.txt]
- Use conversation history to understand follow-up questions
- Be concise and direct

{history_section}CONTEXT:
{context}

CURRENT QUESTION:
{question}

ANSWER:"""

def generate_answer(question: str, chunks: list[RetrievedChunk], history: list[Message] | None = None) -> dict:

    """
    Generate a grounded answer using Gemini.
    Accepts optional conversation history for follow-up question support.
    """

    if not chunks:
        return {
            "status": "error",
            "message": "No context chunks provided - cannot generate specific answer."
        }
    
    # Default to empty history if none provided — keeps backwards compatibility
    history = history or []    
    prompt = _build_prompt(question, chunks, history)

    try:
        response = _client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        answer_text = response.text
        sources = list({chunk.filename for chunk in chunks})

        return {
            "status": "ok",
            "answer": answer_text,
            "sources": sources,
            "chunks_used": len(chunks)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini API error: {str(e)}"
        }