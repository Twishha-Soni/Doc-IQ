from dataclasses import dataclass
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

@dataclass
class Chunk:
    """
    Represents one piece of a document after splitting.
    Holds the text content plus metadata about where it came from.
    """
    text: str
    filename: str
    index: int    # which chunk this is within the document (0, 1, 2, ...)

def chunk_text(text: str, filename: str) -> list[Chunk]:
    """
    Split a document's text into overlapping chunks.

    The sliding window works like this:
    - Start at position 0, take CHUNK_SIZE characters → chunk 0
    - Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) characters
    - Take the next CHUNK_SIZE characters → chunk 1
    - Repeat until the end of the document

    The overlap means the last CHUNK_OVERLAP characters of chunk N
    are also the first CHUNK_OVERLAP characters of chunk N+1.
    This prevents a sentence being cut in half and losing its meaning.
    """
    chunks = []
    start = 0
    index = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text_content = text[start:end].strip()

        # skip empty chunks that can appear at the end of a document
        if chunk_text_content:
            chunks.append(Chunk(
                text=chunk_text_content,
                filename=filename,
                index=index
            )) 

        start += step
        index += 1

    return chunks
