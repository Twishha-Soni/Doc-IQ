# RAG Pipeline Configuration
# These values control how documents are broken up and searched.
# We'll use each of these in the levels ahead — for now, just read the comments.

# How many characters each chunk should be.
# Too large: chunks contain too much noise, retrieval is unfocused.
# Too small: chunks lose context, answers feel fragmented.
# 500 characters (~3-5 sentences) is a good starting point.
CHUNK_SIZE = 500

# How many characters adjacent chunks should overlap.
# Overlap prevents a sentence from being cut in half at a chunk boundary
# and losing meaning. Think of it like a sliding window.
CHUNK_OVERLAP = 50

# How many chunks to retrieve for each question.
# More chunks = more context for the AI, but more noise too.
TOP_K_RESULTS = 3

# The embedding model we'll use to convert text to vectors.
# This runs locally — no API key needed.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"