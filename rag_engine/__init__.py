"""RAG engine package initializer for root-level usage.
Exposes query_handler and embed_chunks modules for convenient imports.
"""
from . import query_handler  # re-export
from . import embed_chunks
