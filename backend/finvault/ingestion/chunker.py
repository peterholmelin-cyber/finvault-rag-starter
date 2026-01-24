import re

import tiktoken
from langchain.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from supabase import Client

from finvault.common import Chunk, DocumentInfo
from finvault.prompts import CONTEXT_ENRICHMENT, get_prompt


class Chunker:
    """Unified pipeline for chunking, enriching, and storing documents."""

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        supabase: Client,
        max_tokens: int = 1024,
        min_tokens: int = 256,
        chunk_overlap: int = 100,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.supabase = supabase
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                ("##", "section"),
                ("###", "subsection"),
            ],
            strip_headers=False,
        )
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
        )

    def ingest(self, markdown: str, doc_info: DocumentInfo) -> int:
        """Process markdown content and store in database. Returns chunk count."""
        # TODO: Orchestrate the full ingestion pipeline by calling _chunk, _enrich, and _store in sequence.
        # TODO: Return the total number of chunks created.
        pass

    def _chunk(self, text: str) -> list[Chunk]:
        """Split markdown into semantic chunks respecting headers and tables."""
        # TODO: Use the header splitter to segment text by markdown headers.
        # TODO: For each section, decide whether to keep it whole or split further based on token count.
        # TODO: Preserve tables as single chunks regardless of size.
        pass

    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge consecutive chunks that are below the minimum token threshold."""
        # TODO: Iterate through chunks and combine adjacent small chunks.
        pass

    def _enrich(self, full_doc: str, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich each chunk with contextual summary using the LLM."""
        # TODO: For each chunk, invoke the LLM with the context enrichment prompt.
        pass

    def _store(self, chunks: list[Chunk], doc_info: DocumentInfo, markdown: str):
        """Persist document and chunks to the database."""
        # TODO: Insert document record into the 'documents' table and retrieve the generated ID.
        # TODO: Batch insert all chunk records with their embeddings into the 'chunks' table.
        pass

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _has_table(self, text: str) -> bool:
        return bool(re.search(r"^\|.+\|$", text, re.MULTILINE)) and "|---" in text
