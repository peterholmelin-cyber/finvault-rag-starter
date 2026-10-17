import re
import sys
from collections.abc import Iterator

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
         # |Left     |Centered | Right  |
        # |:--------|:-------:|-------:|
        # |To the   |   In    |  To the|
        # |left     | center  |   right|
        #
        # but ---| is valid as start and |---- is valid as end
        #
        # optional left |, optional alignment :, some -, optional alignment :,
        #
        # (?: ) non capturing group, i.e not storing match:
        # |, optional alignment :, some -, optional alignment:
        #
        # optional right |
        self._DELIM_ROW = re.compile(r"^\s*\|?\s*:?-+:?(?:\s*\|\s*:?-+:?)*\s*\|?\s*$")
        #
        # fenced code block in markdown, three or more ``` or
        # three or more ~~~
        self._FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


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
        chunks = self._chunk(markdown)
        chunks = self._merge_small(chunks)
        enriched = self._enrich(markdown,chunks)
        self._store(enriched,doc_info,markdown)
        return len(enriched)

    def _is_delimeter_row(self,line: str) -> bool:
        # Require at least one pipe so a bare --- horizontal rule or
        # setext-heading is never misstaken for a table.
        return "|" in line and self._DELIM_ROW.match(line) is not None

    def _chunk_table(self,text : str) -> Iterator[tuple[bool,str]]:
        """
        (false,prose)
        (true,table)
        """
        lines = text.splitlines(keepends=True)
        prose : list[str] = []
        fence : tuple[str,int] | None = None # (fence cha, min length) while
        # inside ```

        def flush() -> Iterator[tuple[bool,str]]:
            """
            Flushes existing prose when entering table
            """
            if prose:
                yield False, "".join(prose)
                prose.clear()

        i = 0
        while i < len(lines):
            line = lines[i]

            if fence is not None:
                # inside fenced code -> prose
                prose.append(line)
                ch,n = fence # ~~~~ gives fence=(~,4)
                if re.fullmatch(rf"\s*{ch}{{{n},}}", line.rstrip("\r\n")):
                    fence = None # found closing fence
                i += 1
                continue

            m = self._FENCE.match(line) # opening fence
            if m:
                prose.append(line)
                fence = (m.group(1)[0], len(m.group(1)))
                i += 1
                continue

            # Startin table
            if self._is_delimeter_row(line):
                header = prose[-1] if prose else ""
                if header.strip() and "|" in header:
                    prose.pop()
                    table = [header, line]
                    i += 1
                    while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                        table.append(lines[i])
                        i += 1
                    # yield existing prose
                    yield from flush()
                    # yield table
                    yield True, "".join(table)
                    continue

            # Accumulating regular prose
            prose.append(line)
            i += 1
        # final yield of prose
        yield from flush()


    def _chunk(self, text: str) -> list[Chunk]:
        """Split markdown into semantic chunks respecting headers and tables."""
        # TODO: Use the header splitter to segment text by markdown headers.
        """ Postcondition:
            The following two are set Chunk.content : str
                                      Chunk.metadata['section'] : str the header
            The rest are not set and are thus None
                Chunk.vector_text
                Chunk.id
                Chunk.score
                Chunk.filename
        """
        chunks = []
        for section in self._header_splitter.split_text(text):

            content = section.page_content.strip()
            if not content:
                continue
            # TODO: For each section, decide whether to keep it whole or split further based on token count.
            token_count = len(self.tokenizer.encode(content))
            if token_count <= self.max_tokens:
                chunks.append(Chunk(content,section.metadata))
                continue

            # TODO: Preserve tables as single chunks regardless of size.
            for is_table,c in self._chunk_table(content):

                if is_table:
                    chunks.append(Chunk(c,section.metadata))
                else:
                    # Recursive split
                    for sub in self._text_splitter.create_documents(
                        [c], metadatas=[section.metadata]):
                        chunks.append(Chunk(sub.page_content, section.metadata))

        return chunks

    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge consecutive chunks that are below the minimum token threshold."""
        # TODO: Iterate through chunks and combine adjacent small chunks.
        if not chunks:
            return []

        # First chunk cannot be merged into anything yet
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(self.tokenizer.encode(chunk.content)) <= self.min_tokens:
                tail = merged[-1]
                merged[-1] = Chunk(tail.content+"\n\n"+chunk.content, tail.metadata)
            else:
                merged.append(chunk)

        return merged

    def _enrich(self, full_doc: str, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich each chunk with contextual summary using the LLM."""
        # TODO: For each chunk, invoke the LLM with the context enrichment prompt.
        enriched = []
        for chunk in chunks:
            prompt = get_prompt(CONTEXT_ENRICHMENT, document=full_doc[:5000],
                                 chunk=chunk.content)
            context = self.llm.invoke(prompt).content.strip()
            content = re.sub(r"\n{3,}", "\n\n", chunk.content).strip()

            enriched.append(
                    Chunk(
                        content=content,
                        metadata={**chunk.metadata, "context" : context},
                        vector_text=f"{context}\n\n{content}",
                        )
                    )
        return enriched

    def _store(self, chunks: list[Chunk], doc_info: DocumentInfo, markdown: str):
        """
        Persist document and chunks to the database.
        supabase/migrations/20260106_init.sql
        """
        # TODO: Insert document record into the 'documents' table and retrieve the generated ID.
        # TODO: Batch insert all chunk records with their embeddings into the 'chunks' table.
        # Embed before any insert so embedding failures (wrong model
        # dimensions, Ollama down) leave nothing in the database.
        texts_to_embed = [c.vector_text for c in chunks]
        embeddings = self.embedding_model.embed_documents(texts_to_embed)

        document = {
                "filename"          : doc_info.filename,
                "markdown_content"  : markdown,
                "ticker"            : doc_info.ticker,
                "fiscal_year"       : doc_info.fiscal_year,
                "fiscal_quarter"    : doc_info.fiscal_quarter,
                "report_type"       : doc_info.report_type,
                }
        result = self.supabase.table("documents").insert(document).execute()
        doc_id = result.data[0]["id"]

        records = [
                {
                    "document_id"   : doc_id,
                    "content"       : c.content,
                    "metadata"      : c.metadata,
                    "embedding"     : emb,
                }
                for c,emb in zip(chunks,embeddings)
                ]
        try:
            result = self.supabase.table('chunks').insert(records).execute()
        except Exception:
            # Each PostgREST call is its own HTTP request, so the two inserts
            # cannot share a DB transaction. Compensate: deleting the document
            # cascades to any stored chunks. Re-raise so the task still fails.
            self.supabase.table("documents").eq("id", doc_id).delete().execute()
            raise


    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _has_table(self, text: str) -> bool:
        return bool(re.search(r"^\|.+\|$", text, re.MULTILINE)) and "|---" in text
