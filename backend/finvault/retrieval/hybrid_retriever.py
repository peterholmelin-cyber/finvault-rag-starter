from langchain_core.embeddings import Embeddings
from supabase import Client

from finvault.common import ReportType
from finvault.retrieval.hyde_expander import HyDEExpander
from finvault.retrieval.multi_query_expander import MultiQueryExpander


class HybridRetriever:
    """Retrieves chunks using hybrid (semantic + full-text) search with optional HyDE expansion."""

    def __init__(
        self,
        supabase: Client,
        embedding_model: Embeddings,
        multi_expander : MultiQueryExpander | None = None,
        expander: HyDEExpander | None = None,
        match_count: int = 3,
    ):
        self.supabase = supabase
        self.embedding_model = embedding_model
        self.multi_expander = multi_expander
        self.expander = expander
        self.match_count = match_count

    def retrieve(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
        report_type: ReportType | None = None,
    ) -> list[dict]:
        """Execute hybrid search with optional metadata filters."""
        # TODO: Call the Supabase RPC function 'hybrid_search' with query text and embedding.
        # TODO: Pass all filter parameters (ticker, fiscal_year, fiscal_quarter, report_type).

        # We only expand the original query not the multiquery variants
        expanded_query = query
        if self.expander is not None:
            expanded_query = self.expander.expand(query)

        if self.multi_expander is not None:
            variants = self.multi_expander.expand(expanded_query)
            queries = [expanded_query,*variants]
        else:
            queries = [expanded_query]

        merged : dict[int,dict] = {} #
        rank_bonus : dict[int,int] = {} # harmonic pooling of rank contributions

        # query_text is original query, embedding are expaned_query variants
        for q in queries:
            query_embedding = self.embedding_model.embed_query(q)
            response = self.supabase.rpc("hybrid_search", {
                "query_text"            : query,
                "query_embedding"       : query_embedding,
                "match_count"           : self.match_count,
                "filter_ticker"         : ticker.upper() if ticker else None,
                "filter_fiscal_year"    : fiscal_year,
                "filter_fiscal_quarter" : fiscal_quarter,
                "filter_report_type"    : report_type.value if report_type else None,
                }).execute()
            for rank, c in enumerate(response.data):
                cid = c["id"]
                # Harmonic decay bonus
                rank_bonus[cid] = rank_bonus.get(cid,0) + 1.0 / (60 + rank + 1)
                if cid not in merged or c["score"]  > merged[cid]["score"]:
                    merged[cid] = {
                          # vector_text of chunk is the field missing
                          # but it is created according to formatting in chunker
                          # {metadata["context"]}\n\n{content}
                    "id" : c["id"], # Chun id. document_id points to id in documents table
                    "query" : query,
                    "content" : c["content"],
                    "metadata" : c["metadata"],
                    "filename" : c["filename"],
                    "score" : c["score"],
                    }

        candidates = sorted(merged.values(),
                              key=lambda c: rank_bonus[c["id"]],reverse=True)
        return candidates[:self.match_count * 2] # pool for the reranker
