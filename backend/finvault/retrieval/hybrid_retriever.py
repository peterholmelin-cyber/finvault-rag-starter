from langchain_core.embeddings import Embeddings
from supabase import Client

from finvault.common import ReportType
from finvault.retrieval.hyde_expander import HyDEExpander


class HybridRetriever:
    """Retrieves chunks using hybrid (semantic + full-text) search with optional HyDE expansion."""

    def __init__(
        self,
        supabase: Client,
        embedding_model: Embeddings,
        expander: HyDEExpander | None = None,
        match_count: int = 3,
    ):
        self.supabase = supabase
        self.embedding_model = embedding_model
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
        # TODO: Apply HyDE expansion to the query if an expander is configured.
        # TODO: Call the Supabase RPC function 'hybrid_search' with query text and embedding.
        # TODO: Pass all filter parameters (ticker, fiscal_year, fiscal_quarter, report_type).
        # TODO: Configure the RRF fusion weights for combining full-text and semantic scores.
        # TODO: Transform the response data into a standardized chunk dictionary format.
        pass
