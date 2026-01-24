from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from finvault.prompts import MULTI_QUERY_EXPANSION, get_prompt


class ExpandedQueries(BaseModel):
    """Multiple query variations for improved retrieval coverage."""

    queries: list[str] = Field(description="List of 3 alternative query formulations")


class MultiQueryExpander:
    """Generates multiple query reformulations to improve retrieval recall."""

    def __init__(self, llm: BaseChatModel):
        # TODO: Configure the LLM with structured output using the ExpandedQueries schema.
        pass

    def expand(self, query: str) -> list[str]:
        """Return the original query plus up to 3 reformulated variants."""
        # TODO: Invoke the LLM with the multi-query expansion prompt.
        pass
