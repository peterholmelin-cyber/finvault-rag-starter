from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from finvault.prompts import HYDE_EXPANSION, get_prompt


class YesNo(BaseModel):
    yes : bool = Field(description="Is the answer yes or no")
    reason : str = Field(description="Explain why you set the yes to true/false?")

class HyDEExpander:
    """Expands queries using Hypothetical Document Embedding."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def expand(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        # TODO: Invoke the LLM to generate a hypothetical answer passage.

        response = self.llm.with_structured_output(YesNo).invoke(
            f"Should the query '{query}' use HyDEExpansios for RAG search?")
        if response.yes:
            prompt = get_prompt(HYDE_EXPANSION)
            query = prompt.format(query=query)
            return self.llm.invoke(query).content
        else:
            return query
