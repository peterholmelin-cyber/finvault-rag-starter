from langchain.chat_models import BaseChatModel

from finvault.prompts import HYDE_EXPANSION, get_prompt


class HyDEExpander:
    """Expands queries using Hypothetical Document Embedding."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def expand(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        # TODO: Construct the prompt using the HyDE expansion template.
        # TODO: Invoke the LLM to generate a hypothetical answer passage.
        # TODO: Return the generated text content.
        pass
