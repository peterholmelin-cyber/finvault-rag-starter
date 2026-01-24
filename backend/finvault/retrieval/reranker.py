from flashrank import Ranker, RerankRequest


class Reranker:
    """Reranks passages using FlashRank cross-encoder."""

    def __init__(self, ranker: Ranker):
        self.ranker = ranker

    def rerank(self, query: str, passages: list[dict]) -> list[dict]:
        """Rerank passages by relevance to the query."""
        # TODO: Execute the reranking operation using the FlashRank ranker.
        pass
