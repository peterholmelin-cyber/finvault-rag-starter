from flashrank import Ranker, RerankRequest


class Reranker:
    """Reranks passages using FlashRank cross-encoder."""

    def __init__(self, ranker: Ranker):
        self.ranker = ranker

    def rerank(self, query: str, passages: list[dict]) -> list[dict]:
        """Rerank passages by relevance to the query."""
        # TODO: Create a RerankRequest with the query and passage list.
        # TODO: Execute the reranking operation using the FlashRank ranker.
        # TODO: Convert score values to native Python floats for JSON serialization.
        # TODO: Return the reranked results.
        pass
