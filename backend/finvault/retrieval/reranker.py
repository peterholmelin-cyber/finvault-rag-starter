from flashrank import Ranker, RerankRequest


class Reranker:
    """Reranks passages using FlashRank cross-encoder."""

    def __init__(self, ranker: Ranker):
        self.ranker = ranker

    def rerank(self, query: str, passages: list[dict]) -> list[dict]:
        """Rerank passages by relevance to the query."""
        # TODO: Execute the reranking operation using the FlashRank ranker.
        # what are dict keys supposed to be?
        # {"id" : int, "text" : str, "meta" : dict}
        # output mutation mutate results with field "score" : float
        # list is sorted descending score
        if not passages:
            return []

        results = self.ranker.rerank(RerankRequest(query=query,passages=passages))
        # results dict: {id : int, text : str, score : float , meta : { "old_score" : float} }
        for r in results:
            # convert from np.float to float
            r["score"] = float(r["score"])
        return results

