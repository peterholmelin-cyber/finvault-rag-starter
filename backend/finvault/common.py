from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReportType(str, Enum):
    QUARTERLY = "10_Q"
    ANNUAL = "10_K"


@dataclass
class DocumentInfo:
    filename: str
    ticker: str
    fiscal_year: int
    report_type: ReportType
    fiscal_quarter: int | None = None

    def __post_init__(self):
        self.ticker = self.ticker.upper()
        if self.report_type == ReportType.QUARTERLY and self.fiscal_quarter is None:
            raise ValueError("fiscal_quarter is required for 10-Q reports")
        if self.report_type == ReportType.ANNUAL and self.fiscal_quarter is not None:
            raise ValueError("fiscal_quarter must be None for 10-K reports")
        if self.fiscal_quarter is not None and not 1 <= self.fiscal_quarter <= 4:
            raise ValueError("fiscal_quarter must be between 1 and 4")


@dataclass
class Chunk:
    """Unified chunk for both ingestion and retrieval."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Ingestion fields
    vector_text: str | None = None

    # Retrieval fields (populated after DB fetch)
    id: int | None = None
    score: float | None = None
    filename: str | None = None

    @property
    def breadcrumb(self) -> str:
        parts = [self.metadata.get(k) for k in ("title", "section", "subsection")]
        return " > ".join(filter(None, parts))

    @classmethod
    def from_retrieved(cls, data: dict) -> "Chunk":
        """Create a Chunk from retriever/reranker output."""
        return cls(
            content=data["text"],
            metadata=data.get("meta", {}),
            id=data["id"],
            score=data.get("score"),
            filename=data.get("filename"),
        )


@dataclass
class StreamEvent:
    pass


@dataclass
class TokenEvent(StreamEvent):
    token: str


@dataclass
class SourceEvent(StreamEvent):
    chunk: Chunk


@dataclass
class DoneEvent(StreamEvent):
    pass
