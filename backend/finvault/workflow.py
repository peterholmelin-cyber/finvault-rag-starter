from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Annotated

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

from finvault import HybridRetriever, Reranker
from finvault.common import (
    Chunk,
    DoneEvent,
    ReportType,
    SourceEvent,
    StreamEvent,
    TokenEvent,
)
from finvault.prompts import RAG_QUERY, RAG_SYSTEM, get_prompt, load_prompt


@dataclass
class QueryFilters:
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    report_type: ReportType | None = None


@dataclass
class ContextSchema:
    llm: BaseChatModel
    retriever: HybridRetriever
    reranker: Reranker


CHUNK_TEMPLATE = "<chunk><id>{id}</id><content>{content}</content></chunk>"


@dataclass
class WorkflowState:
    query: str = ""
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    raw_chunks: list[dict] = field(default_factory=list)
    reranked_chunks: list[dict] = field(default_factory=list)
    answer_text: str = ""


def retrieve_node(
    state: WorkflowState, runtime: Runtime[ContextSchema], config: RunnableConfig
) -> dict:
    """Retrieve relevant chunks from the vector store."""
    # TODO: Call the retriever with the query and all applicable filters.
    # TODO: Return the raw chunks in the expected state update format.
    pass


def rerank_node(state: WorkflowState, runtime: Runtime[ContextSchema]) -> dict:
    """Rerank the retrieved chunks for relevance."""
    # TODO: Rerank the raw_chunks using the original query.
    # TODO: Return the reranked chunks in the expected state update format.
    pass


def generate_answer_node(
    state: WorkflowState, runtime: Runtime[ContextSchema], writer: StreamWriter
) -> dict:
    """Generate streaming answer using the top reranked chunks as context."""
    # TODO: Format the chunks into a context string using the CHUNK_TEMPLATE.
    # TODO: Stream LLM tokens and emit TokenEvent for each chunk via the writer.
    # TODO: After completion, emit SourceEvent for each used chunk.
    pass


def build_workflow() -> CompiledStateGraph[WorkflowState, ContextSchema]:
    """Construct and compile the RAG workflow graph."""
    # TODO: Add nodes for retrieve, rerank, and generate_answer.
    # TODO: Compile the graph with a MemorySaver checkpointer.
    pass


async def stream_response(
    rag_workflow: CompiledStateGraph[WorkflowState, ContextSchema],
    query: str,
    context: ContextSchema,
    filters: QueryFilters | None = None,
    thread_id: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream workflow events as an async generator."""
    # TODO: Configure the workflow run with filters and thread_id.
    # TODO: Yield each event from the workflow execution.
    pass
