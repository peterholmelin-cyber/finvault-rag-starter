import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
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

logger = logging.getLogger(__name__)

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
    # Postcondition: state.raw_chunks contains the retrieved chunks
    f = config["configurable"].get("filter")

    results = runtime.context.retriever.retrieve(
            query=state.query,
            ticker=f.ticker if f else None,
            fiscal_year=f.fiscal_year if f else None,
            fiscal_quarter=f.fiscal_quarter if f else None,
            report_type=f.report_type if f else None,
            )
    return {
            "raw_chunks" : results,
            }



def rerank_node(state: WorkflowState, runtime: Runtime[ContextSchema]) -> dict:
    """Rerank the retrieved chunks for relevance."""
    # TODO: Rerank the raw_chunks using the original query.
    # TODO: Return the reranked chunks in the expected state update format.
    # what should be kept in "meta"
    # Precondition: state.raw_chunks contains raw chunks
    # Postcondition: reranked_chunks contain reranked chuns with
    #                in meta
    passages = [{"id" : r["id"], "text" : r["content"],
                 "meta" : {"filename" : r.get("filename", "?"),
                           "old_score" : r.get("score")}} for r in state.raw_chunks ]
    query = state.query
    reranked = runtime.context.reranker.rerank(query,passages)
    return {
        "reranked_chunks" : reranked,
        }

def generate_answer_node(
    state: WorkflowState, runtime: Runtime[ContextSchema], writer: StreamWriter
) -> dict:
    """Generate streaming answer using the top reranked chunks as context."""
    # TODO: Format the chunks into a context string using the CHUNK_TEMPLATE.
    # TODO: Stream LLM tokens and emit TokenEvent for each chunk via the writer.
    # TODO: After completion, emit SourceEvent for each used chunk.
    """
    Precondition: state.reranked_chunks contains reranked chunks
                  state.query contains query
    Postcondition: messages [SystemMessage with RAG_SYSTEM prompt
                             HumanMessage with chunks according to CHUNK_TEMPLATE
                             AIMessage with reply ]

    state.messages                      llm.invoke
    --------------------------------------------------------------
    HumanMessage(query1) 
                                        [SystemMessage,
                                         state.messages[:-1]
                                         HumanMessage(RAG(query1))
    AIMessage(answer1)
    .........
    HumanMessage(query2)
                                        [SystemMessage,
                                        state.messages[:-1]
                                        HumanMessage(RAG(query2))
    AIMessage(answer2)

    so only HumanMessage(query) in state.messages, but invoke with 
    HumanMessage(RAG(query))
    """
    # Create context of 5 top ranked
    top_5_reranked = state.reranked_chunks[:5]
    context = "\n".join(CHUNK_TEMPLATE.format(
        id=c["id"],
        content=c["text"]) for c in top_5_reranked)

    humanMessage =  HumanMessage(content=get_prompt(RAG_QUERY,context=context,query=state.query))

    # state.messages ends with the plain query for this turn,
    # because the state.messages was CREAtED with query; only prior
    # turns belong in the LLM's history. The context-bearing prompt below
    # is ephemeral to this node.
    history = state.messages[:-1]
    messages = [
            SystemMessage(content=get_prompt(RAG_SYSTEM)),
            *history,
            humanMessage,
                        ]

    answer = ""
    llm = runtime.context.llm

    for m in llm.stream(messages):
        if m.content:
            answer += m.content
            writer(TokenEvent(token=m.content))

    #extract ids from answer text, Note ide might not exist in chunkse
    id_pattern = re.compile(r"\[id:(\d+)\]")
    quoted_ids = set(id_pattern.findall(answer))
    valid_ids = {str(c["id"]) for c in top_5_reranked}
    usable_ids = quoted_ids & valid_ids
    #logger.info("Usable ids=%s", usable_ids)

    id_to_raw = {str(c["id"]): c for c in state.raw_chunks}
    id_to_reranked = {str(c["id"]): c for c in state.reranked_chunks}

    cited = []
    for id_ in usable_ids:
        raw = id_to_raw[id_]
        chunk = Chunk(id=id_,
                      content=raw["content"],
                      score=id_to_reranked[id_]["score"],
                      filename=raw["filename"],
                      metadata=raw["metadata"].copy())
        cited.append(chunk)
        writer(SourceEvent(chunk))

    return {
            "answer_text": answer,
            # only the reply goes to state: the context-bearing prompt is
            # local to this node, add_messages would otherwise checkpoint it.
            # The cited chunks ride along in additional_kwargs so
            # /fetch_messages can restore the sources after a reload.
            "messages": [AIMessage(content=answer,
                                   additional_kwargs={"sources":
                                       [asdict(c) for c in cited]})],
            }

def build_workflow() -> CompiledStateGraph[WorkflowState, ContextSchema]:
    """Construct and compile the RAG workflow graph."""
    # TODO: Add nodes for retrieve, rerank, and generate_answer.
    # TODO: Compile the graph with a MemorySaver checkpointer.
    """
    Precondition:
    Postcondition:
    """
    workflow = StateGraph(WorkflowState, ContextSchema)
    workflow.add_node("retrieve",retrieve_node)
    workflow.add_node("rerank",rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve","rerank")
    workflow.add_edge("rerank","generate_answer")
    workflow.add_edge("generate_answer",END)

    graph = workflow.compile(checkpointer=MemorySaver())
    return graph

async def stream_response(
    rag_workflow: CompiledStateGraph[WorkflowState, ContextSchema],
    query: str,
    context: ContextSchema,
    filters: QueryFilters | None = None,
    thread_id: str | None = None,
) -> AsyncGenerator[StreamEvent]:
    """Stream workflow events as an async generator."""
    # TODO: Configure the workflow run with filters and thread_id.
    # TODO: Yield each event from the workflow execution.
    """
    Precondition:
    Postcondition:
    """
    workflowstate = WorkflowState(query=query,
                                  messages=[HumanMessage(content=query)],
                                  )
    config={"configurable" : { "filter" : filters, "thread_id" : thread_id }}

    async for event in rag_workflow.astream(
            workflowstate,
            context=context,
            config=config,
            stream_mode="custom",
            ):
        yield event
    yield DoneEvent()

