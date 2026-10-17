import json
import logging
import os
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from operator import attrgetter

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from flashrank import Ranker
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, field_validator
from supabase import create_client

from finvault.retrieval.multi_query_expander import MultiQueryExpander

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from finvault import Chunker, DocumentConverter, HybridRetriever, HyDEExpander, Reranker
from finvault.common import DocumentInfo, DoneEvent, ReportType, SourceEvent, TokenEvent

#from finvault.retrieval.reranker import Reranker
from finvault.workflow import (
    ContextSchema,
    QueryFilters,
    WorkflowState,
    build_workflow,
    stream_response,
)

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")
CHAT_MODEL_PROVIDER = os.getenv("CHAT_MODEL_PROVIDER")
OLLAMA_CONTEXT_WINDOW = int(os.getenv("OLLAMA_CONTEXT_WINDOW","16384"))
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
EXPANDER_MODEL = os.getenv("EXPANDER_MODEL")
MULTI_EXPANDER_MODEL = os.getenv("MULTI_EXPANDER_MODEL")

STREAMING_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

context: ContextSchema | None = None
rag_workflow: CompiledStateGraph[WorkflowState, ContextSchema] | None = build_workflow()

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    filename: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int | None
    report_type: str
    chunks_created: int
    error: str | None
    created_at: datetime
    completed_at: datetime | None

@dataclass
class IngestTask:
    id: str
    filename: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int | None
    report_type: str
    status: TaskStatus = TaskStatus.PENDING
    chunks_created: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_response(self) -> dict:
        return {
            "task_id": self.id,
            "status": self.status.value,
            "filename": self.filename,
            "ticker": self.ticker,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter,
            "report_type": self.report_type,
            "chunks_created": self.chunks_created,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


tasks: dict[str, IngestTask] = {}

def escape_sse(s : str) -> str:
    return (s.replace('\\','\\\\')
             .replace('\r', '\\r')
             .replace('\n','\\n'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application dependencies on startup."""
    global context, rag_workflow
    # TODO: Initialize the chat model with appropriate provider-specific parameters.
    # TODO: Initialize the embedding model and Supabase client.
    # TODO: Build the RAG workflow.
    app.state.llm = init_chat_model(
            model=CHAT_MODEL,
            model_provider=CHAT_MODEL_PROVIDER,
            num_ctx=OLLAMA_CONTEXT_WINDOW,
            )
    app.state.embedding_model = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    app.state.supabase = create_client(os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_KEY"))

    expander_llm = init_chat_model(
            model=EXPANDER_MODEL,
            model_provider=CHAT_MODEL_PROVIDER,
            num_ctx=OLLAMA_CONTEXT_WINDOW,
            )
    multi_expander_llm = init_chat_model(
            model=MULTI_EXPANDER_MODEL,
            model_provider=CHAT_MODEL_PROVIDER,
            num_ctx=OLLAMA_CONTEXT_WINDOW,
            )
    retriever=HybridRetriever(
        supabase=app.state.supabase,
        embedding_model=app.state.embedding_model,
        multi_expander=MultiQueryExpander(multi_expander_llm),
        expander=HyDEExpander(expander_llm)
        )
    assert retriever is not None

    reranker=Reranker(Ranker(model_name=RERANKER_MODEL))
    context = ContextSchema(
        llm=app.state.llm,
        retriever=retriever,
        reranker=reranker,
        )

    # rag workflow is compiled

    # This is where we a re ready
    yield
    # This is what we do after finished

app = FastAPI(
    title="FinVault RAG API",
    description="Private RAG API for financial document analysis",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    report_type: ReportType | None = None

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str | None) -> str | None:
        return v.upper() if v else None

    @field_validator("fiscal_quarter")
    @classmethod
    def validate_quarter(cls, v: int | None) -> int | None:
        if v is not None and not 1 <= v <= 4:
            raise ValueError("fiscal_quarter must be between 1 and 4")
        return v


def run_ingestion(task_id: str, pdf_path: str, doc_info: DocumentInfo):
    """Background task for document ingestion."""
    # TODO: Convert the PDF to markdown using DocumentConverter.
    # TODO: Run the chunking pipeline with context enrichment.
    global context,app
    task = tasks[task_id]
    task.status = TaskStatus.PROCESSING
    try:
        converter = DocumentConverter()
        markdown = converter.convert(pdf_path)
        chunker = Chunker(llm=app.state.llm,
                      embedding_model=app.state.embedding_model,
                      supabase=app.state.supabase)
        count = chunker.ingest(markdown,doc_info)
        task.status = TaskStatus.COMPLETED
        task.chunks_created = count
    except Exception as e:
        logger.exception(f"Ingestion failed {e}")
        task.status = TaskStatus.FAILED
        task.error = str(e)
    finally:
        task.completed_at = datetime.now()
        try:
            os.unlink(pdf_path)
        except OSError:
            pass

@app.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    ticker: str = Form(...),
    fiscal_year: int = Form(...),
    report_type: ReportType = Form(...),
    fiscal_quarter: int | None = Form(None),
):
    """Ingest a PDF document (background processing). Poll /ingest/{task_id}/status for progress."""
    # TODO: Create and register an IngestTask.
    # TODO: Schedule the background ingestion task.
    # create id with uuid.uuid4

    # chunks!inner limits matches to documents that actually have chunks,
    # so an ingestion that failed after the documents insert is not
    # treated as a duplicate and can be retried
    existing = (
            app.state.supabase.table("documents")
            .select('id, fiscal_quarter, chunks!inner(document_id)')
            .eq("ticker", ticker.upper())
            .eq("fiscal_year", fiscal_year)
            .eq("report_type", report_type.value)
            .execute()
        ).data

    def same_period(row):
        return (row["fiscal_quarter"] or None) == fiscal_quarter

    logger.info("Searching for duplicates")
    duplicate = next((r for r in existing if same_period(r)),None)
    if  duplicate is not None:
        logger.info(f"File exists id={duplicate["id"]}")
        return JSONResponse(
                status_code=409,
                content={
                    "duplicate" : True,
                    "detail" : f"{ticker} {report_type.value} FY{fiscal_year}"
                    + (f"Q{fiscal_quarter}" if fiscal_quarter else "")
                    + " is already ingested. Delete old to ingest",
                         "task_id" : str(duplicate["id"])},
                )

    logger.info("New file ingesting...")
    # save the file
    suffix = os.path.splitext(file.filename or "upload.pdf")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        pdf_path = tmp.name

    task = IngestTask(
        id=str(uuid.uuid4()),
        filename=file.filename,
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        report_type=report_type,
        )
    tasks[task.id] = task

    # Docinfo  has original filename
    doc_info = DocumentInfo(file.filename,
                            ticker,
                            fiscal_year,
                            report_type,
                            fiscal_quarter,
                            )
    background_tasks.add_task(run_ingestion,
                            task.id,
                            pdf_path,
                            doc_info)
    # is there a yield heret until background_tasks have finishied?
    return task.to_response()


@app.get("/ingest/{task_id}/status")
async def get_ingest_status(task_id: str) -> TaskResponse:
    """Check the status of an ingestion task."""
    # TODO: Look up the task by ID and return its status.
    # TODO: Raise 404 if task not found.
    # status one of PENDING,PROCESSING,COMPLETED,FAILED
    try:
        task = tasks[task_id]
        return TaskResponse(**task.to_response())
    except KeyError:
        logger.info(f"Task {task_id} not found in tasks")
        raise HTTPException(status_code=404, detail="task not found")


@app.get("/ingest/tasks")
async def list_ingest_tasks(limit: int = 20):
    """List recent ingestion tasks."""
    # TODO: Return the most recent tasks sorted by creation time.
    recent = sorted(tasks.values(),
                    key=attrgetter("created_at"), reverse=True)[:limit]
    return [TaskResponse(**t.to_response()) for t in recent]

@app.get("/fetch_messages")
async def fetch_messages(tid : str):
    config = {"configurable" : {"thread_id"  :tid}}

    try:
        reply = []
        snapshot = rag_workflow.get_state(config)
        for message in snapshot.values.get("messages", []):
            match message:
                case HumanMessage(content=content):
                    reply.append({"role" : "user",
                                  "content" : content})
                case AIMessage(content=content):
                    reply.append({"role" : "assistant",
                                  "content" : content,
                                  "sources" : message.additional_kwargs.get("sources", [])})
                case _:
                    pass
    except ValueError:
        logger.info(f"fetch_messages({tid}) failed, thread={tid} not in state")
    return reply





@app.post("/chat")
async def chat(request: ChatRequest):
    """Query the RAG system with streaming SSE response."""
    # TODO: Create an async generator that streams workflow events.
    # TODO: Transform TokenEvent, SourceEvent, and DoneEvent into SSE format.

    async def response_generator() -> AsyncGenerator[str]:
        """
        """
        try:
            global context
            filters=QueryFilters(ticker=request.ticker,
                                 fiscal_year=request.fiscal_year,
                                 fiscal_quarter=request.fiscal_quarter,
                                 report_type=request.report_type,
                                 )
            async for event in stream_response(rag_workflow=rag_workflow,
                                query=request.query,
                                context=context,
                                filters=filters,
                                thread_id=request.thread_id,
                                ):
                match event:
                    case TokenEvent(token=token):
                        # token : str
                        #esc_token = escape_sse(token)
                        yield f'event: TokenEvent\ndata: {json.dumps(token)}\n\n'

                    case SourceEvent(chunk=chunk):
                        # chunk : Chunk
                        # We do not need to escape hat goes into
                        # json.dumps
                        yield f'event: SourceEvent\ndata: {json.dumps(asdict(chunk))}\n\n'
                    case DoneEvent():
                        yield 'event: DoneEvent\ndata: \n\n'
        except Exception as e:
            #esc_e = escape_sse(str(e))
            logger.exception("Chat stream failed")
            yield f'event: Error\ndata: {json.dumps(str(e))}\n\n'

    return StreamingResponse(
            response_generator(),
            media_type="text/event-stream",
            )


@app.get("/health")
async def health():
    return {"status": "healthy", "initialized": context is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
