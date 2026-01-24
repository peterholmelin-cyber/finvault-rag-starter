import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from flashrank import Ranker
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, field_validator
from supabase import create_client

from finvault import Chunker, DocumentConverter, HybridRetriever, HyDEExpander, Reranker
from finvault.common import DocumentInfo, DoneEvent, ReportType, SourceEvent, TokenEvent
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
OLLAMA_CONTEXT_WINDOW = os.getenv("OLLAMA_CONTEXT_WINDOW")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application dependencies on startup."""
    global context, rag_workflow
    # TODO: Validate that required environment variables are set.
    # TODO: Initialize the chat model with appropriate provider-specific parameters.
    # TODO: Initialize the embedding model and Supabase client.
    # TODO: Construct the ContextSchema with LLM, retriever (with HyDE), and reranker.
    # TODO: Build the RAG workflow.
    # TODO: Yield control to the application.
    # TODO: Clean up resources on shutdown.
    yield


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
    # TODO: Update task status to PROCESSING.
    # TODO: Convert the PDF to markdown using DocumentConverter.
    # TODO: Run the chunking pipeline with context enrichment.
    # TODO: Clean up the temporary file.
    # TODO: Update task status to COMPLETED with chunk count.
    # TODO: Handle exceptions by setting FAILED status with error message.
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
    # TODO: Validate that the service is initialized and file is a PDF.
    # TODO: Construct DocumentInfo and validate business rules.
    # TODO: Save the uploaded file to a temporary location.
    # TODO: Create and register an IngestTask.
    # TODO: Schedule the background ingestion task.
    # TODO: Return the task ID and status for polling.
    pass


@app.get("/ingest/{task_id}/status")
async def get_ingest_status(task_id: str):
    """Check the status of an ingestion task."""
    # TODO: Look up the task by ID and return its status.
    # TODO: Raise 404 if task not found.
    pass


@app.get("/ingest/tasks")
async def list_ingest_tasks(limit: int = 20):
    """List recent ingestion tasks."""
    # TODO: Return the most recent tasks sorted by creation time.
    pass


@app.post("/chat")
async def chat(request: ChatRequest):
    """Query the RAG system with streaming SSE response."""
    # TODO: Validate service initialization and query content.
    # TODO: Construct QueryFilters from the request parameters.
    # TODO: Create an async generator that streams workflow events.
    # TODO: Transform TokenEvent, SourceEvent, and DoneEvent into SSE format.
    # TODO: Return a StreamingResponse with appropriate headers.
    pass


@app.get("/health")
async def health():
    return {"status": "healthy", "initialized": context is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
