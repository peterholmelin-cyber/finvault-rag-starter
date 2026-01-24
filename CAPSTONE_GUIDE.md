# FinVault RAG Capstone Project

Welcome to the **FinVault** capstone project. You will build a production-grade Retrieval-Augmented Generation (RAG) system for analyzing SEC financial filings. This scaffold provides the project structure, data models, and configuration—your task is to implement the core AI orchestration, database operations, and streaming response logic.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                           (Streamlit Frontend)                              │
│                                                                             │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │  Document   │  │  Chat Interface │  │  Search Filters                 │  │
│  │  Upload     │  │  (SSE Streaming)│  │  (ticker, year, quarter, type)  │  │
│  └──────┬──────┘  └────────┬────────┘  └─────────────────────────────────┘  │
└─────────┼──────────────────┼────────────────────────────────────────────────┘
          │                  │
          │ HTTP/Multipart   │ HTTP/SSE
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                      │
│                         (FastAPI Backend)                                   │
│                                                                             │
│  ┌─────────────────────┐          ┌─────────────────────────────────────┐   │
│  │   /ingest (POST)    │          │   /chat (POST)                      │   │
│  │   Background Tasks  │          │   StreamingResponse (SSE)           │   │
│  └──────────┬──────────┘          └──────────────────┬──────────────────┘   │
└─────────────┼────────────────────────────────────────┼──────────────────────┘
              │                                        │
              ▼                                        ▼
┌─────────────────────────────┐    ┌──────────────────────────────────────────┐
│      INGESTION PIPELINE     │    │           RAG WORKFLOW                   │
│                             │    │         (LangGraph State Machine)        │
│  ┌───────────────────────┐  │    │                                          │
│  │  DocumentConverter    │  │    │  ┌──────────┐   ┌─────────┐   ┌───────┐  │
│  │  (Docling: PDF→MD)    │  │    │  │ Retrieve │──▶│ Rerank  │──▶│ Gen   │  │
│  └───────────┬───────────┘  │    │  │  Node    │   │  Node   │   │ Node  │  │
│              ▼              │    │  └──────────┘   └─────────┘   └───────┘  │
│  ┌───────────────────────┐  │    │       │                           │      │
│  │  Chunker              │  │    │       ▼                           ▼      │
│  │  • Header splitting   │  │    │  ┌─────────────────────────────────────┐ │
│  │  • Token-based split  │  │    │  │         StreamWriter                │ │
│  │  • Context enrichment │  │    │  │  (TokenEvent, SourceEvent, Done)    │ │
│  └───────────┬───────────┘  │    │  └─────────────────────────────────────┘ │
│              ▼              │    └──────────────────────────────────────────┘
│  ┌───────────────────────┐  │                        │
│  │  Embedding + Store    │  │                        │
│  └───────────┬───────────┘  │                        │
└──────────────┼──────────────┘                        │
               │                                       │
               ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│                    (Supabase + pgvector)                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  documents table          │  chunks table                           │    │
│  │  • filename, ticker       │  • content, metadata                    │    │
│  │  • fiscal_year/quarter    │  • embedding (vector 1024)              │    │
│  │  • report_type            │  • fts (tsvector - full-text search)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  hybrid_search() - RPC function                                     │    │
│  │  • Full-text search (ts_rank_cd)                                    │    │
│  │  • Semantic search (cosine similarity)                              │    │
│  │  • Reciprocal Rank Fusion (RRF)                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit | Interactive chat UI with SSE streaming |
| **API** | FastAPI | REST endpoints with background tasks |
| **Orchestration** | LangGraph | Stateful RAG workflow with checkpointing |
| **LLM** | LangChain + Ollama/Gemini | Model abstraction and invocation |
| **Embeddings** | Ollama (nomic-embed-text) | 1024-dim vector embeddings |
| **Reranking** | FlashRank | Cross-encoder passage reranking |
| **Document Processing** | Docling | PDF to Markdown with table extraction |
| **Database** | Supabase (PostgreSQL + pgvector) | Hybrid search with metadata filtering |

---

## 2. The Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) running locally with models:
  - `gemma3:4b` (or your preferred chat model)
  - `nomic-embed-text` (embeddings)
  - `qwen3-vl:2b` (vision-language for image descriptions)
- Supabase project (local or cloud)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd finvault-rag
   ```

2. **Install backend dependencies:**
   ```bash
   cd backend
   uv sync
   ```

3. **Install frontend dependencies:**
   ```bash
   cd ../frontend
   uv sync
   ```

4. **Configure environment:**
   ```bash
   cd ../backend
   cp .env.example .env
   # Edit .env with your Supabase credentials and model preferences
   ```

5. **Initialize the database:**
   ```bash
   # Apply migrations to your Supabase instance
   supabase db push
   # Or manually run: supabase/migrations/20260106_init.sql
   ```

6. **Verify Ollama models:**
   ```bash
   ollama list
   # Should show: gemma3:4b, nomic-embed-text, qwen3-vl:2b
   ```

### Running the Application

```bash
# Terminal 1: Start the backend
cd backend
uv run uvicorn server:app --reload

# Terminal 2: Start the frontend
cd frontend
uv run streamlit run app.py
```

---

## 3. Engineering Tickets

### Ticket 1: The Ingestion Pipeline

**Module:** `backend/finvault/ingestion/chunker.py`, `converter.py`

**Objective:** Transform PDF financial documents into semantically enriched, searchable chunks stored in the vector database.

**Requirements:**
- Convert PDF documents to markdown preserving tables and extracting image descriptions
- Split markdown by semantic headers (#, ##, ###) while respecting token limits
- Merge consecutive small chunks to maintain context coherence
- Enrich each chunk with LLM-generated contextual summaries
- Batch embed and store chunks with document metadata

**Constraints:**
- Tables must remain intact regardless of token count (detect via pipe/separator pattern)
- Use `cl100k_base` tokenizer for consistent token counting
- Context enrichment must use the provided prompt template
- Vector text for embedding combines context summary + original content
- All database inserts must include proper foreign key relationships

---

### Ticket 2: The Retrieval Layer

**Modules:** `backend/finvault/retrieval/hybrid_retriever.py`, `hyde_expander.py`, `multi_query_expander.py`, `reranker.py`

**Objective:** Implement advanced retrieval strategies combining query expansion, hybrid search, and cross-encoder reranking.

**Requirements:**
- Generate hypothetical document passages (HyDE) to improve semantic search
- Execute hybrid search combining full-text (BM25) and semantic (cosine) similarity
- Apply Reciprocal Rank Fusion (RRF) to merge result sets
- Rerank final candidates using a cross-encoder model
- Support metadata filtering by ticker, fiscal year, quarter, and report type

**Constraints:**
- HyDE expansion uses the embedding of the generated passage, not the original query
- The `hybrid_search` RPC function expects specific parameter names and types
- FlashRank returns numpy float32 scores that must be converted to native floats
- Ticker filters must be uppercase-normalized before database queries
- Multi-query expander must use LangChain's structured output with Pydantic schema

---

### Ticket 3: The RAG Workflow

**Module:** `backend/finvault/workflow.py`

**Objective:** Build a stateful, streaming RAG pipeline using LangGraph with conversation memory.

**Requirements:**
- Define a three-node workflow: retrieve → rerank → generate
- Stream LLM tokens to the client via custom StreamWriter events
- Maintain conversation history across turns using MemorySaver checkpointing
- Format retrieved chunks into XML-structured context for the prompt
- Emit typed events (TokenEvent, SourceEvent, DoneEvent) for frontend consumption

**Constraints:**
- WorkflowState must use the `add_messages` annotation for proper history merging
- Runtime context provides shared dependencies (LLM, retriever, reranker)
- Filters must be passed through RunnableConfig's configurable section
- Only top 5 reranked chunks should be included in the generation context
- The compiled graph must use `stream_mode="custom"` for StreamWriter integration

---

### Ticket 4: The API & Frontend Integration

**Modules:** `backend/server.py`, `frontend/app.py`

**Objective:** Wire up the RAG system with HTTP endpoints and a real-time streaming chat interface.

**Requirements:**
- Initialize all dependencies in the FastAPI lifespan context manager
- Process document uploads as background tasks with status polling
- Stream chat responses as Server-Sent Events (SSE) with proper formatting
- Parse SSE events in the frontend and render incremental token updates
- Display source citations with relevance scores after response completion

**Constraints:**
- Background tasks must update shared task state with proper status transitions
- SSE format requires `event: <type>\ndata: <payload>\n\n` structure
- Token data must escape newlines for single-line SSE transmission
- Streamlit placeholders must update incrementally during streaming
- Session state must persist thread_id for conversation continuity

---

## 4. Technical Hints

<details>
<summary><strong>Hint 1: LangGraph StreamWriter Pattern</strong></summary>

The `generate_answer_node` receives a `StreamWriter` (aliased as `writer`) that you call like a function to emit events. The workflow must be compiled and invoked with `stream_mode="custom"` for the writer to work.

```python
# Inside your node function:
writer(TokenEvent(token="some text"))

# When invoking the workflow:
async for chunk in workflow.astream(..., stream_mode="custom"):
    yield chunk  # These are your StreamWriter emissions
```

The writer is injected by LangGraph when the function signature includes `writer: StreamWriter`.

</details>

<details>
<summary><strong>Hint 2: Supabase RPC Parameter Mapping</strong></summary>

The `hybrid_search` function expects specific parameter names that map to the SQL function signature. Pay attention to:

- `query_text` - the original user query (for full-text search)
- `query_embedding` - must be a Python list, not numpy array: `list(embedding)`
- Filter parameters use `filter_` prefix: `filter_ticker`, `filter_fiscal_year`, etc.
- Enum values must be passed as strings: `report_type.value if report_type else None`

The response data uses different key names than your output format—you'll need to transform `content` → `text` and `metadata` → `meta`.

</details>

<details>
<summary><strong>Hint 3: SSE Parsing in httpx Streaming</strong></summary>

Server-Sent Events arrive as lines with a specific format. The event type comes on one line, the data on the next:

```
event: token
data: Hello

event: token  
data: World
```

When parsing with `httpx.stream()`, use `iter_lines()` and track state between lines:

```python
event_type = None
for line in response.iter_lines():
    if line.startswith("event: "):
        event_type = line[7:]  # Extract after "event: "
    elif line.startswith("data: "):
        data = line[6:]  # Extract after "data: "
        # Now process based on event_type
```

Remember that newlines in token data are escaped as `\\n` and need to be unescaped: `data.replace("\\n", "\n")`.

</details>

---

## Evaluation Criteria

Your implementation will be evaluated on:

1. **Correctness** - All endpoints function as specified
2. **Code Quality** - Clean, readable code following Python best practices
3. **Error Handling** - Graceful handling of edge cases and failures
4. **Performance** - Efficient database queries and streaming implementation
5. **Integration** - Proper wiring between all system components

Good luck, and remember: the scaffold provides the structure—you bring the intelligence.
