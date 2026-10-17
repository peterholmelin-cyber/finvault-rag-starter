import json
import logging
import sys
import uuid
from collections.abc import Generator

import httpx
import streamlit as st

API_URL = "http://localhost:8000"
TIMEOUT = 180

st.set_page_config(page_title="FinVault", page_icon="📊", layout="wide")

st.header(":material/candlestick_chart: FinVault")
st.subheader("Your private AI financial analyst")

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
logging.getLogger("streamlit").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Session state
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#if "thread_id" not in st.session_state:
#    st.session_state.thread_id = str(uuid.uuid4())

if "ingestion_tasks" not in st.session_state:
    st.session_state.ingestion_tasks : [str] = []

def fetch_history(tid : str) -> list[dict]:
    """
    Retrieves messages for thread_id from server
    dict is of form {"role" : "user" or "assistant",
                    "content" : str
                    }
    """
    resp = httpx.get(f"{API_URL}/fetch_messages",
                      params={
                          "tid" : tid,
                          },
                      timeout=TIMEOUT,
                      )
    resp.raise_for_status()
    return resp.json()


if "messages" not in st.session_state:
    # Should be not note if this is reflesh
    tid = st.query_params.get("thread_id")
    st.session_state.messages = fetch_history(tid) if tid else []
    st.session_state.thread_id = tid or str(uuid.uuid4())
    # This persists across reloads
    st.query_params["thread_id"] = st.session_state.thread_id

class APIError(Exception):
    """
    API request failures
    """
    def __init__(self, message : str, status_code : int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

def escape_markdown(text: str) -> str:
    """Escape dollar signs to prevent LaTeX rendering issues."""
    return text.replace("$", r"\$")

@st.fragment(run_every=5)
def check_api_health() -> bool:
    """Check if the backend API is healthy and initialized."""
    # TODO: Return True only if the response indicates initialization is complete.
    # I guess this cannot throw exceptions since called at module level
    try:
        response = httpx.get(f"{API_URL}/health",timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()["initialized"]
    except httpx.ConnectError:
        return False


def upload_document(
    file, ticker: str, fiscal_year: int, report_type: str, fiscal_quarter: int | None
) -> dict:
    """Upload a document for ingestion."""
    # TODO: Construct the multipart form data with file and metadata.

    resp = httpx.post(f"{API_URL}/ingest",
                      files={"file" : (file.name, file.getvalue(), "application/pdf")},
                      data={
                          "ticker" : ticker,
                          "fiscal_year" : fiscal_year,
                          "fiscal_quarter": fiscal_quarter,
                          "report_type" : report_type,
                          },
                      timeout=TIMEOUT,
                      )
    if resp.status_code == 409:
        return resp.json() # duplicate upload: pass backend detail through
    resp.raise_for_status()
    return resp.json()

def get_task_status(task_id: str) -> dict | None:
    """Get the status of an ingestion task.
    the dict should have keys"""
    # TODO: GET the task status from /ingest/{task_id}/status.
    try:
        resp = httpx.get(f"{API_URL}/ingest/{task_id}/status")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError:
        if resp.status_code == 404:
            return None
        return {"status" : "unknown", "error" : f"HTTP {resp.status_code}"}
    except httpx.HTTPError as e:
        return {"status" : "unknown", "error" : str(e)}

def unescape_sse(s : str) -> str:
    # Escaped chaaracters key: value,
    # escape(value) -> \\+key
    _ESCAPES = {'n' : '\n', 'r' : '\r', '\\' : '\\'}

    out, i ,n = [], 0, len(s)
    while i < n:
        c = s[i]
        # '\\' is the character for \
        # so test if next two are in escaped image
        if c == '\\' and i + 1 < n and s[i+1] in _ESCAPES:
            out.append(_ESCAPES[s[i+1]])
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)

def stream_chat(query: str, thread_id: str | None, filters: dict) -> Generator[dict]:
    """Stream chat responses from the API using SSE."""
    # TODO: Construct the request payload with query, thread_id, and filters.
    # TODO: Parse SSE events (event type and data lines).
    payload = {
            "query" : query,
            "thread_id" : thread_id,
            **filters,
            }
    with httpx.stream(method="POST",
                      url=f"{API_URL}/chat",
                      json=payload,
                      timeout=TIMEOUT) as response:
        response.raise_for_status()

        event_type = None
        for line in response.iter_lines():
            # Rebuild sse string
            try:
                if line.startswith("event: "):
                    event_type = line.removeprefix("event: ")
                elif line.startswith("data: ") and event_type is not None:
                    data = line.removeprefix("data: ")
                    # We do not unescape  since we use json.loads
                    yield {'type' : event_type, 'content' : data}
                    event_type = None
            except Exception as e:
                yield {'type' : 'Error', 'content' : f'Client parsing-error: {e!s}' }

@st.fragment(run_every=5)
def task_status():
    # Synchronize tasks
    dropped = set()
    for task in st.session_state.ingestion_tasks[-5:]:
        status = get_task_status(task["task_id"])
        if status is None: # 404 not exist
            dropped.add(task["task_id"])
            continue
        row = {**task, **status}
        icon = {
            "pending": "⏳",
            "processing": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(status["status"], "❓")
        with st.expander(f"{icon} {row['filename'][:20]} ..."):
            st.text(f"Status: {row['status']}")
            st.text(f"Ticker: {row['ticker']}")
            if row["status"] == "completed":
                st.text(f"Chunks: {row['chunks_created']}")
            if row.get("error"):
                st.error(status["error"])
    if dropped:
        st.session_state.ingestion_tasks = [
                t for t in st.session_state.ingestion_tasks
                if t["task_id"] not in dropped
        ]

@st.fragment(run_every=5)
def display_message_count():
    if st.session_state.messages:
        st.caption(f"Messages: {len(st.session_state.messages)}")

# Check API health
api_healthy = check_api_health()

# Sidebar
with st.sidebar:
    st.header(":material/settings: Settings")

    if api_healthy:
        st.success("✓ API Connected")
    else:
        st.error("✗ API Offline - Start with `uvicorn server:app`")

    # Document upload
    st.divider()
    st.subheader(":material/upload_file: Upload Document")

    # report_type/quarter live outside the form so quarter options
    # update immediately when report type changes (forms batch changes
    # until submit)
    report_type = st.selectbox("Report Type", ["10_Q", "10_K"])
    quarter_options = [1, 2, 3, 4] if report_type == "10_Q" else [None]
    fiscal_quarter = st.selectbox(
        "Quarter",
        quarter_options,
        format_func=lambda x: "N/A (Annual)" if x is None else f"Q{x}",
        disabled=report_type == "10_K",
    )

    with st.form("upload_form",clear_on_submit=True):
        uploaded_file = st.file_uploader("Select PDF", type=["pdf"])
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", placeholder="NVDA").upper()
            fiscal_year = st.number_input(
                "Fiscal Year", min_value=2000, max_value=2030, value=2025
            )

        if st.form_submit_button("Upload & Process", use_container_width=True):
            if uploaded_file and ticker:
                with st.spinner("Uploading..."):
                    try:
                        result = upload_document(
                            uploaded_file,
                            ticker,
                            fiscal_year,
                            report_type,
                            fiscal_quarter,
                        )
                        if result.get("duplicate", False):
                            st.error(result["detail"])
                        else:
                            st.session_state.ingestion_tasks.append(
                                    {"task_id" : result["task_id"],
                                     "filename" : uploaded_file.name,
                                     "ticker" : ticker,
                                     })
                            st.success(f"Queued: {result['task_id'][:8]}...")
                    except httpx.HTTPError as e:
                        st.error(str(e))

    # Task status
    if st.session_state.ingestion_tasks:
        st.divider()
        st.subheader(":material/task: Processing Tasks")

        task_status()
            # Filters
    st.divider()
    st.subheader(":material/filter_alt: Search Filters")

    filter_ticker = st.text_input("Filter by Ticker", placeholder="All").upper() or None
    filter_year = (
        st.number_input("Filter by Year", min_value=2020, max_value=2030, value=2025) or None
    )
    filter_quarter = st.selectbox(
        "Filter by Quarter",
        [None, 1, 2, 3, 4],
        format_func=lambda x: "All" if x is None else f"Q{x}",
    )
    filter_type = st.selectbox(
        "Filter by Type",
        [None, "10_Q", "10_K"],
        format_func=lambda x: "All" if x is None else x,
    )

    st.divider()
    st.subheader(":material/chat: Conversation")

    if st.button(":material/add: New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.query_params["thread_id"] = st.session_state.thread_id
        st.rerun()

    display_message_count()

# Main chat area

## display st.query_param
st.divider()

#recreate_message

for msg in st.session_state.messages:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(escape_markdown(msg["content"]))
        if msg.get("sources"):
            with st.expander(f":material/source: Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**[{src['id']}]** {src['filename']} (score: {src['score']:.3f})"
                    )
                    st.caption(src["content"][:300] + "...")
                    st.divider()

if prompt := st.chat_input(
    "Ask about your financial documents... (e.g., 'Summarize NVDA Q3 earnings')",
    disabled=not api_healthy,
):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(escape_markdown(prompt))

    filters : dict[str, str | int | None] = {}
    if filter_ticker:
        filters["ticker"] = filter_ticker
    if filter_year:
        filters["fiscal_year"] = filter_year
    if filter_quarter:
        filters["fiscal_quarter"] = filter_quarter
    if filter_type:
        filters["report_type"] = filter_type

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        status_box = st.empty()
        answer_box = st.empty()
        status_box.status("**Thinking...**", state="running")
        sources = []
        final_content = ""

        # TODO: Implement the streaming chat interaction.
        # TODO: Update the placeholder with incremental token content.
        # TODO: Display sources in an expander when streaming completes.
        try:
            for event in stream_chat(prompt,st.session_state.thread_id,filters):
                if event["type"] == "TokenEvent":
                    final_content +=json.loads(event["content"])
                    answer_box.markdown(escape_markdown(final_content)) # replace element
                elif event["type"] == "SourceEvent":
                    chunk = json.loads(event["content"])
                    assert chunk is not None
                    sources.append(chunk)
                elif event["type"] == "DoneEvent":
                    if final_content:
                        #placeholder.markdown(escape_markdown(final_content))
                        status_box.status("Done",state="complete")
                elif event["type"] == "Error":
                    answer_box.markdown(json.loads(event['content']))
                    status_box.status("Error", state="error")
                else:
                    logger.warning(str(event))
                    status_box.status("Unknon event",state="error")

            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**[{s['id']}]** {s['filename']} (score: {s['score']:.3f})")
                    st.caption(s["content"][:300]+"...")
        except httpx.TimeoutException:
            answer_box.markdown(final_content or "No response -timed out._")
            status_box.status("Timeout", state="error")
        except httpx.HTTPError as e:
            status_box.status(f"API error: {e}", state="error")
        except Exception:
            # logger.exception captures full traceback
            logger.exception("Unexpected error during chat stream")
            answer_box.markdown("Something went wrong - check the server logs.")
            status_box.status("Error", state="error")

    # Should this line be here
    st.session_state.messages.append({"role" : "assistant",
                                      "content" : final_content,
                                      "sources" : sources})


