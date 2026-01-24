import json
import uuid

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FinVault", page_icon="📊", layout="wide")

st.header(":material/candlestick_chart: FinVault")
st.subheader("Your private AI financial analyst")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "ingestion_tasks" not in st.session_state:
    st.session_state.ingestion_tasks = []


def escape_markdown(text: str) -> str:
    """Escape dollar signs to prevent LaTeX rendering issues."""
    return text.replace("$", r"\$")


def check_api_health() -> bool:
    """Check if the backend API is healthy and initialized."""
    # TODO: Make a GET request to the /health endpoint.
    # TODO: Return True only if the response indicates initialization is complete.
    # TODO: Handle connection errors gracefully.
    pass


def upload_document(
    file, ticker: str, fiscal_year: int, report_type: str, fiscal_quarter: int | None
) -> dict:
    """Upload a document for ingestion."""
    # TODO: Construct the multipart form data with file and metadata.
    # TODO: POST to the /ingest endpoint.
    # TODO: Return the JSON response.
    pass


def get_task_status(task_id: str) -> dict:
    """Get the status of an ingestion task."""
    # TODO: GET the task status from /ingest/{task_id}/status.
    # TODO: Return the JSON response.
    pass


def stream_chat(query: str, thread_id: str | None, filters: dict):
    """Stream chat responses from the API using SSE."""
    # TODO: Construct the request payload with query, thread_id, and filters.
    # TODO: Open a streaming POST connection to /chat.
    # TODO: Parse SSE events (event type and data lines).
    # TODO: For token events, accumulate text and yield incremental updates.
    # TODO: For source events, parse JSON and collect sources.
    # TODO: For done events, yield final result with all sources.
    pass


# Check API health
api_healthy = check_api_health()

# Sidebar
with st.sidebar:
    st.header(":material/settings: Settings")

    if api_healthy:
        st.success("✓ API Connected")
    else:
        st.error("✗ API Offline - Start with `uvicorn server:app`")

    st.divider()
    st.subheader(":material/upload_file: Upload Document")

    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Select PDF", type=["pdf"])

        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", placeholder="NVDA").upper()
            fiscal_year = st.number_input(
                "Fiscal Year", min_value=2000, max_value=2030, value=2025
            )
        with col2:
            report_type = st.selectbox("Report Type", ["10_Q", "10_K"])
            fiscal_quarter = st.selectbox(
                "Quarter",
                [None, 1, 2, 3, 4],
                format_func=lambda x: "N/A (Annual)" if x is None else f"Q{x}",
                disabled=report_type == "10_K",
            )

        if report_type == "10_K":
            fiscal_quarter = None

        if st.form_submit_button("Upload & Process", use_container_width=True):
            if uploaded_file and ticker:
                if report_type == "10_Q" and not fiscal_quarter:
                    st.error("Quarter is required for 10-Q reports")
                else:
                    with st.spinner("Uploading..."):
                        result = upload_document(
                            uploaded_file,
                            ticker,
                            fiscal_year,
                            report_type,
                            fiscal_quarter,
                        )
                        if result and "task_id" in result:
                            st.session_state.ingestion_tasks.append(result["task_id"])
                            st.success(f"Queued: {result['task_id'][:8]}...")
                        else:
                            st.error(result.get("detail", "Upload failed") if result else "Upload failed")

    # Task status
    if st.session_state.ingestion_tasks:
        st.divider()
        st.subheader(":material/task: Processing Tasks")

        for task_id in st.session_state.ingestion_tasks[-5:]:
            try:
                status = get_task_status(task_id)
                if status:
                    icon = {
                        "pending": "⏳",
                        "processing": "🔄",
                        "completed": "✅",
                        "failed": "❌",
                    }.get(status["status"], "❓")
                    with st.expander(f"{icon} {status['filename'][:20]}..."):
                        st.text(f"Status: {status['status']}")
                        st.text(f"Ticker: {status['ticker']}")
                        if status["status"] == "completed":
                            st.text(f"Chunks: {status['chunks_created']}")
                        if status.get("error"):
                            st.error(status["error"])
            except Exception:
                pass

        if st.button(":material/refresh: Refresh Status"):
            st.rerun()

    # Filters
    st.divider()
    st.subheader(":material/filter_alt: Search Filters")

    filter_ticker = st.text_input("Filter by Ticker", placeholder="All").upper() or None
    filter_year = (
        st.number_input("Filter by Year", min_value=0, max_value=2030, value=0) or None
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
        st.rerun()

    if st.session_state.messages:
        st.caption(f"Messages: {len(st.session_state.messages)}")


# Main chat area
st.divider()

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
                    st.caption(src["text"][:300] + "...")
                    st.divider()

if prompt := st.chat_input(
    "Ask about your financial documents... (e.g., 'Summarize NVDA Q3 earnings')",
    disabled=not api_healthy,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(escape_markdown(prompt))

    filters = {}
    if filter_ticker:
        filters["ticker"] = filter_ticker
    if filter_year:
        filters["fiscal_year"] = filter_year
    if filter_quarter:
        filters["fiscal_quarter"] = filter_quarter
    if filter_type:
        filters["report_type"] = filter_type

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        placeholder = st.empty()
        placeholder.status("**Thinking...**", state="running")
        sources = []
        final_content = ""

        # TODO: Implement the streaming chat interaction.
        # TODO: Call stream_chat and iterate over the yielded events.
        # TODO: Update the placeholder with incremental token content.
        # TODO: Display sources in an expander when streaming completes.
        # TODO: Append the assistant message to session state.
        # TODO: Handle errors gracefully with st.error.
        pass
