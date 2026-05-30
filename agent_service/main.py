"""FastAPI entrypoint for AgentService."""

import csv
import html
import inspect
import io
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from agent_service.config import Settings
from agent_service.dialogue_policy import DialoguePolicy, DialoguePolicyMode
from agent_service.governance.data_manifest import ProcessedStyleSample
from agent_service.llm import LLMClient, MockLLMClient, OpenAILLMClient
from agent_service.memory.memory_store import MemoryStore
from agent_service.persona import PersonaEngine
from agent_service.rag.chunker import RecursiveTextChunker
from agent_service.rag.document_loader import DocumentLoader
from agent_service.rag.embeddings import MockEmbeddingClient
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.rag.vector_store import ChromaVectorStore
from agent_service.review import (
    ApproveReviewRequest,
    EditReviewRequest,
    HumanReviewDetail,
    HumanReviewInvalidTransitionError,
    HumanReviewList,
    HumanReviewNotFoundError,
    HumanReviewRecord,
    HumanReviewStore,
    ReviewStatus,
)
from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    ChatResponse,
    ErrorEnvelope,
    no_reply_command,
)
from agent_service.style.learning import StyleLearningStore
from agent_service.style.pair_store import StylePairStore
from agent_service.style.style_store import StyleStore
from agent_service.tools import ToolRegistry
from agent_service.tools.builtin import build_default_tool_registry
from agent_service.workflow import resume_agent_review, run_agent_chat

ChatHandler = Callable[[ChatRequest], AgentReplyCommand | Awaitable[AgentReplyCommand]]


async def _default_chat_handler(request: ChatRequest) -> AgentReplyCommand:
    return run_agent_chat(request)


async def _call_chat_handler(
    handler: ChatHandler,
    request: ChatRequest,
) -> AgentReplyCommand:
    call_method = type(handler).__call__ if callable(handler) else None
    if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(call_method):
        result = handler(request)
    else:
        result = await run_in_threadpool(handler, request)
    if inspect.isawaitable(result):
        return await result
    return result


def create_app(
    settings: Settings | None = None,
    chat_handler: ChatHandler | None = None,
    review_store: HumanReviewStore | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    style_pair_store: StylePairStore | None = None,
    style_learning_store: StyleLearningStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    dialogue_policy: DialoguePolicy | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(title="PersonaAgent AgentService")
    store = review_store or HumanReviewStore(app_settings.agent_state_db_path)
    memories = memory_store or MemoryStore(
        sqlite_path=app_settings.memory_db_path,
        chroma_path=app_settings.chroma_path,
        embedding_client=MockEmbeddingClient(),
        top_k=app_settings.memory_top_k,
    )
    styles = style_store or _build_style_store(app_settings)
    style_pairs = style_pair_store or _build_style_pair_store(app_settings)
    knowledge = knowledge_retriever or _build_knowledge_retriever(app_settings)
    tools = tool_registry or build_default_tool_registry()
    persona = persona_engine or PersonaEngine.from_file(
        app_settings.persona_config_path,
        profile_path=app_settings.style_profile_path,
    )
    style_learner = style_learning_store or _build_style_learning_store(
        app_settings,
        style_store=styles,
        persona=persona,
    )
    llm = llm_client or _build_llm_client(app_settings)
    policy = dialogue_policy or _build_dialogue_policy(app_settings, llm)
    handler = chat_handler or (
        lambda request: run_agent_chat(
            request,
            dialogue_policy=policy,
            review_store=store,
            knowledge_retriever=knowledge,
            memory_store=memories,
            style_store=styles,
            style_pair_store=style_pairs,
            tool_registry=tools,
            persona_engine=persona,
            llm_client=llm,
            rag_top_k=app_settings.rag_top_k,
            memory_top_k=app_settings.memory_top_k,
            style_top_k=app_settings.style_top_k,
            style_pair_top_k=app_settings.style_pair_top_k,
            style_persona_id=app_settings.style_persona_id,
            style_on_smalltalk=app_settings.style_on_smalltalk,
            style_on_private_chat=app_settings.style_on_private_chat,
            auto_memory_on_chat=app_settings.auto_memory_on_chat,
            auto_memory_user_name=app_settings.auto_memory_user_name,
            auto_memory_persona_name=app_settings.auto_memory_persona_name,
            style_learning_store=style_learner,
        )
    )

    # Settings 挂在 app.state 上，后续 /chat、LLM、trace 都从这里读取运行配置。
    app.state.settings = app_settings
    app.state.chat_handler = handler
    app.state.review_store = store
    app.state.knowledge_retriever = knowledge
    app.state.memory_store = memories
    app.state.style_store = styles
    app.state.style_pair_store = style_pairs
    app.state.style_learning_store = style_learner
    app.state.tool_registry = tools
    app.state.persona_engine = persona
    app.state.llm_client = llm
    app.state.dialogue_policy = policy

    @app.get("/health")
    def health() -> dict[str, str]:
        # /health 只证明服务进程可用，不触发 DeepSeek、RAG 或 LiteIM 依赖。
        return {"status": "ok", "service": app_settings.service_name}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            command = await _call_chat_handler(handler, request)
            return ChatResponse(ok=True, command=command)
        except Exception as exc:
            return ChatResponse(
                ok=False,
                command=no_reply_command(request, "agent_service_error"),
                error=ErrorEnvelope(
                    code="agent_service_error",
                    message=str(exc),
                    retryable=True,
                ),
            )

    @app.get("/human-review", response_model=HumanReviewList)
    def list_reviews(
        request: Request,
        status: ReviewStatus | None = None,
        q: str | None = None,
        risk_reason: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> HumanReviewList:
        _review_access_token(app_settings, request)
        return store.list_reviews(
            status=status,
            keyword=q,
            risk_reason=risk_reason,
            limit=limit,
            offset=offset,
        )

    @app.get("/human-review/export")
    def export_reviews(
        request: Request,
        format: str = Query(default="json", pattern="^(json|csv)$"),
        status: ReviewStatus | None = None,
        q: str | None = None,
        risk_reason: str | None = None,
    ) -> Response:
        _review_access_token(app_settings, request)
        listing = store.list_reviews(
            status=status,
            keyword=q,
            risk_reason=risk_reason,
            limit=200,
            offset=0,
        )
        if format == "csv":
            return Response(
                content=_reviews_csv(listing.items),
                media_type="text/csv; charset=utf-8",
            )
        return Response(
            content=listing.model_dump_json(),
            media_type="application/json",
        )

    @app.get("/human-review/ui", response_class=HTMLResponse)
    def review_ui(
        request: Request,
        status: ReviewStatus | None = None,
        q: str | None = None,
        risk_reason: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> HTMLResponse:
        token = _review_access_token(app_settings, request)
        listing = store.list_reviews(
            status=status,
            keyword=q,
            risk_reason=risk_reason,
            limit=limit,
            offset=offset,
        )
        return HTMLResponse(
            _review_list_html(
                listing,
                review_token=token,
                status=status,
                q=q,
                risk_reason=risk_reason,
            )
        )

    @app.get("/human-review/ui/{thread_id}", response_class=HTMLResponse)
    def review_detail_ui(request: Request, thread_id: str) -> HTMLResponse:
        token = _review_access_token(app_settings, request)
        try:
            detail = store.detail(thread_id)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        return HTMLResponse(_review_detail_html(detail, review_token=token))

    @app.get("/human-review/{thread_id}", response_model=HumanReviewDetail)
    def get_review_detail(request: Request, thread_id: str) -> HumanReviewDetail:
        _review_access_token(app_settings, request)
        try:
            return store.detail(thread_id)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc

    @app.post("/human-review/{thread_id}/edit", response_model=HumanReviewRecord)
    def edit_review(
        thread_id: str,
        request: Request,
        payload: EditReviewRequest,
    ) -> HumanReviewRecord:
        _review_access_token(app_settings, request)
        try:
            return store.edit(
                thread_id,
                payload.edited_text,
                operator=payload.operator,
            )
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        except HumanReviewInvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/human-review/{thread_id}/approve", response_model=HumanReviewRecord)
    def approve_review(
        thread_id: str,
        request: Request,
        payload: ApproveReviewRequest | None = None,
    ) -> HumanReviewRecord:
        _review_access_token(app_settings, request)
        try:
            return store.approve(
                thread_id,
                payload.edited_text if payload is not None else None,
                operator=payload.operator if payload is not None else "local-admin",
            )
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        except HumanReviewInvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/human-review/{thread_id}/reject", response_model=HumanReviewRecord)
    def reject_review(request: Request, thread_id: str) -> HumanReviewRecord:
        _review_access_token(app_settings, request)
        try:
            return store.reject(thread_id)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        except HumanReviewInvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/human-review/{thread_id}/resume", response_model=ChatResponse)
    def resume_review(request: Request, thread_id: str) -> ChatResponse:
        _review_access_token(app_settings, request)
        try:
            command = resume_agent_review(thread_id, store)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        return ChatResponse(ok=True, command=command)

    return app


def _build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLMClient(model=settings.llm_model)
    return OpenAILLMClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.llm_request_timeout_seconds,
    )


def _build_dialogue_policy(settings: Settings, llm_client: LLMClient) -> DialoguePolicy:
    mode: DialoguePolicyMode = "llm" if settings.dialogue_policy_mode == "llm" else "rule"
    return DialoguePolicy(
        mode=mode,
        llm_client=llm_client,
        max_retries=settings.dialogue_policy_max_retries,
        timeout_seconds=settings.dialogue_policy_timeout_seconds,
    )


def _build_knowledge_retriever(settings: Settings) -> KnowledgeRetriever:
    retriever = KnowledgeRetriever(
        vector_store=ChromaVectorStore(settings.chroma_path, collection_name="knowledge"),
        embedding_client=MockEmbeddingClient(),
        chunker=RecursiveTextChunker(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        ),
        top_k=settings.rag_top_k,
    )
    docs_path = Path(settings.knowledge_docs_path)
    if docs_path.exists():
        documents = DocumentLoader().load_directory(docs_path)
        if documents:
            retriever.index_documents(documents)
    return retriever


def _build_style_store(settings: Settings) -> StyleStore:
    store = StyleStore(
        chroma_path=settings.chroma_path,
        embedding_client=MockEmbeddingClient(),
        top_k=settings.style_top_k,
    )
    samples_path = Path(settings.style_samples_path)
    if samples_path.exists():
        samples = _load_processed_style_samples(samples_path)
        if samples:
            store.index_samples(samples)
    return store


def _load_processed_style_samples(path: Path) -> list[ProcessedStyleSample]:
    samples: list[ProcessedStyleSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            samples.append(ProcessedStyleSample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError):
            continue
    return samples


def _build_style_pair_store(settings: Settings) -> StylePairStore | None:
    path = Path(settings.style_pairs_path)
    if not path.exists():
        return None
    return StylePairStore.from_jsonl(path)


def _build_style_learning_store(
    settings: Settings,
    *,
    style_store: StyleStore,
    persona: PersonaEngine,
) -> StyleLearningStore | None:
    if not settings.style_reinforcement_enabled:
        return None
    return StyleLearningStore(
        samples_path=settings.style_reinforcement_samples_path,
        style_store=style_store,
        persona_id=settings.style_persona_id or persona.config.persona_id,
        consent_id=settings.style_reinforcement_consent_id,
        subject_user_id=settings.style_reinforcement_subject_user_id,
    )


def _review_access_token(settings: Settings, request: Request) -> str | None:
    if not settings.review_ui_token:
        return None
    expected = f"Bearer {settings.review_ui_token}"
    if request.headers.get("authorization") != expected:
        if request.query_params.get("token") != settings.review_ui_token:
            raise HTTPException(status_code=401, detail="review token required")
    return settings.review_ui_token


def _reviews_csv(records: list[HumanReviewRecord]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "thread_id",
            "run_id",
            "status",
            "risk_reason",
            "edited_text",
            "created_at",
            "updated_at",
        ],
    )
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "thread_id": record.thread_id,
                "run_id": record.run_id,
                "status": record.status.value,
                "risk_reason": record.risk_reason or "",
                "edited_text": record.edited_text or "",
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
        )
    return output.getvalue()


def _review_list_html(
    listing: HumanReviewList,
    *,
    review_token: str | None = None,
    status: ReviewStatus | None = None,
    q: str | None = None,
    risk_reason: str | None = None,
) -> str:
    token_query = _token_query(review_token)
    escaped_token_query = html.escape(token_query, quote=True)
    rows = "\n".join(
        _review_row_html(record, escaped_token_query) for record in listing.items
    )
    if not rows:
        rows = "<tr><td colspan=\"5\">No reviews</td></tr>"
    export_href = _review_export_href(
        review_token=review_token,
        status=status,
        q=q,
        risk_reason=risk_reason,
    )
    token_input = (
        f'<input type="hidden" name="token" value="{html.escape(review_token, quote=True)}">'
        if review_token
        else ""
    )
    pagination = _review_pagination_html(
        listing,
        review_token=review_token,
        status=status,
        q=q,
        risk_reason=risk_reason,
    )
    status_options = _review_status_options(status)
    token_value = json.dumps(review_token or "")
    q_value = html.escape(q or "", quote=True)
    risk_reason_value = html.escape(risk_reason or "", quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Human Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d0d7de; padding: 8px; text-align: left; }}
    .status {{ font-weight: 700; }}
    .toolbar, .pagination {{ display: flex; gap: 12px; margin: 16px 0; align-items: center; }}
    button, input, select {{ padding: 6px 8px; }}
  </style>
</head>
<body>
  <h1>Human Review</h1>
  <form class="toolbar" method="get" action="/human-review/ui">
    {token_input}
    <input type="hidden" name="limit" value="{listing.limit}">
    <select name="status">
      {status_options}
    </select>
    <input name="q" placeholder="search" value="{q_value}">
    <input name="risk_reason" placeholder="risk reason" value="{risk_reason_value}">
    <button type="submit">Filter</button>
    <a href="{html.escape(export_href)}">Export CSV</a>
  </form>
  {pagination}
  <table>
    <thead>
      <tr><th>Thread</th><th>Status</th><th>Risk</th><th>User Message</th><th>Updated</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <script>
    const initialReviewToken = {token_value};
    if (initialReviewToken) {{
      localStorage.setItem('review_ui_token', initialReviewToken);
    }}
    function authHeaders() {{
      const token = initialReviewToken || localStorage.getItem('review_ui_token') || '';
      return token ? {{'Authorization': `Bearer ${{token}}`}} : {{}};
    }}
    async function refreshReviews() {{
      const response = await fetch('/human-review', {{headers: authHeaders()}});
      return response.ok;
    }}
  </script>
</body>
</html>"""


def _review_detail_html(detail: HumanReviewDetail, *, review_token: str | None = None) -> str:
    record = detail.record
    trace = "\n".join(f"<li>{html.escape(item)}</li>" for item in detail.trace_summary)
    final_command_json = detail.final_command.model_dump_json() if detail.final_command else "{}"
    audit = "\n".join(
        (
            "<li>"
            f"{html.escape(entry.created_at)} "
            f"{html.escape(entry.operator)} "
            f"{html.escape(entry.action)}"
            "</li>"
        )
        for entry in detail.audit_log
    )
    token_value = json.dumps(review_token or "")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Human Review {html.escape(record.thread_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    textarea {{ width: 100%; min-height: 120px; }}
    section {{ margin-bottom: 20px; }}
    button {{ padding: 8px 10px; margin-right: 8px; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <a href="/human-review/ui{_token_query(review_token)}">Back</a>
  <h1>{html.escape(record.thread_id)}</h1>
  <p>Status: <strong>{html.escape(record.status.value)}</strong></p>
  <p>Risk: {html.escape(record.risk_reason or '')}</p>
  <section>
    <h2>User Message</h2>
    <pre>{html.escape(record.request.text)}</pre>
  </section>
  <section>
    <h2>Agent Draft</h2>
    <pre>{html.escape(detail.agent_draft)}</pre>
  </section>
  <section>
    <h2>Final Command</h2>
    <pre>{html.escape(final_command_json)}</pre>
  </section>
  <section>
    <h2>Review Action</h2>
    <textarea id="edited">{html.escape(record.edited_text or detail.agent_draft)}</textarea>
    <button id="edit">edit</button>
    <button id="approve">approve</button>
    <button id="reject">reject</button>
    <button id="resume">resume</button>
    <button id="edit-approve-resume">edit + approve + resume</button>
    <span id="status"></span>
  </section>
  <section><h2>Trace Summary</h2><ul>{trace}</ul></section>
  <section><h2>Audit Log</h2><ul>{audit}</ul></section>
  <script>
    const threadId = {record.thread_id!r};
    const initialReviewToken = {token_value};
    if (initialReviewToken) {{
      localStorage.setItem('review_ui_token', initialReviewToken);
    }}
    const edited = document.getElementById('edited');
    const status = document.getElementById('status');
    function authHeaders(extra) {{
      const token = initialReviewToken || localStorage.getItem('review_ui_token') || '';
      const headers = token ? {{'Authorization': `Bearer ${{token}}`}} : {{}};
      return Object.assign(headers, extra || {{}});
    }}
    async function post(path, body) {{
      const response = await fetch(path, {{
        method: 'POST',
        headers: authHeaders({{'Content-Type': 'application/json'}}),
        body: body === undefined ? undefined : JSON.stringify(body)
      }});
      status.textContent = response.ok ? 'ok' : 'failed';
      return response;
    }}
    document.getElementById('edit').onclick = () =>
      post(`/human-review/${{threadId}}/edit`, {{edited_text: edited.value}});
    document.getElementById('approve').onclick = () =>
      post(`/human-review/${{threadId}}/approve`, {{edited_text: edited.value}});
    document.getElementById('reject').onclick = () => post(`/human-review/${{threadId}}/reject`);
    document.getElementById('resume').onclick = () => post(`/human-review/${{threadId}}/resume`);
    document.getElementById('edit-approve-resume').onclick = async () => {{
      await post(`/human-review/${{threadId}}/edit`, {{edited_text: edited.value}});
      await post(`/human-review/${{threadId}}/approve`, {{edited_text: edited.value}});
      await post(`/human-review/${{threadId}}/resume`);
    }};
  </script>
</body>
</html>"""


def _review_row_html(record: HumanReviewRecord, escaped_token_query: str) -> str:
    thread_id = html.escape(record.thread_id)
    return (
        "<tr>"
        f'<td><a href="/human-review/ui/{thread_id}{escaped_token_query}">'
        f"{thread_id}</a></td>"
        f'<td><span class="status">{html.escape(record.status.value)}</span></td>'
        f"<td>{html.escape(record.risk_reason or '')}</td>"
        f"<td>{html.escape(record.request.text)}</td>"
        f"<td>{html.escape(record.updated_at)}</td>"
        "</tr>"
    )


def _token_query(review_token: str | None) -> str:
    return _token_arg(review_token, separator="?")


def _token_arg(review_token: str | None, *, separator: str) -> str:
    if not review_token:
        return ""
    return f"{separator}token={quote(review_token)}"


def _review_status_options(selected: ReviewStatus | None) -> str:
    selected_value = selected.value if selected is not None else ""
    options = [("", "all"), *[(status.value, status.value) for status in ReviewStatus]]
    return "\n      ".join(
        (
            f'<option value="{html.escape(value)}"{_selected_attr(value, selected_value)}>'
            f"{html.escape(label)}</option>"
        )
        for value, label in options
    )


def _selected_attr(value: str, selected_value: str) -> str:
    return " selected" if value == selected_value else ""


def _review_pagination_html(
    listing: HumanReviewList,
    *,
    review_token: str | None,
    status: ReviewStatus | None,
    q: str | None,
    risk_reason: str | None,
) -> str:
    prev_offset = max(listing.offset - listing.limit, 0)
    next_offset = listing.offset + listing.limit
    prev_html = (
        _review_page_link(
            label="Prev",
            review_token=review_token,
            status=status,
            q=q,
            risk_reason=risk_reason,
            limit=listing.limit,
            offset=prev_offset,
        )
        if listing.offset > 0
        else "<span>Prev</span>"
    )
    next_html = (
        _review_page_link(
            label="Next",
            review_token=review_token,
            status=status,
            q=q,
            risk_reason=risk_reason,
            limit=listing.limit,
            offset=next_offset,
        )
        if next_offset < listing.total
        else "<span>Next</span>"
    )
    page_end = min(listing.offset + len(listing.items), listing.total)
    return (
        '<nav class="pagination">'
        f"{prev_html}"
        f"<span>{listing.offset}-{page_end} / {listing.total}</span>"
        f"{next_html}"
        "</nav>"
    )


def _review_page_link(
    *,
    label: str,
    review_token: str | None,
    status: ReviewStatus | None,
    q: str | None,
    risk_reason: str | None,
    limit: int,
    offset: int,
) -> str:
    href = _review_ui_href(
        review_token=review_token,
        status=status,
        q=q,
        risk_reason=risk_reason,
        limit=limit,
        offset=offset,
    )
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def _review_ui_href(
    *,
    review_token: str | None,
    status: ReviewStatus | None,
    q: str | None,
    risk_reason: str | None,
    limit: int,
    offset: int,
) -> str:
    params: list[tuple[str, str | int]] = []
    if status is not None:
        params.append(("status", status.value))
    if q:
        params.append(("q", q))
    if risk_reason:
        params.append(("risk_reason", risk_reason))
    params.extend([("limit", limit), ("offset", offset)])
    if review_token:
        params.append(("token", review_token))
    return f"/human-review/ui?{urlencode(params)}"


def _review_export_href(
    *,
    review_token: str | None,
    status: ReviewStatus | None,
    q: str | None,
    risk_reason: str | None,
) -> str:
    params: list[tuple[str, str]] = [("format", "csv")]
    if status is not None:
        params.append(("status", status.value))
    if q:
        params.append(("q", q))
    if risk_reason:
        params.append(("risk_reason", risk_reason))
    if review_token:
        params.append(("token", review_token))
    return f"/human-review/export?{urlencode(params)}"


app = create_app()
