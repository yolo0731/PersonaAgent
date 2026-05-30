from __future__ import annotations

import csv
import html
import io
import json
from urllib.parse import quote, urlencode

from agent_service.review import HumanReviewDetail, HumanReviewList, HumanReviewRecord, ReviewStatus


def reviews_csv(records: list[HumanReviewRecord]) -> str:
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


def review_list_html(
    listing: HumanReviewList,
    *,
    review_token: str | None = None,
    status: ReviewStatus | None = None,
    q: str | None = None,
    risk_reason: str | None = None,
) -> str:
    token_query = token_query_string(review_token)
    escaped_token_query = html.escape(token_query, quote=True)
    rows = "\n".join(review_row_html(record, escaped_token_query) for record in listing.items)
    if not rows:
        rows = "<tr><td colspan=\"5\">No reviews</td></tr>"
    export_href = review_export_href(
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
    pagination = review_pagination_html(
        listing,
        review_token=review_token,
        status=status,
        q=q,
        risk_reason=risk_reason,
    )
    status_options = review_status_options(status)
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


def review_detail_html(detail: HumanReviewDetail, *, review_token: str | None = None) -> str:
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
  <a href="/human-review/ui{token_query_string(review_token)}">Back</a>
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


def review_row_html(record: HumanReviewRecord, escaped_token_query: str) -> str:
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


def token_query_string(review_token: str | None) -> str:
    return token_arg(review_token, separator="?")


def token_arg(review_token: str | None, *, separator: str) -> str:
    if not review_token:
        return ""
    return f"{separator}token={quote(review_token)}"


def review_status_options(selected: ReviewStatus | None) -> str:
    selected_value = selected.value if selected is not None else ""
    options = [("", "all"), *[(status.value, status.value) for status in ReviewStatus]]
    return "\n      ".join(
        (
            f'<option value="{html.escape(value)}"{selected_attr(value, selected_value)}>'
            f"{html.escape(label)}</option>"
        )
        for value, label in options
    )


def selected_attr(value: str, selected_value: str) -> str:
    return " selected" if value == selected_value else ""


def review_pagination_html(
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
        review_page_link(
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
        review_page_link(
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


def review_page_link(
    *,
    label: str,
    review_token: str | None,
    status: ReviewStatus | None,
    q: str | None,
    risk_reason: str | None,
    limit: int,
    offset: int,
) -> str:
    href = review_ui_href(
        review_token=review_token,
        status=status,
        q=q,
        risk_reason=risk_reason,
        limit=limit,
        offset=offset,
    )
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def review_ui_href(
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


def review_export_href(
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
