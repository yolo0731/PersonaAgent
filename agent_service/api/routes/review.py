from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from agent_service.api.review_ui import review_detail_html, review_list_html, reviews_csv
from agent_service.config import Settings
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
from agent_service.schemas import ChatResponse
from agent_service.workflow import resume_agent_review


def register_review_routes(
    app: FastAPI,
    *,
    settings: Settings,
    store: HumanReviewStore,
) -> None:
    @app.get("/human-review", response_model=HumanReviewList)
    def list_reviews(
        request: Request,
        status: ReviewStatus | None = None,
        q: str | None = None,
        risk_reason: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> HumanReviewList:
        review_access_token(settings, request)
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
        review_access_token(settings, request)
        listing = store.list_reviews(
            status=status,
            keyword=q,
            risk_reason=risk_reason,
            limit=200,
            offset=0,
        )
        if format == "csv":
            return Response(
                content=reviews_csv(listing.items),
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
        token = review_access_token(settings, request)
        listing = store.list_reviews(
            status=status,
            keyword=q,
            risk_reason=risk_reason,
            limit=limit,
            offset=offset,
        )
        return HTMLResponse(
            review_list_html(
                listing,
                review_token=token,
                status=status,
                q=q,
                risk_reason=risk_reason,
            )
        )

    @app.get("/human-review/ui/{thread_id}", response_class=HTMLResponse)
    def review_detail_ui(request: Request, thread_id: str) -> HTMLResponse:
        token = review_access_token(settings, request)
        try:
            detail = store.detail(thread_id)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        return HTMLResponse(review_detail_html(detail, review_token=token))

    @app.get("/human-review/{thread_id}", response_model=HumanReviewDetail)
    def get_review_detail(request: Request, thread_id: str) -> HumanReviewDetail:
        review_access_token(settings, request)
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
        review_access_token(settings, request)
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
        review_access_token(settings, request)
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
        review_access_token(settings, request)
        try:
            return store.reject(thread_id)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        except HumanReviewInvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/human-review/{thread_id}/resume", response_model=ChatResponse)
    def resume_review(request: Request, thread_id: str) -> ChatResponse:
        review_access_token(settings, request)
        try:
            command = resume_agent_review(thread_id, store)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        return ChatResponse(ok=True, command=command)


def review_access_token(settings: Settings, request: Request) -> str | None:
    if not settings.review_ui_token:
        return None
    expected = f"Bearer {settings.review_ui_token}"
    if request.headers.get("authorization") != expected:
        if request.query_params.get("token") != settings.review_ui_token:
            raise HTTPException(status_code=401, detail="review token required")
    return settings.review_ui_token
