"""FastAPI application that exposes the ToT scheduler over HTTP."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fsm import (
    DEFAULT_CHAT_API_URL,
    DEFAULT_MODELING_MODEL,
    DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    DEFAULT_PLANNING_MODEL,
    DEFAULT_REVIEW_MODEL,
    ChatBackendError,
    DeterministicContextBackendAdapter,
    NodeDeletionReviewAdapter,
    ReasoningBackendAdapter,
    ToTTreeScheduler,
    build_local_chat_adapter_bundle,
)

SchedulerOperationResult = TypeVar("SchedulerOperationResult")
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

DEFAULT_PROBLEM_CONTEXT_DRAFT: dict[str, Any] = {
    "task": (
        "Use the modeling model to propose the next reasoning step, then score each step for physical consistency "
        "and variable grounding."
    ),
    "notes": [
        "The frontend polls the live scheduler state and renders it as an ASCII tree.",
        "Node deletion is routed through the backend review model before the subtree is removed.",
    ],
    "known_context": {
        "objective": "Derive and prune a useful physical reasoning tree.",
        "expected_output": "concise, structured, and physically valid intermediate steps",
    },
}

AdapterBundleFactory = Callable[
    ["ChatBackendConfig"],
    tuple[Callable[[dict[str, Any]], ReasoningBackendAdapter], NodeDeletionReviewAdapter],
]


class ChatBackendConfig(BaseModel):
    base_url: str = DEFAULT_CHAT_API_URL
    planning_model: str = DEFAULT_PLANNING_MODEL
    modeling_model: str = DEFAULT_MODELING_MODEL
    review_model: str = DEFAULT_REVIEW_MODEL
    non_terminal_evaluation_model: str = DEFAULT_NON_TERMINAL_EVALUATION_MODEL
    timeout: float = Field(default=60.0, gt=0.0)


class SchedulerConfig(BaseModel):
    max_reflections: int = Field(default=2, ge=0)
    expansion_budget: int = Field(default=8, ge=0)
    max_frontier_size: int = Field(default=8, ge=1)
    max_children_per_expansion: int = Field(default=4, ge=1)
    max_frontier_per_diversity_key: int = Field(default=2, ge=1)
    children_key: str = "children"


class CreateSessionRequest(BaseModel):
    problem_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context merged with problem_statement. See /api/tot/defaults for the UI draft shape.",
    )
    backend: ChatBackendConfig = Field(default_factory=ChatBackendConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    run_on_create: bool = True


class FrontendDefaultsResponse(BaseModel):
    problem_context: dict[str, Any] = Field(
        default_factory=lambda: deepcopy(DEFAULT_PROBLEM_CONTEXT_DRAFT)
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class RunSessionRequest(BaseModel):
    additional_budget: int = Field(default=0, ge=0)


class DeleteNodeRequest(BaseModel):
    reason: str
    requested_by: str = "frontend"


class SessionStateResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class DeleteNodeResponse(BaseModel):
    session_id: str
    deleted: bool
    node_id: str
    deleted_node_ids: list[str] = Field(default_factory=list)
    review: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any]


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class SchedulerSessionStore:
    """In-memory session store for live ToT schedulers."""

    def __init__(self) -> None:
        self._sessions: dict[str, ToTTreeScheduler] = {}
        self._session_locks: dict[str, RLock] = {}
        self._session_snapshots: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def create(self, scheduler: ToTTreeScheduler) -> str:
        session_id = uuid4().hex
        snapshot = jsonable_encoder(scheduler.snapshot())
        with self._lock:
            self._sessions[session_id] = scheduler
            self._session_locks[session_id] = RLock()
            self._session_snapshots[session_id] = snapshot
        return session_id

    def get(self, session_id: str) -> ToTTreeScheduler:
        with self._lock:
            scheduler = self._sessions.get(session_id)
        if scheduler is None:
            raise KeyError(session_id)
        return scheduler

    def snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            scheduler = self._sessions.get(session_id)
            session_lock = self._session_locks.get(session_id)
            cached_snapshot = deepcopy(self._session_snapshots.get(session_id, {}))
        if scheduler is None or session_lock is None:
            raise KeyError(session_id)
        if not session_lock.acquire(blocking=False):
            return self._merge_snapshot_with_live_run_state(cached_snapshot, scheduler)
        try:
            snapshot = jsonable_encoder(scheduler.snapshot())
        finally:
            session_lock.release()

        with self._lock:
            self._session_snapshots[session_id] = snapshot
        return deepcopy(snapshot)

    def execute(
        self,
        session_id: str,
        operation: Callable[[ToTTreeScheduler], SchedulerOperationResult],
    ) -> SchedulerOperationResult:
        with self._lock:
            scheduler = self._sessions.get(session_id)
            session_lock = self._session_locks.get(session_id)
        if scheduler is None or session_lock is None:
            raise KeyError(session_id)
        with session_lock:
            result = operation(scheduler)
            snapshot = jsonable_encoder(scheduler.snapshot())
        with self._lock:
            self._session_snapshots[session_id] = snapshot
        return result

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session_lock = self._session_locks.get(session_id)
        if session_lock is None:
            return False
        with session_lock:
            with self._lock:
                deleted = self._sessions.pop(session_id, None) is not None
                self._session_locks.pop(session_id, None)
                self._session_snapshots.pop(session_id, None)
                return deleted

    def _merge_snapshot_with_live_run_state(
        self,
        cached_snapshot: dict[str, Any],
        scheduler: ToTTreeScheduler,
    ) -> dict[str, Any]:
        snapshot = deepcopy(cached_snapshot) if cached_snapshot else {}
        snapshot["run_state"] = {
            "status": str(getattr(scheduler, "run_status", "idle")),
            "phase": str(getattr(scheduler, "run_phase", "created")),
            "problem_context_prepared": bool(getattr(scheduler, "_problem_context_prepared", False)),
            "auto_run_requested": bool(getattr(scheduler, "auto_run_requested", False)),
            "target_expansion_budget": int(
                getattr(scheduler, "target_expansion_budget", getattr(scheduler, "expansion_budget", 0))
            ),
            "last_error": str(getattr(scheduler, "last_error", "")),
        }
        return snapshot


def _raise_backend_http_error(exc: ChatBackendError) -> None:
    raise HTTPException(status_code=502, detail=str(exc)) from exc


def _get_session_or_404(app: FastAPI, session_id: str) -> ToTTreeScheduler:
    try:
        return app.state.session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _get_session_state_or_404(app: FastAPI, session_id: str) -> dict[str, Any]:
    try:
        return _serialize_session_state(app.state.session_store, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _execute_session_or_404(
    app: FastAPI,
    session_id: str,
    operation: Callable[[ToTTreeScheduler], SchedulerOperationResult],
) -> SchedulerOperationResult:
    try:
        return app.state.session_store.execute(session_id, operation)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _default_adapter_bundle_factory(
    config: ChatBackendConfig,
) -> tuple[Callable[[dict[str, Any]], ReasoningBackendAdapter], NodeDeletionReviewAdapter]:
    return build_local_chat_adapter_bundle(
        base_url=config.base_url,
        timeout=config.timeout,
        planning_model=config.planning_model,
        modeling_model=config.modeling_model,
        review_model=config.review_model,
        non_terminal_evaluation_model=config.non_terminal_evaluation_model,
    )


def _serialize_state(scheduler: ToTTreeScheduler) -> dict[str, Any]:
    return jsonable_encoder(scheduler.snapshot())


def _serialize_session_state(store: SchedulerSessionStore, session_id: str) -> dict[str, Any]:
    return store.snapshot(session_id)


def _run_scheduler(scheduler: ToTTreeScheduler, additional_budget: int) -> ToTTreeScheduler:
    scheduler.target_expansion_budget = max(
        int(getattr(scheduler, "target_expansion_budget", scheduler.expansion_budget)),
        scheduler.expansion_budget + additional_budget,
    )
    scheduler.expansion_budget += additional_budget
    scheduler.run()
    return scheduler


def _progressive_auto_run_session(
    session_store: SchedulerSessionStore,
    session_id: str,
    target_budget: int,
) -> None:
    try:
        session_store.execute(session_id, lambda scheduler: _run_scheduler(scheduler, 0))
        for _ in range(max(0, target_budget)):
            state = session_store.execute(
                session_id,
                lambda scheduler: _run_scheduler(scheduler, 1).snapshot(),
            )
            if not state.get("frontier"):
                break
    except Exception as exc:
        try:
            session_store.execute(
                session_id,
                lambda scheduler: getattr(scheduler, "_set_run_state", lambda **_: None)(
                    status="error",
                    phase="error",
                    last_error=str(exc),
                ) or scheduler,
            )
        except KeyError:
            pass
    finally:
        try:
            session_store.execute(
                session_id,
                lambda scheduler: setattr(scheduler, "auto_run_requested", False) or scheduler,
            )
        except KeyError:
            pass


def _start_progressive_auto_run(
    session_store: SchedulerSessionStore,
    session_id: str,
    target_budget: int,
) -> None:
    Thread(
        target=_progressive_auto_run_session,
        args=(session_store, session_id, target_budget),
        daemon=True,
    ).start()


def _delete_scheduler_node(
    scheduler: ToTTreeScheduler,
    *,
    node_id: str,
    request: DeleteNodeRequest,
) -> tuple[ToTTreeScheduler, dict[str, Any]]:
    result = scheduler.delete_node(
        node_id,
        reason=request.reason,
        requested_by=request.requested_by,
    )
    return scheduler, result


def create_app(
    *,
    session_store: Optional[SchedulerSessionStore] = None,
    adapter_bundle_factory: Optional[AdapterBundleFactory] = None,
) -> FastAPI:
    app = FastAPI(title="ToT API", version="0.1.0")
    app.state.session_store = session_store or SchedulerSessionStore()
    app.state.adapter_bundle_factory = adapter_bundle_factory or _default_adapter_bundle_factory
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend-static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_INDEX)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/tot/defaults", response_model=FrontendDefaultsResponse)
    def get_defaults() -> FrontendDefaultsResponse:
        return FrontendDefaultsResponse()

    @app.post("/api/tot/sessions", response_model=SessionStateResponse)
    def create_session(request: CreateSessionRequest) -> SessionStateResponse:
        try:
            backend_factory, deletion_review_adapter = app.state.adapter_bundle_factory(request.backend)
            scheduler = ToTTreeScheduler(
                root_problem_context=deepcopy(request.problem_context),
                max_reflections=request.scheduler.max_reflections,
                expansion_budget=0,
                max_frontier_size=request.scheduler.max_frontier_size,
                max_children_per_expansion=request.scheduler.max_children_per_expansion,
                max_frontier_per_diversity_key=request.scheduler.max_frontier_per_diversity_key,
                children_key=request.scheduler.children_key,
                backend_adapter_factory=backend_factory,
                deletion_review_adapter=deletion_review_adapter,
            )
            scheduler.target_expansion_budget = request.scheduler.expansion_budget
            scheduler.auto_run_requested = bool(request.run_on_create)
            scheduler.run_status = "busy" if request.run_on_create else "idle"
            scheduler.run_phase = "queued" if request.run_on_create else "created"
        except ChatBackendError as exc:
            _raise_backend_http_error(exc)
        session_id = app.state.session_store.create(scheduler)
        state = _serialize_session_state(app.state.session_store, session_id)
        if request.run_on_create:
            _start_progressive_auto_run(
                app.state.session_store,
                session_id,
                request.scheduler.expansion_budget,
            )
        return SessionStateResponse(session_id=session_id, state=state)

    @app.get("/api/tot/sessions/{session_id}", response_model=SessionStateResponse)
    def get_session(session_id: str) -> SessionStateResponse:
        return SessionStateResponse(session_id=session_id, state=_get_session_state_or_404(app, session_id))

    @app.post("/api/tot/sessions/{session_id}/run", response_model=SessionStateResponse)
    def run_session(session_id: str, request: RunSessionRequest) -> SessionStateResponse:
        try:
            _execute_session_or_404(
                app,
                session_id,
                lambda current_scheduler: _run_scheduler(current_scheduler, request.additional_budget),
            )
        except ChatBackendError as exc:
            _raise_backend_http_error(exc)
        return SessionStateResponse(session_id=session_id, state=_get_session_state_or_404(app, session_id))

    @app.delete("/api/tot/sessions/{session_id}/nodes/{node_id}", response_model=DeleteNodeResponse)
    def delete_node(session_id: str, node_id: str, request: DeleteNodeRequest) -> DeleteNodeResponse:
        try:
            scheduler, result = _execute_session_or_404(
                app,
                session_id,
                lambda current_scheduler: _delete_scheduler_node(
                    current_scheduler,
                    node_id=node_id,
                    request=request,
                ),
            )
        except ChatBackendError as exc:
            _raise_backend_http_error(exc)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return DeleteNodeResponse(
            session_id=session_id,
            deleted=bool(result["deleted"]),
            node_id=str(result["node_id"]),
            deleted_node_ids=list(result["deleted_node_ids"]),
            review=dict(result["review"]),
            state=_serialize_state(scheduler),
        )

    @app.delete("/api/tot/sessions/{session_id}", response_model=SessionDeleteResponse)
    def delete_session(session_id: str) -> SessionDeleteResponse:
        deleted = app.state.session_store.delete(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found.")
        return SessionDeleteResponse(session_id=session_id, deleted=True)

    return app


app = create_app()


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()