"""FastAPI backend.

GET  /health                    Ollama + models status
POST /runs        {goal}        start a run -> {thread_id}
GET  /runs/{id}/events?after=n  SSE stream of the run's events (replays from n)
GET  /runs/{id}                 run state from the checkpointer
POST /runs/{id}/resume          continue a run that stopped part-way
POST /runs/{id}/approval        approve or reject the pending action {approved, note}
GET  /files                    list workspace files
POST /files                     upload a file into the workspace
GET  /files/{path}              download a workspace file (e.g. a chart)
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from orchestraos import __version__
from orchestraos.api.runs import RunManager, RunNotFound
from orchestraos.config import Settings, get_settings
from orchestraos.ollama import OllamaUnavailable, list_local_models, missing_models
from orchestraos.tools.workspace import Workspace, WorkspaceError

GraphFactory = Callable[[], AbstractAsyncContextManager[CompiledStateGraph]]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class OllamaStatus(BaseModel):
    reachable: bool
    missing_models: list[str] = []
    error: str | None = None


class HealthResponse(BaseModel):
    # "degraded" = the API is up but can't do useful work (Ollama down or models missing).
    status: Literal["ok", "degraded"]
    version: str
    ollama: OllamaStatus


class RunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)


class RunStarted(BaseModel):
    thread_id: str


class Decision(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=1000)


def create_app(
    open_graph: GraphFactory | None = None, workspace_dir: Path | None = None
) -> FastAPI:
    """`open_graph` builds the supervisor (default: the real one); tests pass fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if open_graph is None:
            from orchestraos.graph.factory import open_app  # starts MCP servers, Ollama clients

            settings = get_settings()
            graph_context, root = open_app(settings), settings.workspace_dir
        else:
            graph_context, root = open_graph(), workspace_dir or Path("workspace")
        app.state.workspace = Workspace(root)
        async with graph_context as graph:
            app.state.runs = RunManager(graph)
            yield

    app = FastAPI(title="OrchestraOS API", version=__version__, lifespan=lifespan)

    def runs(request: Request) -> RunManager:
        return request.app.state.runs

    def workspace(request: Request) -> Workspace:
        return request.app.state.workspace

    Runs = Annotated[RunManager, Depends(runs)]
    Files = Annotated[Workspace, Depends(workspace)]

    @app.get("/health", response_model=HealthResponse)
    def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        # A plain `def` endpoint runs in FastAPI's threadpool, so the blocking call is fine.
        try:
            available = list_local_models(settings.ollama_base_url)
        except OllamaUnavailable as exc:
            return HealthResponse(
                status="degraded",
                version=__version__,
                ollama=OllamaStatus(reachable=False, error=str(exc)),
            )
        missing = missing_models(settings.required_models(), available)
        return HealthResponse(
            status="degraded" if missing else "ok",
            version=__version__,
            ollama=OllamaStatus(reachable=True, missing_models=missing),
        )

    @app.post("/runs", status_code=202, response_model=RunStarted)
    async def start_run(body: RunRequest, manager: Runs) -> RunStarted:
        return RunStarted(thread_id=manager.start(body.goal).thread_id)

    @app.get("/runs/{thread_id}/events")
    async def run_events(thread_id: str, manager: Runs, after: int = 0) -> EventSourceResponse:
        if thread_id not in manager.runs:
            raise HTTPException(404, f"no live run {thread_id}; GET /runs/{thread_id} for state")

        async def stream():
            async for index, event in manager.events(thread_id, after):
                yield {"id": str(index), "event": event["type"], "data": json.dumps(event)}

        return EventSourceResponse(stream())

    @app.get("/runs/{thread_id}")
    async def get_run(thread_id: str, manager: Runs) -> dict:
        try:
            return await manager.snapshot(thread_id)
        except RunNotFound:
            raise HTTPException(404, f"no run {thread_id}") from None

    @app.post("/runs/{thread_id}/resume", status_code=202, response_model=RunStarted)
    async def resume_run(thread_id: str, manager: Runs) -> RunStarted:
        try:
            await manager.resume(thread_id)
        except RunNotFound:
            raise HTTPException(404, f"no run {thread_id}") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return RunStarted(thread_id=thread_id)

    @app.post("/runs/{thread_id}/approval", status_code=202, response_model=RunStarted)
    async def decide(thread_id: str, body: Decision, manager: Runs) -> RunStarted:
        try:
            await manager.decide(thread_id, body.approved, body.note)
        except RunNotFound:
            raise HTTPException(404, f"no run {thread_id}") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return RunStarted(thread_id=thread_id)

    @app.get("/files")
    def list_files(files: Files) -> list[str]:
        return files.list_files()

    @app.post("/files", status_code=201)
    async def upload_file(upload: UploadFile, files: Files) -> dict:
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large (max 20 MB)")
        try:
            target = files.resolve(Path(upload.filename or "").name)
        except WorkspaceError as exc:
            raise HTTPException(400, str(exc)) from None
        if target == files.root:
            raise HTTPException(400, "missing file name")
        target.write_bytes(data)
        return {"path": files.relative(target)}

    @app.get("/files/{path:path}")
    def get_file(path: str, files: Files) -> FileResponse:
        try:
            target = files.resolve(path)
        except WorkspaceError as exc:
            raise HTTPException(400, str(exc)) from None
        if not target.is_file():
            raise HTTPException(404, f"no such file: {path}")
        return FileResponse(target)

    return app


app = create_app()
