"""Tracing with Langfuse (self-hosted, `observability` Compose profile). Off unless configured.

One LangChain callback handler, attached to the supervisor graph, traces every
step of a run: each supervisor node, each agent subgraph and its nodes, every
LLM call (prompt, output, tokens, latency) and every MCP tool call. Runs are
grouped into a Langfuse *session* per thread id, so a run paused for approval
and resumed later shows up as one session.

Off unless TRACING=true and the Langfuse keys are set; then nothing else is sent anywhere.
"""

import logging
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from orchestraos import __version__
from orchestraos.config import Settings

logger = logging.getLogger(__name__)

EXPORT_TIMEOUT_SECONDS = 30


def tracing_enabled(settings: Settings) -> bool:
    return settings.tracing and bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def make_handler(settings: Settings) -> Any | None:
    """A Langfuse callback handler, or None when tracing isn't configured."""
    if not tracing_enabled(settings):
        return None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    # Creating the client registers it; the handler finds it by public key.
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment="local",
        release=__version__,
        # The SDK default is 5s. With a 7B model saturating a laptop, the final (largest)
        # batch took longer, the upload was aborted mid-request, and the end of the trace
        # (writer, finalize, root span) was silently dropped.
        timeout=EXPORT_TIMEOUT_SECONDS,
    )
    return CallbackHandler(public_key=settings.langfuse_public_key)


def traced(graph: CompiledStateGraph, settings: Settings) -> CompiledStateGraph:
    """The same graph with tracing attached (or unchanged when tracing is off)."""
    handler = make_handler(settings)
    if handler is None:
        return graph
    logger.info("tracing runs to Langfuse at %s", settings.langfuse_base_url)
    return graph.with_config(callbacks=[handler])


def flush(settings: Settings) -> None:
    """Send buffered traces (call on shutdown so the last run isn't lost)."""
    if not tracing_enabled(settings):
        return
    from langfuse import get_client

    get_client(public_key=settings.langfuse_public_key).flush()


def session_metadata(thread_id: str) -> dict[str, str]:
    """Run config metadata that groups a thread's traces into one Langfuse session."""
    return {"langfuse_session_id": thread_id}
