"""The research agent, reached over A2A instead of in-process (RESEARCH_MODE=a2a).

To the supervisor it looks exactly like the local research agent: the same
AgentState in and out. Inside, the one node sends an A2A message to the remote
service and reads the "findings" artifact from the completed task.

    START -> call_remote -> END
"""

from collections.abc import Callable

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text
from a2a.types import SendMessageRequest, TaskState
from langgraph.graph import END, START, StateGraph

from orchestraos.a2a.protocol import read_result, task_message
from orchestraos.agents.base import Agent, AgentState
from orchestraos.agents.common import AgentError

# Research on a laptop can take minutes (model + several page fetches).
TIMEOUT = httpx.Timeout(10, read=600)

HttpClientFactory = Callable[[], httpx.AsyncClient]


def build_a2a_research_agent(
    url: str, http_client: HttpClientFactory = lambda: httpx.AsyncClient(timeout=TIMEOUT)
) -> Agent:
    async def call_remote(state: AgentState) -> AgentState:
        request = SendMessageRequest(
            message=task_message(
                state["instruction"], state.get("goal", ""), state.get("context", "")
            )
        )
        async with http_client() as http:
            # Non-streaming: one request, one response holding the finished task.
            config = ClientConfig(streaming=False, httpx_client=http)
            client = await create_client(url, client_config=config)  # fetches the Agent Card
            task = None
            async for response in client.send_message(request):
                if response.HasField("task"):
                    task = response.task
            await client.close()

        if task is None:
            raise AgentError("the research service returned no task")
        if task.status.state != TaskState.TASK_STATE_COMPLETED:
            reason = (
                get_message_text(task.status.message) if task.status.HasField("message") else ""
            )
            raise AgentError(f"remote research {TaskState.Name(task.status.state)}: {reason}")
        result = read_result(list(task.artifacts))
        return {**result, "artifacts": []}

    graph = StateGraph(AgentState)
    graph.add_node("call_remote", call_remote)
    graph.add_edge(START, "call_remote")
    graph.add_edge("call_remote", END)
    return graph.compile(name="research_agent_a2a")
