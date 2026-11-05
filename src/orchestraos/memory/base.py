"""The interface the supervisor uses to talk to the memory agent."""

from typing import Protocol


class Memory(Protocol):
    async def recall(self, goal: str) -> str:
        """Context relevant to `goal`, as text for the planner prompt ('' if nothing)."""
        ...

    async def remember(self, goal: str, answer: str) -> None:
        """Store what was learned from a finished run."""
        ...


class NullMemory:
    """Remembers nothing. Used until the real memory layer is wired in, and in tests."""

    async def recall(self, goal: str) -> str:
        return ""

    async def remember(self, goal: str, answer: str) -> None:
        return None
