"""Test doubles shared across test modules."""

from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage

from orchestraos.llm.router import ModelRouter, NamedModel


class FakeChat:
    """A chat model whose reply is computed from the messages it receives."""

    def __init__(self, respond: Callable[[list[BaseMessage]], str]):
        self.respond = respond
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, input: list[BaseMessage]) -> AIMessage:
        self.calls.append(input)
        return AIMessage(self.respond(input))


def system_prompt(messages: list[BaseMessage]) -> str:
    return str(messages[0].content) if messages else ""


def fake_router(respond: Callable[[list[BaseMessage]], str]) -> ModelRouter:
    """Both models answer the same way, so tests don't depend on which one is picked."""
    return ModelRouter(
        small=NamedModel("small", FakeChat(respond)),
        large=NamedModel("large", FakeChat(respond)),
    )


class FakeTools:
    """A ToolCaller backed by plain functions: FakeTools(web_search=lambda query, **_: [...])."""

    def __init__(self, **handlers):
        self.handlers = handlers
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, **args):
        self.calls.append((name, args))
        if name not in self.handlers:
            raise AssertionError(f"unexpected tool call: {name}")
        return self.handlers[name](**args)

    def called(self, name: str) -> list[dict]:
        return [args for called_name, args in self.calls if called_name == name]
