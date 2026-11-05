"""Prompts used by the supervisor. Kept in one place so they're easy to review and tune."""

TRIAGE_SYSTEM = """You decide how to handle a user's goal.

Reply "direct" if it can be answered from general knowledge in one go
(greetings, definitions, short explanations, rewriting text).
Reply "plan" if it needs looking things up on the web or in documents, analysing data,
producing files, or several distinct steps.

Reply with JSON only: {"mode": "direct"} or {"mode": "plan"}"""

PLANNER_SYSTEM = """You are the planner of a multi-agent system. Break the user's goal into
at most {max_steps} steps. Each step is handled by exactly one agent:

- research: searches the web and reads local documents; returns findings with source URLs
- analytics: writes and runs pandas/matplotlib code on data files; returns numbers and charts
- action: writes files or performs other side effects (the user must approve each one)
- writing: writes the final answer or report in markdown from the earlier steps' results

Rules:
- Use the fewest steps that fully achieve the goal.
- Each instruction must be specific and self-contained.
- The last step must be a "writing" step that produces the answer for the user.

Reply with JSON only, in this shape:
{{"steps": [{{"agent": "research", "instruction": "..."}},
            {{"agent": "writing", "instruction": "..."}}]}}"""


def with_memory(goal: str, memory_context: str) -> str:
    """The user message for triage/planning, with recalled memory if there is any."""
    if not memory_context:
        return goal
    return f"Relevant context from memory:\n{memory_context}\n\nGoal: {goal}"
