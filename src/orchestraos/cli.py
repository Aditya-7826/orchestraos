"""`orchestraos run "<goal>"`: run the supervisor from the terminal (handy before the UI exists).

    uv run orchestraos run "Explain what a vector database is"
    uv run orchestraos run --thread ab12cd34     # resume a run that stopped part-way

One thread = one run: a thread's checkpoints hold that run's plan and results,
so a new goal always gets a new thread.
"""

import argparse
import logging
import sys
import uuid

from orchestraos.config import get_settings
from orchestraos.graph.factory import build_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orchestraos")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a goal through the supervisor")
    run.add_argument("goal", nargs="?", help="omit with --thread to resume that run")
    run.add_argument("--thread", help="resume this thread id instead of starting a new run")
    run.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if bool(args.goal) == bool(args.thread):
        parser.error("give either a goal (new run) or --thread (resume), not both")

    thread_id = args.thread or uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
    app = build_app(get_settings())

    if args.thread:
        snapshot = app.get_state(config)
        if not snapshot.values:
            parser.error(f"no run found for thread {thread_id}")
        if not snapshot.next:
            parser.error(f"thread {thread_id} already finished")

    # A goal starts a new run; None means "continue from the last checkpoint".
    graph_input = {"goal": args.goal} if args.goal else None
    for update in app.stream(graph_input, config, stream_mode="updates"):
        for node in update:
            print(f"  - {node}", file=sys.stderr)

    state = app.get_state(config).values
    print(f"\n{state.get('final_answer', '(no answer)')}\n")
    if sources := state.get("sources"):
        print("Sources:\n" + "\n".join(f"  - {s}" for s in sources))
    print(f"\n[thread {thread_id}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
