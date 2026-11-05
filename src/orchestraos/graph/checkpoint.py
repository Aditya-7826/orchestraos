"""SQLite checkpointer: saves supervisor state after every node so runs survive restarts."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def open_checkpointer(db_path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Open (and create if needed) the checkpoint database; closes the connection on exit."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as conn:
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        yield saver
