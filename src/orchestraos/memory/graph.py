"""Graph memory: entities and relationships (user -> prefers -> X, project -> uses -> Y).

Neo4j Community runs in the optional `graph` Compose profile. Without it the
app uses `NullGraph` and keeps working with semantic memory only.

Relationship types can't be query parameters in Cypher, so every edge is a
`:REL` with its type stored as a property; that keeps the queries parameterised
(no string-built Cypher from model output).
"""

import logging
from typing import Protocol

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class Fact(BaseModel):
    subject: str
    relation: str
    object: str

    @field_validator("subject", "relation", "object")
    @classmethod
    def normalize(cls, value: str) -> str:
        return " ".join(value.lower().split())[:100]

    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"


class GraphStore(Protocol):
    async def add_facts(self, facts: list[Fact]) -> None: ...
    async def related(self, text: str, limit: int = 10) -> list[Fact]: ...
    async def close(self) -> None: ...


class NullGraph:
    async def add_facts(self, facts: list[Fact]) -> None:
        return None

    async def related(self, text: str, limit: int = 10) -> list[Fact]:
        return []

    async def close(self) -> None:
        return None


ADD_FACT = """
MERGE (s:Entity {name: $subject})
MERGE (o:Entity {name: $object})
MERGE (s)-[r:REL {type: $relation}]->(o)
SET r.updated = timestamp()
"""

# Facts about the user, plus facts about any entity named in the text.
RELATED = """
MATCH (s:Entity)-[r:REL]->(o:Entity)
WHERE s.name = 'user' OR $text CONTAINS s.name OR $text CONTAINS o.name
RETURN s.name AS subject, r.type AS relation, o.name AS object
ORDER BY r.updated DESC
LIMIT $limit
"""


class Neo4jGraph:
    def __init__(self, driver):
        self._driver = driver

    @classmethod
    async def connect(cls, uri: str, user: str, password: str) -> "Neo4jGraph":
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        await driver.verify_connectivity()
        return cls(driver)

    async def add_facts(self, facts: list[Fact]) -> None:
        for fact in facts:
            await self._driver.execute_query(ADD_FACT, **fact.model_dump())

    async def related(self, text: str, limit: int = 10) -> list[Fact]:
        records, _, _ = await self._driver.execute_query(RELATED, text=text.lower(), limit=limit)
        return [Fact(**record.data()) for record in records]

    async def close(self) -> None:
        await self._driver.close()
