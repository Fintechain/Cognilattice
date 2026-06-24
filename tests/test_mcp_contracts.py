from __future__ import annotations

import asyncio
import json

from memory_server.backends.lancedb_backend import LanceDBStore
from memory_server.curator import MemoryCurator
from memory_server.db import MemoryDB
from memory_server.steelprint import MemorySteelprint


class DeleteFailingVectorStore(LanceDBStore):
    def delete(self, memory_id: str) -> None:
        raise RuntimeError("delete unavailable")


def _bind_server(monkeypatch, tmp_path, vector_class=LanceDBStore):
    import memory_server.server as server

    database = MemoryDB(str(tmp_path / "memory.db"))
    vectors = vector_class(str(tmp_path / "vectors"))
    memory_curator = MemoryCurator(database, vectors)
    monkeypatch.setattr(server, "db", database)
    monkeypatch.setattr(server, "vector_store", vectors)
    monkeypatch.setattr(server, "curator", memory_curator)
    monkeypatch.setattr(server, "steelprint", MemorySteelprint(database))
    return server, database


def test_all_mcp_tools_have_unique_valid_schemas():
    import memory_server.server as server

    tools = asyncio.run(server.mcp.list_tools())
    names = [tool.name for tool in tools]

    assert len(tools) >= 62
    assert len(names) == len(set(names))
    for tool in tools:
        assert tool.description
        assert tool.inputSchema["type"] == "object"
        assert isinstance(tool.inputSchema.get("properties", {}), dict)


def test_structured_inputs_are_not_restricted_to_json_strings():
    import memory_server.server as server

    tools = {
        tool.name: tool.inputSchema
        for tool in asyncio.run(server.mcp.list_tools())
    }
    structured_fields = {
        "write_memory": ["tags", "source_files"],
        "write_memories_batch": ["memories"],
        "record_memory_feedback": ["memory_ids"],
        "record_decision": ["alternatives"],
        "record_task_snapshot": ["completed", "remaining"],
        "register_provenance_source": ["metadata"],
        "add_evidence_span": ["metadata"],
        "steelprint_memory": ["evidence_ids"],
        "verify_grounded_answer": ["claims"],
        "run_hallucination_evaluation": ["cases"],
        "analyze_project_code": ["paths"],
    }
    for tool_name, fields in structured_fields.items():
        for field in fields:
            schema = tools[tool_name]["properties"][field]
            assert schema.get("type") != "string", (tool_name, field, schema)


def test_core_tool_lifecycle_accepts_native_and_string_json(monkeypatch, tmp_path):
    server, database = _bind_server(monkeypatch, tmp_path)
    created = server.create_project("contract-project", context_type="SOFTWARE")
    assert created["status"] == "created"

    wing = server.create_wing("contract-project", "Backend")
    assert wing["status"] == "created"
    room = server.create_room("backend", "contract-project", "API")
    assert room["status"] == "created"

    memory = server.write_memory(
        "contract-project", "fact", "Runtime", "The runtime is Python.",
        tags=["runtime", "python"], source_files=("README.md",),
        refresh_summary=False,
    )
    assert memory["status"] == "written"

    batch = server.write_memories_batch("contract-project", [
        {
            "memory_type": "fact",
            "title": "Malformed score",
            "content": "Malformed numeric scores fall back safely.",
            "importance": "not-a-number",
            "confidence": "0.9",
        },
        {
            "memory_type": "decision",
            "content": "Native arrays are accepted by MCP-facing batch tools.",
        },
    ])
    assert batch["status"] == "written"
    assert batch["count"] == 2
    assert batch.get("warnings")

    assert server.search_memory(
        project_name="contract-project", tags=["python"], limit="bad"
    )["count"] >= 1
    assert server.list_memories("missing-project")["error"]
    assert server.list_wings("missing-project")["error"]
    assert server.list_pending_candidates("missing-project")["error"]
    assert server.get_graph_stats("missing-project")["error"]
    database.close()


def test_delete_is_durable_when_vector_cleanup_fails(monkeypatch, tmp_path):
    server, database = _bind_server(monkeypatch, tmp_path, DeleteFailingVectorStore)
    server.create_project("delete-contract")
    written = server.write_memory(
        "delete-contract", "fact", "Delete me", "Durable delete.",
        refresh_summary=False,
    )
    memory_id = written["memory"]["id"]

    result = server.delete_memory(memory_id)

    assert result["status"] == "deleted"
    assert result["warnings"]
    assert database.get_memory(memory_id) is None
    database.close()


def test_retrieval_hits_are_batched_in_one_commit(tmp_path, monkeypatch):
    database = MemoryDB(str(tmp_path / "memory.db"))
    project = database.create_project("hit-batch")
    memories = [
        database.write_memory(
            project["id"], "fact", f"Memory {index}", f"Content {index}"
        )
        for index in range(3)
    ]
    commits = 0
    connection = database.conn
    original_commit = connection.commit

    class CommitProxy:
        def __getattr__(self, name):
            return getattr(connection, name)

        def commit(self):
            nonlocal commits
            commits += 1
            return original_commit()

    monkeypatch.setattr(database._local, "connection", CommitProxy())
    database.record_memory_hits([memory["id"] for memory in memories])

    assert commits == 1
    assert all(database.get_memory(memory["id"])["hit_count"] == 1 for memory in memories)
    database.close()


def test_temporal_graph_works_without_optional_palace(monkeypatch, tmp_path):
    server, database = _bind_server(monkeypatch, tmp_path)

    def unavailable(*args, **kwargs):
        raise RuntimeError("optional palace unavailable")

    monkeypatch.setattr(server.curator.palace, "add_knowledge_triple", unavailable)
    monkeypatch.setattr(server.curator.palace, "query_knowledge", unavailable)
    monkeypatch.setattr(server.curator.palace, "invalidate_knowledge", unavailable)

    triple = server.add_knowledge_triple("Memery", "mode", "local")
    assert triple["subject"] == "Memery"
    queried = server.query_entity("Memery")
    assert queried["count"] == 1
    invalidated = server.invalidate_triple("Memery", "mode", "local")
    assert invalidated["invalidated"] is True
    assert server.query_entity("Memery")["count"] == 0
    assert server.query_entity_timeline("Memery")["count"] == 1
    database.close()


def test_malformed_structured_inputs_return_errors_not_exceptions(monkeypatch, tmp_path):
    server, database = _bind_server(monkeypatch, tmp_path)
    server.create_project("malformed")

    assert "error" in server.write_memories_batch("malformed", "{bad json")
    assert "error" in server.analyze_project_code("malformed", "{bad json")
    assert "error" in server.register_provenance_source("malformed", " ")
    assert "error" in server.add_evidence_span("missing", "text")
    assert "error" in server.steelprint_memory("missing", {"bad": "shape"})
    assert "error" in server.verify_grounded_answer(
        "malformed", "q", "a", claims={"bad": "shape"}
    )
    assert "error" in server.run_hallucination_evaluation(
        "malformed", "bad", cases={"bad": "shape"}
    )
    assert "error" in server.get_graph_neighbors("missing", "sideways")
    database.close()


def test_builtin_profiles_cannot_be_overwritten(monkeypatch, tmp_path):
    server, database = _bind_server(monkeypatch, tmp_path)

    result = server.create_memory_profile(
        "generalist_v1", traits={"rigor": 0.0}, name="Corrupted built-in"
    )

    assert "error" in result
    row = database.conn.execute(
        "SELECT source FROM memory_profiles WHERE profile_id='generalist_v1'"
    ).fetchone()
    assert row["source"] == "built-in"
    database.close()


def test_steelprint_tolerates_malformed_claim_members(monkeypatch, tmp_path):
    server, database = _bind_server(monkeypatch, tmp_path)
    server.create_project("steel-contract")
    source = server.register_provenance_source(
        "steel-contract", "file:///spec.txt", content="The service is local."
    )["source"]
    evidence = server.add_evidence_span(
        source["id"], "The service is local.", page_number=0
    )["evidence"]

    result = server.verify_grounded_answer(
        "steel-contract",
        "Where is it?",
        "The service is local. It is public.",
        claims=[
            {
                "sentence_index": "not-an-index",
                "text": "The service is local.",
                "evidence_ids": evidence["id"],
            },
            42,
        ],
    )

    assert result["status"] == "refused"
    assert result["verification"]["unsupported_claims"] == 1
    assert "error" in server.add_evidence_span(
        source["id"], "bad locator", line_start=5, line_end=2
    )
    database.close()
