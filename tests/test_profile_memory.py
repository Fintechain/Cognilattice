from __future__ import annotations

import json

from memory_server.curator import MemoryCurator
from memory_server.db import MemoryDB


class EmptyVectorStore:
    def insert(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def delete(self, *args, **kwargs) -> None:
        pass

    def search(self, *args, **kwargs) -> list:
        return []


def test_project_profile_shapes_candidate_layer_and_score(tmp_path):
    db = MemoryDB(str(tmp_path / "memory.db"))
    project = db.create_project(
        "profile-engineer",
        context_type="software",
        profile_id="software_engineer_v1",
    )
    curator = MemoryCurator(db, EmptyVectorStore())

    candidates = curator._extract_candidates(
        project["id"],
        (
            "The API interface contract failed because tests missed a regression; "
            "add a verification checklist for this failure mode."
        ),
        "conversation",
    )

    assert candidates
    candidate = candidates[0]
    assert candidate["profile_id"] == "software_engineer_v1"
    assert candidate["memory_layer"] in {"error", "procedural"}
    assert candidate["store_score"] >= candidate["base_score"]
    features = json.loads(candidate["profession_features"])
    assert features["professional_relevance"] > 0
    assert features["object_hits"]
    db.close()


def test_review_candidate_preserves_profile_metadata(tmp_path):
    db = MemoryDB(str(tmp_path / "memory.db"))
    project = db.create_project(
        "profile-review",
        context_type="software",
        profile_id="software_engineer_v1",
    )
    curator = MemoryCurator(db, EmptyVectorStore())
    candidate = curator._extract_candidates(
        project["id"],
        "A deployment failure showed the dependency version must be pinned and verified by tests.",
        "conversation",
    )[0]

    result = curator.review_candidate(candidate["id"], "accepted", refresh_summary=False)

    assert result["decision"] == "accepted"
    memory = result["memory"]
    assert memory["profile_id"] == "software_engineer_v1"
    assert memory["memory_layer"] in {"error", "procedural", "semantic", "episodic"}
    assert memory["store_score"] == candidate["store_score"]
    assert json.loads(memory["trait_features"])
    db.close()


def test_outcome_feedback_adjusts_memory_scores(tmp_path):
    db = MemoryDB(str(tmp_path / "memory.db"))
    project = db.create_project("feedback-test", profile_id="software_engineer_v1")
    memory = db.write_memory(
        project_id=project["id"],
        profile_id="software_engineer_v1",
        memory_layer="procedural",
        memory_type="coding_rule",
        title="Verify contracts",
        content="Always verify API contracts with regression tests.",
        importance=0.8,
        confidence=0.8,
        novelty=0.5,
        reusability=0.9,
        actionability=0.9,
        store_score=0.7,
    )

    success = db.record_outcome(
        project_id=project["id"],
        profile_id="software_engineer_v1",
        outcome_type="success",
        memory_ids=[memory["id"]],
    )
    boosted = db.get_memory(memory["id"])
    assert success["outcome_type"] == "success"
    assert boosted["store_score"] > memory["store_score"]

    db.record_outcome(
        project_id=project["id"],
        profile_id="software_engineer_v1",
        outcome_type="harmful",
        memory_ids=[memory["id"]],
    )
    quarantined = db.get_memory(memory["id"])
    assert quarantined["promotion_state"] == "quarantined"
    assert quarantined["store_score"] < boosted["store_score"]
    db.close()


def test_compaction_promotes_profile_memories(tmp_path):
    db = MemoryDB(str(tmp_path / "memory.db"))
    project = db.create_project(
        "promotion-test",
        context_type="software",
        profile_id="software_engineer_v1",
    )
    curator = MemoryCurator(db, EmptyVectorStore())
    for index, content in enumerate([
        "The API interface needs contract tests to verify regressions.",
        "The interface failure mode should be captured in a verification checklist.",
        "A regression happened because the API contract was not validated.",
    ]):
        scored = curator._extract_candidates(project["id"], content, "case")[0]
        curator.review_candidate(scored["id"], "accepted", refresh_summary=False)

    result = curator.compact_project_memory("promotion-test")
    promoted = db.search_memories(project_id=project["id"], memory_type="practice_pattern")
    error_patterns = db.search_memories(project_id=project["id"], memory_type="error_pattern")

    assert result["promoted"]["semantic"] >= 1
    assert promoted
    assert error_patterns
    assert all(memory["promotion_state"] == "consolidated" for memory in promoted)
    db.close()


def test_server_feedback_generates_error_memory(tmp_path, monkeypatch):
    import memory_server.server as server

    db = MemoryDB(str(tmp_path / "memory.db"))
    project = db.create_project("server-feedback", profile_id="software_engineer_v1")
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server.curator, "db", db)

    result = server.record_memory_feedback(
        project_name="server-feedback",
        outcome_type="harmful",
        task="fix API regression",
        feedback="The remembered fix caused a deployment failure.",
        reflection="Store this as an error pattern and check deployment constraints next time.",
    )

    assert result["status"] == "recorded"
    assert result["generated_memory"]["memory_layer"] == "error"
    assert result["generated_memory"]["memory_type"] == "error_pattern"
    assert db.get_stats(project["id"])["by_layer"]["error"] == 1
    db.close()
