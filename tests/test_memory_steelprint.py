from __future__ import annotations

from memory_server.db import MemoryDB
from memory_server.steelprint import MemorySteelprint


def _setup(tmp_path):
    db = MemoryDB(str(tmp_path / "memory.db"))
    project = db.create_project("steelprint-test")
    service = MemorySteelprint(db)
    source = service.register_source(
        project_id=project["id"],
        uri="file:///docs/api-spec.pdf",
        title="API specification",
        version="1.0",
        content="The API requires authentication for every request.",
    )
    evidence = service.add_evidence(
        source_id=source["id"],
        quoted_text="The API requires authentication for every request.",
        page_number=7,
        paragraph_number=3,
        section_title="Authentication",
    )
    return db, project, service, source, evidence


def test_provenance_chain_preserves_exact_locator_and_hash(tmp_path):
    db, project, service, source, evidence = _setup(tmp_path)
    memory = db.write_memory(
        project_id=project["id"],
        memory_type="api_contract",
        title="Authentication required",
        content="Every API request requires authentication.",
    )

    stamped = service.stamp_memory(
        memory["id"],
        [evidence["id"]],
        claim_subject="API request",
        claim_predicate="requires",
        claim_object="authentication",
    )

    assert stamped["steelprinted"] is True
    assert stamped["evidence"][0]["page_number"] == 7
    assert stamped["evidence"][0]["paragraph_number"] == 3
    assert stamped["evidence"][0]["uri"] == "file:///docs/api-spec.pdf"
    assert len(stamped["evidence"][0]["steelprint_hash"]) == 64
    db.close()


def test_conflicting_claims_are_detected(tmp_path):
    db, project, service, source, evidence = _setup(tmp_path)
    allow = db.write_memory(
        project_id=project["id"], memory_type="fact", title="Feature",
        content="Remote access is enabled.",
    )
    deny = db.write_memory(
        project_id=project["id"], memory_type="fact", title="Feature override",
        content="Remote access is disabled.",
    )
    service.stamp_memory(
        allow["id"], [evidence["id"]],
        claim_subject="remote access", claim_predicate="state",
        claim_object="enabled",
    )
    result = service.stamp_memory(
        deny["id"], [evidence["id"]],
        claim_subject="remote access", claim_predicate="state",
        claim_object="disabled",
    )

    assert result["conflicts"]
    assert service.list_conflicts(project["id"])[0]["severity"] == 0.9
    db.close()


def test_strict_answer_policy_requires_supported_citation_per_claim(tmp_path):
    db, project, service, source, evidence = _setup(tmp_path)
    supported = service.verify_answer(
        project_id=project["id"],
        question="Does the API require authentication?",
        answer="The API requires authentication for every request.",
        claims=[{
            "text": "The API requires authentication for every request.",
            "factual": True,
            "evidence_ids": [evidence["id"]],
        }],
        policy="strict",
    )
    unsupported = service.verify_answer(
        project_id=project["id"],
        question="Does the API require authentication?",
        answer="The API allows anonymous access.",
        claims=[{
            "text": "The API allows anonymous access.",
            "factual": True,
            "evidence_ids": [],
        }],
        policy="strict",
    )

    assert supported["status"] == "grounded"
    assert supported["citation_coverage"] == 1.0
    assert unsupported["status"] == "refused"
    assert unsupported["allowed"] is False
    db.close()


def test_strict_policy_catches_answer_sentence_omitted_from_claim_map(tmp_path):
    db, project, service, source, evidence = _setup(tmp_path)
    result = service.verify_answer(
        project_id=project["id"],
        question="What does the API require?",
        answer=(
            "The API requires authentication for every request. "
            "Anonymous access is enabled."
        ),
        claims=[{
            "sentence_index": 0,
            "text": "The API requires authentication for every request.",
            "evidence_ids": [evidence["id"]],
        }],
        policy="strict",
    )

    assert result["status"] == "refused"
    assert result["citation_coverage"] == 0.5
    assert result["verification"]["unsupported_claims"] == 1
    db.close()


def test_chinese_claim_support_uses_cjk_bigrams(tmp_path):
    db = MemoryDB(str(tmp_path / "memory.db"))
    project = db.create_project("steelprint-zh")
    service = MemorySteelprint(db)
    source = service.register_source(
        project["id"], "file:///产品规范.pdf", content="所有接口请求必须经过身份验证。"
    )
    evidence = service.add_evidence(
        source["id"], "所有接口请求必须经过身份验证。", page_number=2
    )

    result = service.verify_answer(
        project["id"], "接口是否需要验证？", "接口请求必须经过身份验证。",
        claims=[{
            "text": "接口请求必须经过身份验证。",
            "evidence_ids": [evidence["id"]],
        }],
    )

    assert result["status"] == "grounded"
    db.close()


def test_hallucination_evaluation_reports_quantitative_metrics(tmp_path):
    db, project, service, source, evidence = _setup(tmp_path)
    result = service.evaluate(
        project["id"],
        "steelprint-baseline",
        [
            {
                "id": "supported",
                "question": "Authentication?",
                "answer": "The API requires authentication for every request.",
                "claims": [{
                    "text": "The API requires authentication for every request.",
                    "evidence_ids": [evidence["id"]],
                }],
                "expected_supported": True,
            },
            {
                "id": "unsupported",
                "question": "Anonymous access?",
                "answer": "Anonymous access is enabled.",
                "claims": [{
                    "text": "Anonymous access is enabled.",
                    "evidence_ids": [],
                }],
                "expected_supported": False,
            },
        ],
    )

    assert result["metrics"]["cases"] == 2
    assert result["metrics"]["decision_accuracy"] == 1.0
    assert result["metrics"]["refusal_rate"] == 0.5
    assert service.list_evaluations(project["id"])[0]["name"] == "steelprint-baseline"
    db.close()
