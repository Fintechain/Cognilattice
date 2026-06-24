# -*- coding: utf-8 -*-
"""Memory Steelprint: provenance, conflict detection, grounding, and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .db import MemoryDB, _now, _uid


_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_SENTENCE_RE = re.compile(r"[^。！？.!?\n]+[。！？.!?]?")
_NEGATIONS = {
    "不", "不是", "不能", "没有", "并非", "禁止", "never", "not", "no",
    "cannot", "can't", "mustn't", "without", "disabled", "false",
}


def _hash(*parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    tokens = {
        token.casefold() for token in _WORD_RE.findall(text or "") if len(token) > 1
    }
    for run in _CJK_RE.findall(text or ""):
        tokens.update(run[index:index + 2] for index in range(max(0, len(run) - 1)))
        if len(run) == 1:
            tokens.add(run)
    return tokens


def _normal(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _polarity(text: str) -> int:
    lowered = _normal(text)
    english = set(_WORD_RE.findall(lowered))
    chinese_negations = {"不", "不是", "不能", "没有", "并非", "禁止", "禁用"}
    english_negations = _NEGATIONS - chinese_negations
    return -1 if (
        any(term in lowered for term in chinese_negations)
        or bool(english & english_negations)
    ) else 1


def _overlap(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _finite_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number or number in {float("inf"), float("-inf")}:
        return default
    return number


def _safe_json(value: Any, default: Any) -> Any:
    if isinstance(value, type(default)):
        return value
    try:
        parsed = json.loads(value or "")
    except (json.JSONDecodeError, TypeError):
        return default
    return parsed if isinstance(parsed, type(default)) else default


class MemorySteelprint:
    """Attach evidence to memory and enforce grounded answer policies."""

    def __init__(self, db: MemoryDB) -> None:
        self.db = db

    def register_source(
        self,
        project_id: str,
        uri: str,
        title: str = "",
        source_type: str = "document",
        version: str = "",
        content: str = "",
        content_hash: str = "",
        trust_score: float = 0.8,
        metadata: dict | None = None,
    ) -> dict:
        digest = content_hash or _hash(content)
        existing = self.db.conn.execute(
            """SELECT * FROM provenance_sources
               WHERE project_id=? AND uri=? AND version=? AND content_hash=?""",
            (project_id, uri, version, digest),
        ).fetchone()
        if existing:
            return dict(existing)
        source_id = _uid()
        self.db.conn.execute(
            """INSERT INTO provenance_sources
               (id, project_id, source_type, uri, title, version, content_hash,
                trust_score, metadata, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                source_id, project_id, source_type, uri, title, version, digest,
                max(0.0, min(1.0, _finite_float(trust_score, 0.8))),
                json.dumps(metadata or {}, ensure_ascii=False), _now(),
            ),
        )
        self.db.conn.commit()
        return self.get_source(source_id)

    def get_source(self, source_id: str) -> dict:
        row = self.db.conn.execute(
            "SELECT * FROM provenance_sources WHERE id=?", (source_id,)
        ).fetchone()
        return dict(row) if row else {}

    def add_evidence(
        self,
        source_id: str,
        quoted_text: str,
        page_number: int | None = None,
        paragraph_number: int | None = None,
        section_title: str = "",
        line_start: int | None = None,
        line_end: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        locator: str = "",
        metadata: dict | None = None,
    ) -> dict:
        if not self.get_source(source_id):
            raise ValueError(f"Unknown provenance source '{source_id}'.")
        text = (quoted_text or "").strip()
        if not text:
            raise ValueError("quoted_text cannot be empty.")
        for field_name, value in (
            ("page_number", page_number),
            ("paragraph_number", paragraph_number),
            ("line_start", line_start),
            ("line_end", line_end),
            ("char_start", char_start),
            ("char_end", char_end),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative.")
        if line_start is not None and line_end is not None and line_start > line_end:
            raise ValueError("line_start cannot be greater than line_end.")
        if char_start is not None and char_end is not None and char_start > char_end:
            raise ValueError("char_start cannot be greater than char_end.")
        evidence_id = _uid()
        digest = _hash(text)
        self.db.conn.execute(
            """INSERT INTO evidence_spans
               (id, source_id, page_number, paragraph_number, section_title,
                line_start, line_end, char_start, char_end, quoted_text,
                content_hash, locator, metadata, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                evidence_id, source_id, page_number, paragraph_number, section_title,
                line_start, line_end, char_start, char_end, text, digest, locator,
                json.dumps(metadata or {}, ensure_ascii=False), _now(),
            ),
        )
        self.db.conn.commit()
        return self.get_evidence(evidence_id)

    def get_evidence(self, evidence_id: str) -> dict:
        row = self.db.conn.execute(
            """SELECT e.*, s.uri, s.title AS source_title, s.version,
                      s.trust_score, s.content_hash AS source_content_hash
               FROM evidence_spans e
               JOIN provenance_sources s ON s.id=e.source_id
               WHERE e.id=?""",
            (evidence_id,),
        ).fetchone()
        return dict(row) if row else {}

    def stamp_memory(
        self,
        memory_id: str,
        evidence_ids: list[str],
        claim_subject: str = "",
        claim_predicate: str = "",
        claim_object: str = "",
        support_type: str = "supports",
        entailment_score: float = 1.0,
    ) -> dict:
        memory = self.db.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Unknown memory '{memory_id}'.")
        if not evidence_ids:
            raise ValueError("At least one evidence_id is required.")
        stamps = []
        for evidence_id in evidence_ids:
            evidence = self.get_evidence(evidence_id)
            if not evidence:
                raise ValueError(f"Unknown evidence '{evidence_id}'.")
            stamp_hash = _hash(
                memory_id, memory["content"], evidence_id, evidence["content_hash"],
                claim_subject, claim_predicate, claim_object, support_type,
            )
            stamp_id = _uid()
            self.db.conn.execute(
                """INSERT OR IGNORE INTO memory_steelprints
                   (id, memory_id, evidence_id, claim_subject, claim_predicate,
                    claim_object, support_type, entailment_score, steelprint_hash,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    stamp_id, memory_id, evidence_id, claim_subject, claim_predicate,
                    claim_object, support_type,
                    max(0.0, min(1.0, _finite_float(entailment_score, 1.0))),
                    stamp_hash, _now(),
                ),
            )
        self.db.conn.commit()
        stamps = self.get_memory_steelprint(memory_id)["evidence"]
        conflicts = self.detect_conflicts(memory["project_id"], memory_id=memory_id)
        return {
            "memory_id": memory_id,
            "steelprinted": True,
            "evidence": stamps,
            "conflicts": conflicts,
        }

    def get_memory_steelprint(self, memory_id: str) -> dict:
        rows = self.db.conn.execute(
            """SELECT ms.*, e.quoted_text, e.page_number, e.paragraph_number,
                      e.section_title, e.line_start, e.line_end, e.locator,
                      s.uri, s.title AS source_title, s.version, s.trust_score
               FROM memory_steelprints ms
               JOIN evidence_spans e ON e.id=ms.evidence_id
               JOIN provenance_sources s ON s.id=e.source_id
               WHERE ms.memory_id=?
               ORDER BY ms.created_at""",
            (memory_id,),
        ).fetchall()
        return {"memory_id": memory_id, "evidence": [dict(row) for row in rows]}

    @staticmethod
    def _objects_conflict(left: str, right: str) -> bool:
        a = _normal(left)
        b = _normal(right)
        if not a or not b or a == b:
            return False
        if _polarity(a) != _polarity(b):
            return True
        exclusive = {
            ("enabled", "disabled"), ("true", "false"), ("allow", "deny"),
            ("允许", "禁止"), ("启用", "禁用"), ("是", "不是"),
        }
        return any(x in a and y in b or y in a and x in b for x, y in exclusive)

    def detect_conflicts(self, project_id: str, memory_id: str | None = None) -> list[dict]:
        params: list[Any] = [project_id]
        scope = ""
        if memory_id:
            scope = "AND (a.memory_id=? OR b.memory_id=?)"
            params.extend([memory_id, memory_id])
        rows = self.db.conn.execute(
            f"""SELECT a.memory_id AS memory_id_a, b.memory_id AS memory_id_b,
                       a.claim_subject, a.claim_predicate,
                       a.claim_object AS object_a, b.claim_object AS object_b
                FROM memory_steelprints a
                JOIN memories ma ON ma.id=a.memory_id
                JOIN memory_steelprints b
                  ON a.claim_subject=b.claim_subject
                 AND a.claim_predicate=b.claim_predicate
                 AND a.memory_id < b.memory_id
                JOIN memories mb ON mb.id=b.memory_id
                WHERE ma.project_id=? AND mb.project_id=ma.project_id
                  AND ma.status='active' AND mb.status='active'
                  AND a.claim_subject<>'' AND a.claim_predicate<>'' {scope}""",
            params,
        ).fetchall()
        detected = []
        for row in rows:
            item = dict(row)
            if not self._objects_conflict(item["object_a"], item["object_b"]):
                continue
            explanation = (
                f"{item['claim_subject']} / {item['claim_predicate']} has incompatible "
                f"values: {item['object_a']} <> {item['object_b']}."
            )
            conflict_id = _uid()
            self.db.conn.execute(
                """INSERT OR IGNORE INTO memory_conflicts
                   (id, project_id, memory_id_a, memory_id_b, claim_subject,
                    claim_predicate, severity, explanation, detected_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    conflict_id, project_id, item["memory_id_a"], item["memory_id_b"],
                    item["claim_subject"], item["claim_predicate"], 0.9,
                    explanation, _now(),
                ),
            )
            detected.append({**item, "severity": 0.9, "explanation": explanation})
        self.db.conn.commit()
        return detected

    def list_conflicts(self, project_id: str, status: str = "open") -> list[dict]:
        rows = self.db.conn.execute(
            """SELECT * FROM memory_conflicts
               WHERE project_id=? AND status=?
               ORDER BY severity DESC, detected_at DESC""",
            (project_id, status),
        ).fetchall()
        return [dict(row) for row in rows]

    def resolve_conflict(self, conflict_id: str, status: str = "resolved") -> dict:
        if status not in {"resolved", "dismissed", "open"}:
            raise ValueError("status must be open, resolved, or dismissed.")
        resolved_at = None if status == "open" else _now()
        self.db.conn.execute(
            "UPDATE memory_conflicts SET status=?, resolved_at=? WHERE id=?",
            (status, resolved_at, conflict_id),
        )
        self.db.conn.commit()
        row = self.db.conn.execute(
            "SELECT * FROM memory_conflicts WHERE id=?", (conflict_id,)
        ).fetchone()
        return dict(row) if row else {}

    def _verify_claim(self, text: str, evidence_ids: list[str]) -> dict:
        evidence = [self.get_evidence(eid) for eid in evidence_ids]
        evidence = [item for item in evidence if item]
        if not evidence:
            return {
                "status": "unsupported", "support_score": 0.0,
                "contradiction_score": 0.0, "reason": "No valid evidence citation.",
            }
        scores = [
            _overlap(text, item["quoted_text"]) * float(item.get("trust_score") or 0.0)
            for item in evidence
        ]
        support = max(scores, default=0.0)
        polarity_conflict = any(
            _polarity(text) != _polarity(item["quoted_text"]) and score >= 0.25
            for item, score in zip(evidence, scores)
        )
        contradiction = max(scores, default=0.0) if polarity_conflict else 0.0
        if contradiction >= 0.25:
            status = "contradicted"
            reason = "Cited evidence has opposing polarity."
        elif support >= 0.35:
            status = "supported"
            reason = "Claim is lexically grounded in cited evidence."
        else:
            status = "weak"
            reason = "Citation exists, but support overlap is weak."
        return {
            "status": status,
            "support_score": round(support, 4),
            "contradiction_score": round(contradiction, 4),
            "reason": reason,
            "evidence": evidence,
        }

    def _evidence_conflict_score(self, evidence_ids: list[str]) -> float:
        if not evidence_ids:
            return 0.0
        placeholders = ",".join("?" for _ in evidence_ids)
        row = self.db.conn.execute(
            f"""SELECT MAX(c.severity) AS severity
                FROM memory_steelprints ms
                JOIN memory_conflicts c
                  ON (c.memory_id_a=ms.memory_id OR c.memory_id_b=ms.memory_id)
                WHERE ms.evidence_id IN ({placeholders}) AND c.status='open'""",
            evidence_ids,
        ).fetchone()
        return float(row["severity"] or 0.0) if row else 0.0

    def verify_answer(
        self,
        project_id: str,
        question: str,
        answer: str,
        claims: list[dict] | None = None,
        policy: str = "strict",
        min_coverage: float = 1.0,
        min_confidence: float = 0.65,
    ) -> dict:
        if policy not in {"strict", "balanced", "permissive"}:
            raise ValueError("policy must be strict, balanced, or permissive.")
        answer_sentences = [
            match.group(0).strip()
            for match in _SENTENCE_RE.finditer(answer or "")
            if match.group(0).strip()
        ]
        if claims is None:
            claims = [
                {"text": sentence, "evidence_ids": [], "factual": True}
                for sentence in answer_sentences
            ]
        else:
            normalized_claims = []
            for claim in claims:
                if isinstance(claim, dict):
                    normalized_claims.append(dict(claim))
                elif isinstance(claim, str) and claim.strip():
                    normalized_claims.append({
                        "text": claim.strip(), "evidence_ids": [], "factual": True,
                    })
            claims = normalized_claims
            covered_indexes = {
                int(claim["sentence_index"])
                for claim in claims
                if str(claim.get("sentence_index", "")).lstrip("-").isdigit()
                and 0 <= int(claim["sentence_index"]) < len(answer_sentences)
            }
            covered_text = {_normal(str(claim.get("text", ""))) for claim in claims}
            for sentence_index, sentence in enumerate(answer_sentences):
                if sentence_index not in covered_indexes and _normal(sentence) not in covered_text:
                    claims.append({
                        "text": sentence,
                        "sentence_index": sentence_index,
                        "evidence_ids": [],
                        "factual": True,
                    })
        answer_id = _uid()
        self.db.conn.execute(
            """INSERT INTO grounded_answers
               (id, project_id, question, answer, policy, status, confidence,
                citation_coverage, verification, created_at)
               VALUES (?,?,?,?,?,'verifying',0.0,0.0,'{}',?)""",
            (answer_id, project_id, question, answer, policy, _now()),
        )
        verified_claims = []
        factual_count = 0
        cited_count = 0
        support_scores = []
        contradiction_scores = []
        for index, claim in enumerate(claims):
            text = str(claim.get("text", "")).strip()
            if not text:
                continue
            factual = bool(claim.get("factual", True))
            raw_evidence_ids = claim.get("evidence_ids", [])
            if isinstance(raw_evidence_ids, str):
                raw_evidence_ids = [raw_evidence_ids]
            if not isinstance(raw_evidence_ids, (list, tuple, set)):
                raw_evidence_ids = []
            evidence_ids = [str(item) for item in raw_evidence_ids if str(item).strip()]
            verification = (
                self._verify_claim(text, evidence_ids)
                if factual else {
                    "status": "not_factual", "support_score": 1.0,
                    "contradiction_score": 0.0, "reason": "Marked non-factual.",
                    "evidence": [],
                }
            )
            evidence_conflict = self._evidence_conflict_score(evidence_ids)
            if factual and evidence_conflict > 0:
                verification["status"] = "conflicted"
                verification["contradiction_score"] = max(
                    verification["contradiction_score"], evidence_conflict
                )
                verification["reason"] = "Cited evidence is attached to an open memory conflict."
            if factual:
                factual_count += 1
                cited_count += int(bool(evidence_ids))
                support_scores.append(verification["support_score"])
                contradiction_scores.append(verification["contradiction_score"])
            try:
                sentence_index = int(claim.get("sentence_index", index))
            except (TypeError, ValueError, OverflowError):
                sentence_index = index
            claim_id = _uid()
            self.db.conn.execute(
                """INSERT INTO answer_claims
                   (id, answer_id, sentence_index, claim_text, factual,
                    support_status, support_score, contradiction_score,
                    verification_reason, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    claim_id, answer_id, sentence_index, text, int(factual),
                    verification["status"], verification["support_score"],
                    verification["contradiction_score"], verification["reason"], _now(),
                ),
            )
            for evidence_id in evidence_ids:
                if self.get_evidence(evidence_id):
                    self.db.conn.execute(
                        """INSERT OR IGNORE INTO answer_citations
                           (id, claim_id, evidence_id, citation_label, created_at)
                           VALUES (?,?,?,?,?)""",
                        (_uid(), claim_id, evidence_id, f"S{sentence_index + 1}", _now()),
                    )
            verified_claims.append({
                "claim_id": claim_id,
                "sentence_index": sentence_index,
                "text": text,
                "factual": factual,
                "evidence_ids": evidence_ids,
                **{key: value for key, value in verification.items() if key != "evidence"},
            })
        coverage = cited_count / factual_count if factual_count else 1.0
        mean_support = sum(support_scores) / len(support_scores) if support_scores else 1.0
        max_contradiction = max(contradiction_scores, default=0.0)
        confidence = max(0.0, min(1.0, coverage * 0.45 + mean_support * 0.55 - max_contradiction * 0.5))
        unsupported = sum(
            1 for claim in verified_claims
            if claim["factual"] and claim["status"] in {
                "unsupported", "weak", "contradicted", "conflicted",
            }
        )
        strict_failure = coverage < min_coverage or unsupported > 0 or confidence < min_confidence
        status = "refused" if policy == "strict" and strict_failure else (
            "low_confidence" if strict_failure else "grounded"
        )
        verification_summary = {
            "factual_claims": factual_count,
            "cited_claims": cited_count,
            "unsupported_claims": unsupported,
            "contradiction_score": round(max_contradiction, 4),
            "mean_support": round(mean_support, 4),
        }
        self.db.conn.execute(
            """UPDATE grounded_answers
               SET status=?, confidence=?, citation_coverage=?, verification=?
               WHERE id=?""",
            (
                status, round(confidence, 4), round(coverage, 4),
                json.dumps(verification_summary, ensure_ascii=False), answer_id,
            ),
        )
        self.db.conn.commit()
        return {
            "answer_id": answer_id,
            "status": status,
            "allowed": status != "refused",
            "confidence": round(confidence, 4),
            "citation_coverage": round(coverage, 4),
            "claims": verified_claims,
            "verification": verification_summary,
            "refusal_reason": (
                "Insufficient or contradictory evidence under strict steelprint policy."
                if status == "refused" else None
            ),
        }

    def evaluate(
        self,
        project_id: str,
        name: str,
        cases: list[dict],
        policy: str = "strict",
    ) -> dict:
        results = []
        for case in cases:
            if not isinstance(case, dict):
                results.append({
                    "case_id": None,
                    "expected_supported": False,
                    "correct": False,
                    "status": "invalid_case",
                    "confidence": 0.0,
                    "citation_coverage": 0.0,
                    "unsupported_claims": 1,
                    "contradiction_score": 0.0,
                })
                continue
            result = self.verify_answer(
                project_id=project_id,
                question=str(case.get("question", "")),
                answer=str(case.get("answer", "")),
                claims=case.get("claims"),
                policy=policy,
            )
            expected_supported = bool(case.get("expected_supported", True))
            correct = result["allowed"] == expected_supported
            results.append({
                "case_id": case.get("id"),
                "expected_supported": expected_supported,
                "correct": correct,
                "status": result["status"],
                "confidence": result["confidence"],
                "citation_coverage": result["citation_coverage"],
                "unsupported_claims": result["verification"]["unsupported_claims"],
                "contradiction_score": result["verification"]["contradiction_score"],
            })
        total = len(results)
        metrics = {
            "cases": total,
            "decision_accuracy": round(
                sum(item["correct"] for item in results) / total, 4
            ) if total else 0.0,
            "citation_coverage": round(
                sum(item["citation_coverage"] for item in results) / total, 4
            ) if total else 0.0,
            "unsupported_claim_rate": round(
                sum(item["unsupported_claims"] > 0 for item in results) / total, 4
            ) if total else 0.0,
            "grounded_answer_rate": round(
                sum(item["status"] == "grounded" for item in results) / total, 4
            ) if total else 0.0,
            "contradiction_rate": round(
                sum(item["contradiction_score"] > 0 for item in results) / total, 4
            ) if total else 0.0,
            "refusal_rate": round(
                sum(item["status"] == "refused" for item in results) / total, 4
            ) if total else 0.0,
            "mean_confidence": round(
                sum(item["confidence"] for item in results) / total, 4
            ) if total else 0.0,
        }
        evaluation_id = _uid()
        self.db.conn.execute(
            """INSERT INTO hallucination_evaluations
               (id, project_id, name, metrics, cases, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                evaluation_id, project_id, name,
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(results, ensure_ascii=False), _now(),
            ),
        )
        self.db.conn.commit()
        return {
            "evaluation_id": evaluation_id,
            "name": name,
            "metrics": metrics,
            "results": results,
        }

    def list_evaluations(self, project_id: str, limit: int = 20) -> list[dict]:
        rows = self.db.conn.execute(
            """SELECT * FROM hallucination_evaluations
               WHERE project_id=? ORDER BY created_at DESC LIMIT ?""",
            (project_id, max(1, min(int(limit), 200))),
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["metrics"] = _safe_json(item.get("metrics"), {})
            item["cases"] = _safe_json(item.get("cases"), [])
            results.append(item)
        return results
