# -*- coding: utf-8 -*-
r"""MCP Memory Server v2 — Enhanced with Graphify + MemPalace.

Inspired by:
  - graphify: `safishamsi/graphify` — code knowledge graph pipeline
  - mempalace: `MemPalace/mempalace` — spatial memory system

Tools (29+):
  ── Palace Management ──
    create_project, list_projects
    create_wing, list_wings, create_room, list_rooms
    list_halls

  ── Memory Write/Read ──
    write_memory, search_memory, recall_for_task
    get_context_bundle, delete_memory, deprecate_memory
    list_memories, get_memory

  ── Memory Curator Pipeline ──
    ingest_conversation, extract_memory_candidates
    review_memory_candidate, list_pending_candidates
    compact_project_memory, prune_low_value_memories

  ── Graph Pipeline (graphify) ──
    analyze_project_code, query_graph_path, get_graph_neighbors
    get_graph_stats

  ── Knowledge Graph (mempalace) ──
    add_knowledge_triple, query_entity, query_entity_timeline
    invalidate_triple, get_timeline

  ── Task & Decisions ──
    record_decision, list_decisions
    record_task_snapshot, list_task_snapshots
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import get_config, Confidence
from .db import MemoryDB, _uid, _now
from .curator import MemoryCurator, PROTECTED_TYPES, TOP_LEVEL_MEMORY_TYPES
from .steelprint import MemorySteelprint
from .service_lock import ServiceLockError, acquire_service_lock, exit_for_lock_error
from .profiles import compose_profile_from_presets, score_memory_with_profile, score_retrieval_with_profile
from .stdio import run_fastmcp_stdio

logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

# Lazy imports for heavy deps
_backends = None


def _get_backends():
    global _backends
    if _backends is None:
        from .backends.lancedb_backend import LanceDBStore
        _backends = LanceDBStore
    return _backends


# ── Initialize ──────────────────────────────────────────────────────────

mcp = FastMCP("Memory Curator Server v2 (Graphify+MemPalace)")

cfg = get_config()
db = MemoryDB()
vector_store = _get_backends()()
curator = MemoryCurator(db, vector_store)
steelprint = MemorySteelprint(db)
_runtime = {
    "mode": "imported",
    "endpoint": None,
    "pid": os.getpid(),
}


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Lightweight health and connection-discovery endpoint."""
    return JSONResponse({
        "status": "ok",
        "service": "memery",
        **_runtime,
    })


def _ensure_project(project_name: str) -> dict | None:
    project = db.get_project_by_name(project_name)
    if not project:
        return {"error": f"Project '{project_name}' not found."}
    return project


def _parse_string_list(value, field_name: str) -> tuple[list[str], str | None]:
    """Accept JSON arrays, comma-separated strings, or a single plain value."""
    if value is None or value == "":
        return [], None
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()], None
    if not isinstance(value, str):
        return [str(value)], f"{field_name} was coerced to a string list."
    raw = value.strip()
    if not raw:
        return [], None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if "," in raw:
            return [part.strip() for part in raw.split(",") if part.strip()], None
        return [raw], f"{field_name} was not valid JSON; treated as one value."
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()], None
    if isinstance(parsed, str):
        return [parsed], f"{field_name} JSON value was a string; treated as one value."
    return [str(parsed)], f"{field_name} JSON value was not an array; coerced."


def _parse_json_object_field(value: Any, field_name: str, default: dict | None = None) -> tuple[dict, str | None]:
    """Accept a JSON object or an already-decoded dict with field-specific errors."""
    if value is None or value == "":
        return dict(default or {}), None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return {}, f"Invalid JSON in field '{field_name}': {exc}"
    if not isinstance(value, dict):
        return {}, (
            f"Invalid JSON in field '{field_name}': expected an object, "
            f"got {type(value).__name__}."
        )
    return value, None


def _parse_json_array_field(value: Any, field_name: str) -> tuple[list, str | None]:
    """Accept a JSON array or an already-decoded list."""
    if value is None or value == "":
        return [], None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return [], f"Invalid JSON in field '{field_name}': {exc}"
    if not isinstance(value, list):
        return [], (
            f"Invalid JSON in field '{field_name}': expected an array, "
            f"got {type(value).__name__}."
        )
    return value, None


def _coerce_float(value: Any, default: float = 0.0) -> tuple[float, str | None]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default, f"Invalid numeric value for '{value}'; using {default}."
    if number != number or number in {float("inf"), float("-inf")}:
        return default, f"Non-finite numeric value '{value}'; using {default}."
    return number, None


def _bounded_limit(value: Any, default: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(number, maximum))


def _split_fields(fields: str | None) -> set[str] | None:
    if not fields:
        return None
    selected = {part.strip() for part in fields.split(",") if part.strip()}
    return selected or None


def _filter_fields(item: dict, fields: set[str] | None) -> dict:
    if not fields:
        return item
    return {key: value for key, value in item.items() if key in fields}


def _compact_personality_preset(item: dict) -> dict:
    return {
        "personality_id": item.get("personality_id"),
        "name": item.get("name"),
        "description": item.get("description"),
        "traits": item.get("traits", {}),
    }


def _compact_profession_preset(item: dict) -> dict:
    profession = item.get("profession", {})
    return {
        "profession_id": item.get("profession_id"),
        "name": item.get("name"),
        "description": item.get("description"),
        "domain": item.get("domain"),
        "role": profession.get("role"),
        "objects": profession.get("objects", [])[:8],
        "methods": profession.get("methods", [])[:8],
        "evaluation": profession.get("evaluation", [])[:8],
    }


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_override_object(overrides: dict, field_name: str) -> str | None:
    if field_name in overrides and not isinstance(overrides[field_name], dict):
        return (
            f"Invalid JSON in field 'overrides.{field_name}': expected an object, "
            f"got {type(overrides[field_name]).__name__}."
        )
    return None


def _clean_text(value, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_memory_type(memory_type) -> str:
    return _clean_text(memory_type, "fact")


def _normalize_content(content) -> str:
    return _clean_text(content)


def _warning(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _safe_refresh_summary(project_id: str) -> tuple[dict | None, str | None]:
    try:
        return curator.refresh_project_summary(project_id=project_id), None
    except Exception as exc:
        return None, _warning(exc)


# ═══════════════════════════════════════════════════════════════════════════
# Palace Management
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_project(
    name: str,
    slug: str | None = None,
    description: str = "",
    context_type: str = "auto",
    profile_id: str | None = None,
) -> dict:
    """Create an isolated context for software or any other ongoing matter."""
    name = _clean_text(name)
    if not name:
        return {"error": "name cannot be empty."}
    context_type = _clean_text(context_type, "auto").lower()
    allowed = {"auto", "software", "research", "business", "learning", "general"}
    if context_type not in allowed:
        return {"error": f"context_type must be one of {sorted(allowed)}"}
    if profile_id and not db.get_memory_profile(profile_id):
        return {"error": f"Memory profile '{profile_id}' not found."}
    existing = db.get_project_by_name(name)
    if existing:
        return {"warning": f"Project '{name}' already exists.", "project": existing}
    project = db.create_project(name, slug, description, context_type, profile_id=profile_id)
    summary = curator.refresh_project_summary(project_id=project["id"])
    return {"status": "created", "project": project,
            "active_profile": db.get_active_profile_for_project(project["id"]),
            "project_summary": summary.get("summary")}


@mcp.tool()
def create_context(
    name: str,
    description: str = "",
    context_type: str = "auto",
    profile_id: str | None = None,
) -> dict:
    """Create a general context: software, research, business, learning, or other."""
    name = _clean_text(name)
    if not name:
        return {"error": "name cannot be empty."}
    context_type = _clean_text(context_type, "auto").lower()
    allowed = {"auto", "software", "research", "business", "learning", "general"}
    if context_type not in allowed:
        return {"error": f"context_type must be one of {sorted(allowed)}"}
    if profile_id and not db.get_memory_profile(profile_id):
        return {"error": f"Memory profile '{profile_id}' not found."}
    existing = db.get_project_by_name(name)
    if existing:
        return {"warning": f"Context '{name}' already exists.", "context": existing}
    context = db.create_project(
        name, description=description, context_type=context_type, profile_id=profile_id,
    )
    summary = curator.refresh_project_summary(project_id=context["id"])
    return {"status": "created", "context": context,
            "active_profile": db.get_active_profile_for_project(context["id"]),
            "top_level_memory": summary}


@mcp.tool()
def list_projects() -> dict:
    """List all projects."""
    projects = db.list_projects()
    return {"count": len(projects), "projects": projects}


@mcp.tool()
def list_memory_profiles() -> dict:
    """List built-in and custom memory profiles."""
    profiles = db.list_memory_profiles()
    return {"count": len(profiles), "profiles": profiles}


@mcp.tool()
def list_personality_presets(
    id: str | None = None,
    offset: int = 0,
    limit: int = 50,
    compact: bool = True,
    fields: str | None = None,
) -> dict:
    """List personality presets with pagination, compact output, and field filtering."""
    presets = db.list_personality_presets()
    if id:
        preset = next((item for item in presets if item.get("personality_id") == id), None)
        if not preset:
            return {"error": f"Personality preset '{id}' not found."}
        item = _compact_personality_preset(preset) if compact else preset
        return {"count": 1, "personality": _filter_fields(item, _split_fields(fields))}
    total = len(presets)
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError, OverflowError):
        offset = 0
    limit = _bounded_limit(limit, 50, 100)
    page = presets[offset:offset + limit]
    if compact:
        page = [_compact_personality_preset(item) for item in page]
    selected = _split_fields(fields)
    return {
        "count": len(page),
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
        "personalities": [_filter_fields(item, selected) for item in page],
    }


@mcp.tool()
def list_profession_presets(
    id: str | None = None,
    offset: int = 0,
    limit: int = 50,
    compact: bool = True,
    fields: str | None = None,
) -> dict:
    """List the 216 profession presets with pagination, compact output, and field filtering."""
    presets = db.list_profession_presets()
    if id:
        preset = next((item for item in presets if item.get("profession_id") == id), None)
        if not preset:
            return {"error": f"Profession preset '{id}' not found."}
        item = _compact_profession_preset(preset) if compact else preset
        return {"count": 1, "profession": _filter_fields(item, _split_fields(fields))}
    total = len(presets)
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError, OverflowError):
        offset = 0
    limit = _bounded_limit(limit, 50, 100)
    page = presets[offset:offset + limit]
    if compact:
        page = [_compact_profession_preset(item) for item in page]
    selected = _split_fields(fields)
    return {
        "count": len(page),
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
        "professions": [_filter_fields(item, selected) for item in page],
    }


@mcp.tool()
def get_setup_status(compact: bool = True, include_presets: bool = True) -> dict:
    """Return first-run personality/profession setup status without huge dumps."""
    profile = db.get_memory_profile(cfg.default_profile_id)
    personalities = db.list_personality_presets()
    professions = db.list_profession_presets()
    profiles = db.list_memory_profiles()
    status = {
        "configured": bool(cfg.default_profile_id),
        "setup_required": cfg.setup_required() or profile is None,
        "default_profile_id": cfg.default_profile_id or None,
        "default_personality_id": cfg.default_personality_id or None,
        "default_profession_id": cfg.default_profession_id or None,
        "setup_completed_at": cfg.setup_completed_at or None,
        "default_profile": profile,
        "available_counts": {
            "personalities": len(personalities),
            "professions": len(professions),
            "profiles": len(profiles),
        },
        "available_profiles": [
            {
                "profile_id": item.get("profile_id"),
                "name": item.get("name"),
                "description": item.get("description"),
                "source": item.get("source"),
            }
            for item in profiles
        ] if compact else profiles,
        "configure_command": "memery configure",
        "preset_lookup": (
            "Use list_personality_presets(id='...') or "
            "list_profession_presets(id='...') for one preset; pass compact=false "
            "only when the full scaffold is needed."
        ),
    }
    if include_presets:
        status["available_personalities"] = (
            [_compact_personality_preset(item) for item in personalities]
            if compact else personalities
        )
        status["available_professions"] = (
            [_compact_profession_preset(item) for item in professions[:50]]
            if compact else professions
        )
        if compact and len(professions) > 50:
            status["available_professions_truncated"] = True
            status["available_professions_next_offset"] = 50
    if cfg.default_profile_id and profile is None:
        status["warning"] = (
            f"Configured default profile '{cfg.default_profile_id}' was not found."
        )
    return status


@mcp.tool()
def configure_memory_defaults(
    personality_id: str | None = None,
    profession_id: str | None = None,
    profile_id: str | None = None,
) -> dict:
    """Set the default memory profile from personality+profession presets.

    Prefer personality_id + profession_id. profile_id is kept for compatibility
    with existing saved profiles.
    """
    if personality_id and not profession_id and not profile_id:
        existing = db.get_memory_profile(personality_id)
        if existing:
            profile_id = personality_id
            personality_id = None
    if profile_id:
        profile = db.get_memory_profile(profile_id)
        if not profile:
            return {"error": f"Memory profile '{profile_id}' not found."}
        personality_id = profile.get("development_plan", {}).get("personality_id")
        profession_id = profile.get("development_plan", {}).get("profession_id")
    else:
        if not personality_id or not profession_id:
            return {
                "error": (
                    "Provide personality_id and profession_id, or provide an "
                    "existing profile_id for compatibility."
                )
            }
        try:
            profile = db.compose_memory_profile(personality_id, profession_id)
        except ValueError as exc:
            return {"error": str(exc)}
    cfg.mark_profile_setup(
        profile["profile_id"],
        personality_id=personality_id,
        profession_id=profession_id,
    )
    return {
        "status": "configured",
        "configured": cfg.profile_setup_configured,
        "default_profile_id": cfg.default_profile_id,
        "default_personality_id": cfg.default_personality_id or None,
        "default_profession_id": cfg.default_profession_id or None,
        "setup_completed_at": cfg.setup_completed_at,
        "default_profile": profile,
        "note": (
            "New projects/contexts without an explicit profile will use this "
            "composed personality/profession profile."
        ),
    }


@mcp.tool()
def get_memory_profile(profile_id: str) -> dict:
    """Get a personality/profession memory profile."""
    profile = db.get_memory_profile(profile_id)
    if not profile:
        return {"error": f"Memory profile '{profile_id}' not found."}
    return {"profile": profile}


@mcp.tool()
def create_memory_profile(
    profile_id: str,
    traits: Any = None,
    profession: Any = None,
    personality_id: str | None = None,
    profession_id: str | None = None,
    development_plan: Any = None,
    calibration: Any = None,
    overrides: Any = None,
    name: str | None = None,
    description: str | None = None,
    version: str = "1",
) -> dict:
    """Create or update a memory profile from preset ids or object configs.

    Preferred path:
    create_memory_profile(
        profile_id="rigorous_stats_v1",
        personality_id="rigorous_validator_v1",
        profession_id="statistician_v1",
        overrides={"calibration": {"store_threshold": 0.8}},
    )

    `calibration` is numeric scoring configuration, not prose guidance. Useful
    keys include trait_weight, profession_weight, learning_weight,
    risk_penalty_weight, store_threshold, and retrieve_threshold.
    """
    existing_profile_row = db.conn.execute(
        "SELECT source FROM memory_profiles WHERE profile_id=?", (_clean_text(profile_id),)
    ).fetchone()
    if existing_profile_row and existing_profile_row["source"] == "built-in":
        return {
            "error": f"Built-in memory profile '{profile_id}' cannot be overwritten."
        }
    overrides_obj, error = _parse_json_object_field(overrides, "overrides")
    if error:
        return {"error": error}
    for field_name in ("traits", "profession", "development_plan", "calibration"):
        error = _validate_override_object(overrides_obj, field_name)
        if error:
            return {"error": error}

    if personality_id or profession_id:
        if not personality_id or not profession_id:
            return {
                "error": (
                    "Provide both personality_id and profession_id when creating "
                    "a profile from presets."
                )
            }
        try:
            base_profile = compose_profile_from_presets(
                personality_id,
                profession_id,
                profile_id=profile_id,
                name=name,
            )
        except ValueError as exc:
            return {"error": str(exc)}
        traits_obj = dict(base_profile.get("traits", {}))
        profession_obj = dict(base_profile.get("profession", {}))
        plan_obj = dict(base_profile.get("development_plan", {}))
        calibration_obj = dict(base_profile.get("calibration", {}))
        name = name or base_profile.get("name")
        description = description or base_profile.get("description")
        version = version or base_profile.get("version", "1")
    else:
        traits_obj, error = _parse_json_object_field(traits, "traits")
        if error:
            return {"error": error}
        profession_obj, error = _parse_json_object_field(profession, "profession")
        if error:
            return {"error": error}
        plan_obj, error = _parse_json_object_field(development_plan, "development_plan")
        if error:
            return {"error": error}
        calibration_obj, error = _parse_json_object_field(calibration, "calibration")
        if error:
            return {"error": error}

    if traits is not None and (personality_id or profession_id):
        parsed, error = _parse_json_object_field(traits, "traits")
        if error:
            return {"error": error}
        traits_obj = _deep_merge(traits_obj, parsed)
    if profession is not None and (personality_id or profession_id):
        parsed, error = _parse_json_object_field(profession, "profession")
        if error:
            return {"error": error}
        profession_obj = _deep_merge(profession_obj, parsed)
    if development_plan is not None and (personality_id or profession_id):
        parsed, error = _parse_json_object_field(development_plan, "development_plan")
        if error:
            return {"error": error}
        plan_obj = _deep_merge(plan_obj, parsed)
    if calibration is not None and (personality_id or profession_id):
        parsed, error = _parse_json_object_field(calibration, "calibration")
        if error:
            return {"error": error}
        calibration_obj = _deep_merge(calibration_obj, parsed)

    traits_obj = _deep_merge(traits_obj, overrides_obj.get("traits", {}))
    profession_obj = _deep_merge(profession_obj, overrides_obj.get("profession", {}))
    plan_obj = _deep_merge(plan_obj, overrides_obj.get("development_plan", {}))
    calibration_obj = _deep_merge(calibration_obj, overrides_obj.get("calibration", {}))
    name = overrides_obj.get("name", name)
    description = overrides_obj.get("description", description)
    version = overrides_obj.get("version", version)

    try:
        profile = db.upsert_memory_profile(
            profile_id=profile_id,
            traits=traits_obj,
            profession=profession_obj,
            development_plan=plan_obj,
            calibration=calibration_obj,
            name=name,
            description=description,
            version=version,
            source="composed" if personality_id or profession_id else "custom",
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return {"status": "saved", "profile": profile}


@mcp.tool()
def delete_memory_profile(profile_id: str) -> dict:
    """Delete a custom or composed memory profile. Built-in profiles are protected."""
    result = db.delete_memory_profile(profile_id)
    if not result:
        return {"error": f"Memory profile '{profile_id}' not found."}
    if result.get("protected"):
        return {
            "error": f"Built-in memory profile '{profile_id}' cannot be deleted.",
            "profile": result.get("profile"),
        }
    return {"status": "deleted", "profile_id": profile_id}


@mcp.tool()
def set_project_profile(project_name: str, profile_id: str | None) -> dict:
    """Bind a memory profile to a project/context."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    updated = db.set_project_profile(project["id"], profile_id)
    if not updated:
        return {"error": f"Memory profile '{profile_id}' not found."}
    summary = curator.refresh_project_summary(project_id=project["id"])
    return {
        "status": "bound",
        "project": updated,
        "active_profile": db.get_active_profile_for_project(project["id"]),
        "project_summary": summary.get("summary"),
    }


@mcp.tool()
def create_wing(project_name: str, name: str, slug: str | None = None,
                description: str = "") -> dict:
    """Create a wing (person/project/topic) in the memory palace."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    name = _clean_text(name)
    if not name:
        return {"error": "name cannot be empty."}
    wing = db.create_wing(project["id"], name, slug, description)
    return {"status": "created", "wing": wing}


@mcp.tool()
def list_wings(project_name: str | None = None) -> dict:
    """List wings in the palace."""
    pid = None
    if project_name:
        project = db.get_project_by_name(project_name)
        if not project:
            return {"error": f"Project '{project_name}' not found."}
        pid = project["id"]
    wings = db.list_wings(pid)
    return {"count": len(wings), "wings": wings}


@mcp.tool()
def create_room(wing_name: str, project_name: str,
                name: str, slug: str | None = None,
                description: str = "") -> dict:
    """Create a room (topic) within a wing."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    clean_wing_name = _clean_text(wing_name)
    if not clean_wing_name:
        return {"error": "wing_name cannot be empty."}
    wing = db.get_wing_by_slug(
        project["id"], clean_wing_name.lower().replace(" ", "_")
    )
    if not wing:
        return {"error": f"Wing '{wing_name}' not found."}
    room = db.create_room(wing["id"], name, slug, description)
    return {"status": "created", "room": room}


@mcp.tool()
def list_rooms(wing_name: str, project_name: str) -> dict:
    """List rooms in a wing."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    clean_wing_name = _clean_text(wing_name)
    if not clean_wing_name:
        return {"error": "wing_name cannot be empty."}
    wing = db.get_wing_by_slug(
        project["id"], clean_wing_name.lower().replace(" ", "_")
    )
    if not wing:
        return {"error": f"Wing '{wing_name}' not found."}
    rooms = db.list_rooms(wing["id"])
    return {"count": len(rooms), "rooms": rooms}


@mcp.tool()
def list_halls() -> dict:
    """List all hall types (memory classifications)."""
    halls = db.list_halls()
    return {"count": len(halls), "halls": halls}


@mcp.tool()
def get_service_status() -> dict:
    """Return the active transport mode and shared endpoint, if any."""
    return {"status": "ok", "service": "memery", **_runtime}


# ═══════════════════════════════════════════════════════════════════════════
# Memory Write/Read
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def write_memory(
    project_name: str, memory_type: str, title: str, content: str,
    wing_name: str | None = None, room_name: str | None = None,
    hall_id: str = "general",
    tags: Any = None, source_files: Any = None,
    importance: float = 0.5, confidence: float = 0.5,
    novelty: float = 0.5, reusability: float = 0.5,
    actionability: float = 0.5,
    refresh_summary: bool = True,
) -> dict:
    """Write a memory directly."""
    memory_type = _normalize_memory_type(memory_type)
    if memory_type in TOP_LEVEL_MEMORY_TYPES:
        return {
            "error": (
                f"'{memory_type}' is managed as a pinned singleton. Use "
                "refresh_project_summary or update_latest_conversation_summary."
            )
        }
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    project_id = project["id"]

    wing_id = None
    room_id = None
    if wing_name:
        wing = db.get_wing_by_slug(project_id, wing_name.lower().replace(" ", "_"))
        if wing:
            wing_id = wing["id"]
            if room_name:
                room = db.get_room_by_slug(wing_id, room_name.lower().replace(" ", "_"))
                if room:
                    room_id = room["id"]

    content = _normalize_content(content)
    if not content:
        return {"error": "content cannot be empty."}

    warnings = []
    tags_list, tags_warning = _parse_string_list(tags, "tags")
    files_list, files_warning = _parse_string_list(source_files, "source_files")
    warnings.extend(w for w in (tags_warning, files_warning) if w)
    if not db.get_hall(hall_id):
        warnings.append(f"Unknown hall_id '{hall_id}'; using 'general'.")
        hall_id = "general"
    base_score = db._calc_score(importance, confidence, novelty, reusability, actionability)
    active_profile = db.get_active_profile_for_project(project_id)
    profile_score = score_memory_with_profile(
        text=content,
        memory_type=memory_type,
        base_score=base_score,
        profile=active_profile,
        source_type="direct",
    )

    try:
        memory = db.write_memory(
            project_id=project_id, memory_type=memory_type,
            title=title, content=content,
            wing_id=wing_id, room_id=room_id, hall_id=hall_id,
            confidence_label=Confidence.INFERRED, confidence_score=confidence,
            tags=tags_list, source_files=files_list,
            importance=importance, confidence=confidence,
            novelty=novelty, reusability=reusability, actionability=actionability,
            profile_id=profile_score.get("profile_id"),
            memory_layer=profile_score.get("memory_layer", "semantic"),
            base_score=profile_score.get("base_score"),
            store_score=profile_score.get("store_score"),
            trait_features=profile_score.get("trait_features"),
            profession_features=profile_score.get("profession_features"),
            score_reason=profile_score.get("score_reason", ""),
        )
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        vector_store.insert(memory["id"], project_id, content)
    except Exception as exc:
        warnings.append(f"Vector index write failed; SQLite memory was saved. {_warning(exc)}")
    if not refresh_summary:
        result = {"status": "written", "memory": memory, "summary_refresh": "deferred"}
        if warnings:
            result["warnings"] = warnings
        return result
    summary, summary_warning = _safe_refresh_summary(project_id)
    if summary_warning:
        warnings.append(f"Project summary refresh failed; memory was saved. {summary_warning}")
    result = {"status": "written", "memory": memory,
              "project_summary": summary.get("summary") if summary else None}
    if warnings:
        result["warnings"] = warnings
    return result


@mcp.tool()
def write_memories_batch(project_name: str, memories: Any) -> dict:
    """Write a JSON array of memories in one transaction and refresh once."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    warnings = []
    if isinstance(memories, list):
        items = memories
    else:
        try:
            items = json.loads(memories)
        except (json.JSONDecodeError, TypeError) as exc:
            return {"error": f"Invalid memories JSON: {exc}"}
    if not isinstance(items, list) or not items:
        return {"error": "memories must be a non-empty JSON array."}
    if len(items) > 5000:
        return {"error": "A single batch cannot exceed 5000 memories."}

    rows = []
    vector_items = []
    active_profile = db.get_active_profile_for_project(project["id"])
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return {"error": f"Memory at index {index} must be an object."}
        memory_type = _normalize_memory_type(item.get("memory_type", "fact"))
        if memory_type in TOP_LEVEL_MEMORY_TYPES:
            return {"error": f"Memory at index {index} uses protected type '{memory_type}'."}
        content = _normalize_content(item.get("content"))
        if not content:
            return {"error": f"Memory at index {index} has empty content."}
        hall_id = str(item.get("hall_id", "general") or "general")
        if not db.get_hall(hall_id):
            warnings.append(
                f"Memory at index {index} uses unknown hall_id '{hall_id}'; using 'general'."
            )
            hall_id = "general"
        tags_list, tags_warning = _parse_string_list(
            item.get("tags"), f"memories[{index}].tags"
        )
        files_list, files_warning = _parse_string_list(
            item.get("source_files"), f"memories[{index}].source_files"
        )
        warnings.extend(w for w in (tags_warning, files_warning) if w)
        row = {
            **item,
            "project_id": project["id"],
            "memory_type": memory_type,
            "title": _clean_text(item.get("title")) or curator._extract_title(content),
            "content": content,
            "hall_id": hall_id,
            "tags": tags_list,
            "source_files": files_list,
        }
        numeric = {}
        for field_name in (
            "importance", "confidence", "novelty", "reusability", "actionability",
        ):
            numeric[field_name], numeric_warning = _coerce_float(
                item.get(field_name, 0.0), 0.0
            )
            if numeric_warning:
                warnings.append(f"Memory at index {index}: {field_name}: {numeric_warning}")
            row[field_name] = numeric[field_name]
        base_score = db._calc_score(
            numeric["importance"], numeric["confidence"], numeric["novelty"],
            numeric["reusability"], numeric["actionability"],
        )
        profile_score = score_memory_with_profile(
            text=content,
            memory_type=memory_type,
            base_score=base_score,
            profile=active_profile,
            source_type=str(item.get("source_type", "batch")),
        )
        row.update({
            "profile_id": profile_score.get("profile_id"),
            "memory_layer": profile_score.get("memory_layer", "semantic"),
            "base_score": profile_score.get("base_score"),
            "store_score": profile_score.get("store_score"),
            "trait_features": profile_score.get("trait_features"),
            "profession_features": profile_score.get("profession_features"),
            "score_reason": profile_score.get("score_reason", ""),
        })
        rows.append(row)

    try:
        written = db.write_memories_batch(rows)
    except ValueError as exc:
        return {"error": str(exc)}
    for memory in written:
        vector_items.append({
            "id": memory["id"],
            "project_id": project["id"],
            "text": memory["content"],
        })
    try:
        vector_store.insert_batch(vector_items)
    except Exception as exc:
        warnings.append(f"Vector index batch write failed; SQLite memories were saved. {_warning(exc)}")
    summary, summary_warning = _safe_refresh_summary(project["id"])
    if summary_warning:
        warnings.append(f"Project summary refresh failed; memories were saved. {summary_warning}")
    result = {
        "status": "written",
        "count": len(written),
        "memories": written,
        "project_summary": summary.get("summary") if summary else None,
    }
    if warnings:
        result["warnings"] = warnings
    return result


@mcp.tool()
def search_memory(
    project_name: str | None = None, wing_name: str | None = None,
    room_name: str | None = None, hall_id: str | None = None,
    memory_type: str | None = None, memory_layer: str | None = None,
    keyword: str | None = None,
    semantic_query: str | None = None, tags: Any = None,
    confidence_label: str | None = None, min_score: float | None = None,
    limit: int = 20,
) -> dict:
    """Search memories by various filters."""
    project_id = None
    wing_id = None
    room_id = None

    if project_name:
        project = db.get_project_by_name(project_name)
        if not project:
            return {"error": f"Project '{project_name}' not found."}
        project_id = project["id"]
        if wing_name:
            wing = db.get_wing_by_slug(
                project_id, _clean_text(wing_name).lower().replace(" ", "_")
            )
            if not wing:
                return {"error": f"Wing '{wing_name}' not found."}
            wing_id = wing["id"]
            if room_name:
                room = db.get_room_by_slug(
                    wing_id, _clean_text(room_name).lower().replace(" ", "_")
                )
                if not room:
                    return {"error": f"Room '{room_name}' not found."}
                room_id = room["id"]

    limit = _bounded_limit(limit, 20, 200)
    active_profile = db.get_active_profile_for_project(project_id) if project_id else None
    if semantic_query:
        vr = vector_store.search(semantic_query, project_id=project_id, limit=limit)
        ids = [r.id for r in vr]
        results = db.search_memories_by_ids(ids) if ids else []
        for mem in results:
            for v in vr:
                if v.id == mem["id"]:
                    mem["semantic_similarity"] = v.similarity
                    break
            rerank = score_retrieval_with_profile(
                mem, semantic_query, active_profile,
                semantic_relevance=mem.get("semantic_similarity"),
            )
            mem.update(rerank)
        if memory_layer:
            results = [mem for mem in results if mem.get("memory_layer") == memory_layer]
        results.sort(key=lambda mem: mem.get("retrieve_score", 0), reverse=True)
        db.record_memory_hits([mem["id"] for mem in results])
    else:
        tags_list, _ = _parse_string_list(tags, "tags")
        results = db.search_memories(
            project_id=project_id, wing_id=wing_id, room_id=room_id,
            hall_id=hall_id, memory_type=memory_type, memory_layer=memory_layer,
            keyword=keyword,
            tags=tags_list or None, confidence_label=confidence_label,
            min_score=min_score, limit=limit,
        )
        query_text = keyword or memory_type or memory_layer or ""
        for mem in results:
            mem.update(score_retrieval_with_profile(mem, query_text, active_profile))
        results.sort(key=lambda mem: mem.get("retrieve_score", mem.get("score", 0)), reverse=True)
        db.record_memory_hits([mem["id"] for mem in results])

    return {"count": len(results), "results": results}


@mcp.tool()
def recall_for_task(project_name: str, task: str, limit: int = 10) -> dict:
    """Recall relevant memories for a task."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    project_id = project["id"]
    active_profile = db.get_active_profile_for_project(project_id)

    limit = _bounded_limit(limit, 10, 100)
    sr = vector_store.search(task, project_id=project_id, limit=limit)
    kr = db.search_memories(project_id=project_id, keyword=task, limit=limit)

    seen = set()
    combined = []
    for r in sr:
        mem = db.get_memory(r.id)
        if mem and r.id not in seen:
            seen.add(r.id)
            mem["semantic_similarity"] = r.similarity
            combined.append(mem)
    for r in kr:
        if r["id"] not in seen:
            seen.add(r["id"])
            combined.append(r)
    for m in combined:
        m.update(score_retrieval_with_profile(
            m, task, active_profile, semantic_relevance=m.get("semantic_similarity"),
        ))
    combined.sort(key=lambda m: m.get("retrieve_score", 0), reverse=True)
    db.record_memory_hits([m["id"] for m in combined])

    return {
        "task": task, "project": project_name,
        "active_profile": active_profile,
        "relevant_memories": combined[:limit],
    }


@mcp.tool()
def wake_up(project_name: str) -> dict:
    """Activate memory context for this session. Call once at session start.

    Returns the durable project summary plus compact L0+L1 context. No API
    keys needed. Use search_memory / recall_for_task for deeper L2/L3 queries.

    L0: project identity, wing structure
    L1: top decisions, key rules, recent decisions
    """
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    project_id = project["id"]
    active_profile = db.get_active_profile_for_project(project_id)

    # L0: Palace layout
    wings = db.list_wings(project_id)
    wing_names = [w["name"] for w in wings[:5]]

    refreshed = curator.get_top_level_context(project_id=project_id)
    summary = refreshed.get("summary")
    project_core = refreshed.get("project_core")
    latest_conversation = refreshed.get("latest_conversation_summary")

    # L1: Key memories, with pinned project context sorted first.
    top_memories = db.search_memories(project_id=project_id, min_score=0.4, limit=10)
    rules = db.search_memories(project_id=project_id, memory_type="coding_rule", limit=3)
    decisions = db.list_decisions(project_id, limit=3)
    pending = db.list_pending_candidates(project_id)

    # Compact context for AI
    context = {
        "project": project_name,
        "active_profile": active_profile,
        "top_level_memory": {
            "project_core": project_core.get("content") if project_core else None,
            "latest_conversation_summary": (
                latest_conversation.get("content") if latest_conversation else None
            ),
        },
        "project_summary": summary.get("content") if summary else None,
        "wings": wing_names,
        "key_rules": [{"title": r["title"], "content": r["content"][:120]} for r in rules],
        "recent_decisions": [{"title": d["title"], "rationale": d.get("rationale", "")[:80]} for d in decisions],
        "top_memories": [m["title"] for m in top_memories[:5]],
        "pending_review": len(pending),
        "hint": "Use search_memory() for deep recall, recall_for_task() for task context, analyze_project_code() for code insights.",
    }

    stats = db.get_stats(project_id)
    return {"context": context, "stats": stats}


@mcp.tool()
def get_context_bundle(project_name: str) -> dict:
    """Get full context bundle: active memories, tasks, decisions."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    project_id = project["id"]

    refreshed = curator.get_top_level_context(project_id=project_id)
    summary = refreshed.get("summary")
    project_core = refreshed.get("project_core")
    latest_conversation = refreshed.get("latest_conversation_summary")
    memories = db.list_memories(project_id=project_id, limit=50)
    tasks = db.list_task_snapshots(project_id, limit=10)
    decisions = db.list_decisions(project_id, limit=10)
    pending = db.list_pending_candidates(project_id)
    active_profile = db.get_active_profile_for_project(project_id)

    by_type: dict[str, list] = {}
    for m in memories:
        t = m.get("memory_type", "other")
        by_type.setdefault(t, []).append(m["title"])

    return {
        "project": project_name,
        "active_profile": active_profile,
        "top_level_memory": {
            "project_core": project_core.get("content") if project_core else None,
            "latest_conversation_summary": (
                latest_conversation.get("content") if latest_conversation else None
            ),
        },
        "total_memories": len(memories),
        "memories_by_type": {k: len(v) for k, v in by_type.items()},
        "project_summary": summary["content"] if summary else None,
        "active_tasks": [{"name": t["task_name"], "status": t["status"]} for t in tasks],
        "recent_decisions": [{"title": d["title"], "created": d["created_at"]} for d in decisions],
        "pending_candidates": len(pending),
    }


@mcp.tool()
def get_top_level_memory(project_name: str) -> dict:
    """Return pinned project core and latest-conversation memory only."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    refreshed = curator.get_top_level_context(project_id=project["id"])
    return {
        "project": project_name,
        "project_core": refreshed.get("project_core"),
        "latest_conversation_summary": refreshed.get("latest_conversation_summary"),
        "project_summary": refreshed.get("summary"),
    }


@mcp.tool()
def get_memory(memory_id: str) -> dict:
    """Get a specific memory by ID with its graph neighbors."""
    mem = db.get_memory(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found."}
    neighbors = db.get_neighbors(memory_id)
    return {"memory": mem, "neighbors": neighbors}


@mcp.tool()
def delete_memory(memory_id: str) -> dict:
    """Delete a memory."""
    mem = db.get_memory(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found."}
    if mem["memory_type"] in PROTECTED_TYPES and mem.get("importance", 0) >= 0.8:
        return {"error": f"Cannot delete protected memory.", "memory": mem}
    db.delete_memory(memory_id)
    warnings = []
    try:
        vector_store.delete(memory_id)
    except Exception as exc:
        warnings.append(f"Vector index delete failed; SQLite memory was deleted. {_warning(exc)}")
    summary, summary_warning = _safe_refresh_summary(mem["project_id"])
    if summary_warning:
        warnings.append(f"Project summary refresh failed. {summary_warning}")
    result = {
        "status": "deleted", "memory_id": memory_id,
        "project_summary": summary.get("summary") if summary else None,
    }
    if warnings:
        result["warnings"] = warnings
    return result


@mcp.tool()
def deprecate_memory(memory_id: str) -> dict:
    """Mark a memory as deprecated."""
    mem = db.get_memory(memory_id)
    if not mem:
        return {"status": "not_found", "memory_id": memory_id}
    if mem["memory_type"] in PROTECTED_TYPES:
        return {"error": "Cannot deprecate protected memory.", "memory": mem}
    ok = db.deprecate_memory(memory_id)
    summary = curator.refresh_project_summary(project_id=mem["project_id"])
    return {"status": "deprecated" if ok else "not_found", "memory_id": memory_id,
            "project_summary": summary.get("summary")}


@mcp.tool()
def list_memories(project_name: str | None = None,
                  wing_name: str | None = None,
                  status: str = "active", limit: int = 200) -> dict:
    """List memories."""
    project_id = None
    wing_id = None
    if project_name:
        project = db.get_project_by_name(project_name)
        if not project:
            return {"error": f"Project '{project_name}' not found."}
        project_id = project["id"]
        if wing_name:
            wing = db.get_wing_by_slug(project_id, _clean_text(wing_name).lower().replace(" ", "_"))
            if not wing:
                return {"error": f"Wing '{wing_name}' not found."}
            wing_id = wing["id"]
    limit = _bounded_limit(limit, 200, 1000)
    memories = db.list_memories(project_id=project_id, wing_id=wing_id,
                                status=status, limit=limit)
    return {"count": len(memories), "memories": memories}


# ═══════════════════════════════════════════════════════════════════════════
# Memory Curator Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def ingest_conversation(project_name: str, conversation_text: str,
                        source_type: str = "conversation") -> dict:
    """Ingest raw conversation and extract candidate memories."""
    return curator.ingest_conversation(project_name, conversation_text, source_type)


@mcp.tool()
def ingest_update(context_name: str, update_text: str,
                  source_type: str = "update") -> dict:
    """Ingest any ongoing matter update and refresh its pinned context."""
    return curator.ingest_conversation(context_name, update_text, source_type)


@mcp.tool()
def extract_memory_candidates(project_name: str,
                              conversation_text: str) -> dict:
    """Extract structured candidate memories from conversation."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    import re
    segments = re.split(r'(?<=[。！？\.\!\?\n])\s*', conversation_text)
    segments = [s.strip() for s in segments if len(s.strip()) > 10]
    candidates_output = []
    for seg in segments[:20]:
        candidates_output.append({
            "project_name": project_name,
            "raw_text": seg,
            "should_store": None,
            "memory_type": None,
        })
    return {"candidates": candidates_output}


@mcp.tool()
def review_memory_candidate(candidate_id: str, decision: str,
                            merged_to: str | None = None,
                            reason: str | None = None) -> dict:
    """Review a candidate: accept / reject / merge / needs_review."""
    return curator.review_candidate(candidate_id, decision, merged_to, reason)


@mcp.tool()
def list_pending_candidates(project_name: str | None = None,
                            wing_name: str | None = None) -> dict:
    """List pending memory candidates."""
    pid = None
    wid = None
    if project_name:
        project = db.get_project_by_name(project_name)
        if not project:
            return {"error": f"Project '{project_name}' not found."}
        pid = project["id"]
        if wing_name:
            wing = db.get_wing_by_slug(pid, _clean_text(wing_name).lower().replace(" ", "_"))
            if not wing:
                return {"error": f"Wing '{wing_name}' not found."}
            wid = wing["id"]
    candidates = db.list_pending_candidates(pid, wid)
    return {"count": len(candidates), "candidates": candidates}


@mcp.tool()
def record_memory_feedback(
    project_name: str,
    outcome_type: str,
    task: str = "",
    memory_ids: Any = None,
    feedback: str = "",
    reflection: str = "",
) -> dict:
    """Record task outcome feedback for profile-aware memory growth.

    outcome_type: success | ineffective | harmful | missed_recall
    """
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    allowed = {"success", "ineffective", "harmful", "missed_recall"}
    if outcome_type not in allowed:
        return {"error": f"outcome_type must be one of {sorted(allowed)}"}
    ids, warning = _parse_string_list(memory_ids, "memory_ids")
    active_profile = db.get_active_profile_for_project(project["id"])
    outcome = db.record_outcome(
        project_id=project["id"],
        profile_id=active_profile.get("profile_id") if active_profile else None,
        outcome_type=outcome_type,
        task=task,
        memory_ids=ids,
        feedback=feedback,
        reflection=reflection,
    )
    generated_memory = None
    if feedback or reflection or outcome_type in {"harmful", "missed_recall"}:
        layer = "error" if outcome_type in {"harmful", "missed_recall"} else "procedural"
        memory_type = "error_pattern" if layer == "error" else "practice_pattern"
        title = (
            f"{outcome_type} feedback: {task[:48]}"
            if task else f"{outcome_type} memory feedback"
        )
        content = "\n".join([
            f"Outcome: {outcome_type}",
            f"Task: {task or '(not specified)'}",
            f"Feedback: {feedback or '(none)'}",
            f"Reflection: {reflection or '(none)'}",
            f"Referenced memories: {', '.join(ids) if ids else '(none)'}",
        ])
        base_score = 0.82 if layer == "error" else 0.72
        profile_score = score_memory_with_profile(
            text=content,
            memory_type=memory_type,
            base_score=base_score,
            profile=active_profile,
            source_type="feedback",
        )
        generated_memory = db.write_memory(
            project_id=project["id"],
            profile_id=profile_score.get("profile_id"),
            memory_layer=layer,
            memory_type=memory_type,
            title=title,
            content=content,
            hall_id="bugs" if layer == "error" else "rules",
            tags=["feedback", outcome_type, layer],
            source_files=ids,
            importance=0.85 if layer == "error" else 0.75,
            confidence=0.8,
            novelty=0.65,
            reusability=0.85,
            actionability=0.85,
            base_score=profile_score.get("base_score"),
            store_score=profile_score.get("store_score"),
            trait_features=profile_score.get("trait_features"),
            profession_features=profile_score.get("profession_features"),
            score_reason=profile_score.get("score_reason", ""),
            promotion_state="feedback",
        )
    result = {
        "status": "recorded",
        "outcome": outcome,
        "active_profile": active_profile,
        "generated_memory": generated_memory,
    }
    if warning:
        result["warning"] = warning
    return result


@mcp.tool()
def list_memory_feedback(project_name: str | None = None, limit: int = 50) -> dict:
    """List recorded outcome feedback."""
    project_id = None
    if project_name:
        project = _ensure_project(project_name)
        if "error" in project:
            return project
        project_id = project["id"]
    outcomes = db.list_outcomes(project_id, limit=_bounded_limit(limit, 50, 500))
    return {"count": len(outcomes), "outcomes": outcomes}


@mcp.tool()
def compact_project_memory(project_name: str) -> dict:
    """Compact project memories by merging/deprecating low-value ones."""
    return curator.compact_project_memory(project_name)


@mcp.tool()
def refresh_project_summary(
    project_name: str,
    latest_conversation_text: str | None = None,
    source_type: str = "conversation",
) -> dict:
    """Refresh pinned project context, optionally replacing the latest conversation."""
    return curator.refresh_project_summary(
        project_name=project_name,
        latest_conversation_text=latest_conversation_text,
        source_type=source_type,
    )


@mcp.tool()
def update_latest_conversation_summary(
    project_name: str,
    conversation_text: str,
    source_type: str = "conversation",
) -> dict:
    """Replace the pinned latest-conversation summary and refresh project context."""
    latest = curator.update_latest_conversation_summary(
        project_name=project_name,
        conversation_text=conversation_text,
        source_type=source_type,
    )
    if "error" in latest:
        return latest
    refreshed = curator.refresh_project_summary(project_name=project_name)
    return {
        "status": "updated",
        "latest_conversation_summary": latest.get("summary"),
        "project_core": refreshed.get("project_core"),
        "project_summary": refreshed.get("summary"),
    }


@mcp.tool()
def prune_low_value_memories(project_name: str) -> dict:
    """Prune low-value memories from a project."""
    return curator.prune_low_value_memories(project_name)


# ═══════════════════════════════════════════════════════════════════════════
# Graph Pipeline (graphify)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def register_provenance_source(
    project_name: str,
    uri: str,
    title: str = "",
    source_type: str = "document",
    version: str = "",
    content: str = "",
    content_hash: str = "",
    trust_score: float = 0.8,
    metadata: Any = None,
) -> dict:
    """Register an original source whose identity is sealed by a content hash."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    metadata_object, error = _parse_json_object_field(metadata, "metadata")
    if error:
        return {"error": error}
    uri = _clean_text(uri)
    if not uri:
        return {"error": "uri cannot be empty."}
    source = steelprint.register_source(
        project_id=project["id"], uri=uri, title=title,
        source_type=source_type, version=version, content=content,
        content_hash=content_hash, trust_score=trust_score,
        metadata=metadata_object,
    )
    return {"status": "registered", "source": source}


@mcp.tool()
def add_evidence_span(
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
    metadata: Any = None,
) -> dict:
    """Add an exact page/paragraph/line span to a provenance source."""
    metadata_object, error = _parse_json_object_field(metadata, "metadata")
    if error:
        return {"error": error}
    try:
        evidence = steelprint.add_evidence(
            source_id=source_id, quoted_text=quoted_text,
            page_number=page_number, paragraph_number=paragraph_number,
            section_title=section_title, line_start=line_start, line_end=line_end,
            char_start=char_start, char_end=char_end, locator=locator,
            metadata=metadata_object,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return {"status": "recorded", "evidence": evidence}


@mcp.tool()
def steelprint_memory(
    memory_id: str,
    evidence_ids: Any,
    claim_subject: str = "",
    claim_predicate: str = "",
    claim_object: str = "",
    support_type: str = "supports",
    entailment_score: float = 1.0,
) -> dict:
    """Bind a memory claim to exact evidence and automatically scan conflicts."""
    ids, error = _parse_json_array_field(evidence_ids, "evidence_ids")
    if error:
        return {"error": error}
    try:
        return steelprint.stamp_memory(
            memory_id=memory_id, evidence_ids=[str(item) for item in ids],
            claim_subject=claim_subject, claim_predicate=claim_predicate,
            claim_object=claim_object, support_type=support_type,
            entailment_score=entailment_score,
        )
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_memory_steelprint(memory_id: str) -> dict:
    """Return the complete memory-to-evidence provenance chain."""
    return steelprint.get_memory_steelprint(memory_id)


@mcp.tool()
def detect_memory_conflicts(project_name: str, memory_id: str | None = None) -> dict:
    """Detect incompatible steelprinted claims about the same subject and predicate."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    detected = steelprint.detect_conflicts(project["id"], memory_id=memory_id)
    open_conflicts = steelprint.list_conflicts(project["id"])
    return {
        "detected": len(detected), "open_count": len(open_conflicts),
        "conflicts": open_conflicts,
    }


@mcp.tool()
def list_memory_conflicts(project_name: str, status: str = "open") -> dict:
    """List memory conflicts by review status."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    conflicts = steelprint.list_conflicts(project["id"], status=status)
    return {"count": len(conflicts), "conflicts": conflicts}


@mcp.tool()
def resolve_memory_conflict(conflict_id: str, status: str = "resolved") -> dict:
    """Resolve, dismiss, or reopen a detected memory conflict."""
    try:
        conflict = steelprint.resolve_conflict(conflict_id, status=status)
    except ValueError as exc:
        return {"error": str(exc)}
    if not conflict:
        return {"error": f"Conflict '{conflict_id}' not found."}
    return {"status": "updated", "conflict": conflict}


@mcp.tool()
def verify_grounded_answer(
    project_name: str,
    question: str,
    answer: str,
    claims: Any = None,
    policy: str = "strict",
    min_coverage: float = 1.0,
    min_confidence: float = 0.65,
) -> dict:
    """Verify every factual answer sentence against its evidence citations.

    claims is a JSON array of:
    {"text": "...", "factual": true, "evidence_ids": ["..."]}.
    Strict policy refuses answers containing uncited, weak, or contradicted claims.
    """
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    parsed_claims = None
    if claims is not None:
        parsed_claims, error = _parse_json_array_field(claims, "claims")
        if error:
            return {"error": error}
    try:
        return steelprint.verify_answer(
            project_id=project["id"], question=question, answer=answer,
            claims=parsed_claims, policy=policy, min_coverage=min_coverage,
            min_confidence=min_confidence,
        )
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def run_hallucination_evaluation(
    project_name: str,
    name: str,
    cases: Any,
    policy: str = "strict",
) -> dict:
    """Run a reproducible hallucination benchmark and persist its metrics."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    parsed_cases, error = _parse_json_array_field(cases, "cases")
    if error:
        return {"error": error}
    try:
        return steelprint.evaluate(
            project_id=project["id"], name=name, cases=parsed_cases, policy=policy,
        )
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_hallucination_evaluations(project_name: str, limit: int = 20) -> dict:
    """List persisted hallucination benchmark runs and metrics."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    evaluations = steelprint.list_evaluations(
        project["id"], limit=_bounded_limit(limit, 20, 200)
    )
    return {"count": len(evaluations), "evaluations": evaluations}


@mcp.tool()
def analyze_project_code(project_name: str, paths: Any) -> dict:
    """Run graphify pipeline: extract code structure, cluster, find god nodes.

    Args:
        project_name: Target project.
        paths: JSON array of file/directory paths to analyze.
    """
    path_list, error = _parse_json_array_field(paths, "paths")
    if error:
        return {"error": error}
    path_list = [_clean_text(path) for path in path_list if _clean_text(path)]
    if not path_list:
        return {"error": "No paths provided."}
    return curator.analyze_project_code(project_name, path_list)


@mcp.tool()
def query_graph_path(memory_id_a: str, memory_id_b: str,
                     max_depth: int = 4) -> dict:
    """Find the path between two memory nodes in the graph."""
    if not db.get_memory(memory_id_a):
        return {"error": f"Memory '{memory_id_a}' not found."}
    if not db.get_memory(memory_id_b):
        return {"error": f"Memory '{memory_id_b}' not found."}
    max_depth = _bounded_limit(max_depth, 4, 20)
    path = db.find_path(memory_id_a, memory_id_b, max_depth)
    if path is None:
        return {"found": False, "path": []}
    return {"found": True, "length": len(path), "path": path}


@mcp.tool()
def get_graph_neighbors(memory_id: str, direction: str = "both") -> dict:
    """Get graph neighbors of a memory node."""
    if direction not in {"in", "out", "both"}:
        return {"error": "direction must be 'in', 'out', or 'both'."}
    if not db.get_memory(memory_id):
        return {"error": f"Memory '{memory_id}' not found."}
    neighbors = db.get_neighbors(memory_id, direction)
    return {"memory_id": memory_id, "neighbor_count": len(neighbors),
            "neighbors": neighbors}


@mcp.tool()
def get_graph_stats(project_name: str | None = None) -> dict:
    """Get graph and memory statistics."""
    pid = None
    if project_name:
        project = db.get_project_by_name(project_name)
        if not project:
            return {"error": f"Project '{project_name}' not found."}
        pid = project["id"]
    stats = db.get_stats(pid)
    return {"stats": stats}


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Graph (mempalace)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def add_knowledge_triple(
    subject: str, predicate: str, obj: str,
    valid_from: str | None = None, valid_to: str | None = None,
) -> dict:
    """Add a temporal knowledge triple (entity-relationship with time)."""
    subject = _clean_text(subject)
    predicate = _clean_text(predicate)
    obj = _clean_text(obj)
    if not subject or not predicate or not obj:
        return {"error": "subject, predicate, and obj cannot be empty."}
    return curator.add_triple(subject, predicate, obj, valid_from, valid_to)


@mcp.tool()
def query_entity(entity: str, as_of: str | None = None) -> dict:
    """Query all knowledge graph triples about an entity."""
    entity = _clean_text(entity)
    if not entity:
        return {"error": "entity cannot be empty."}
    return curator.query_entity(entity, as_of=as_of)


@mcp.tool()
def query_entity_timeline(entity: str) -> dict:
    """Get the timeline of changes for an entity."""
    entity = _clean_text(entity)
    if not entity:
        return {"error": "entity cannot be empty."}
    results = db.get_timeline(entity)
    return {"entity": entity, "count": len(results), "timeline": results}


@mcp.tool()
def invalidate_triple(subject: str, predicate: str, obj: str,
                      ended: str | None = None) -> dict:
    """Invalidate a temporal triple (mark as no longer true)."""
    subject = _clean_text(subject)
    predicate = _clean_text(predicate)
    obj = _clean_text(obj)
    if not subject or not predicate or not obj:
        return {"error": "subject, predicate, and obj cannot be empty."}
    try:
        ok = curator.palace.invalidate_knowledge(subject, predicate, obj, ended)
    except RuntimeError:
        row = db.conn.execute(
            """SELECT id FROM temporal_triples
               WHERE subject=? AND predicate=? AND object=? AND invalidated_at IS NULL
               ORDER BY created_at DESC LIMIT 1""",
            (subject, predicate, obj),
        ).fetchone()
        ok = db.invalidate_triple(row["id"], ended) if row else False
    return {"invalidated": ok, "subject": subject, "predicate": predicate, "object": obj}


# ═══════════════════════════════════════════════════════════════════════════
# Task & Decisions
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def record_decision(project_name: str, title: str, content: str,
                    rationale: str = "", alternatives: Any = None) -> dict:
    """Record an architectural design decision."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    title = _clean_text(title)
    content = _clean_text(content)
    if not title or not content:
        return {"error": "title and content cannot be empty."}
    alt_list, _ = _parse_string_list(alternatives, "alternatives")
    decision = db.record_decision(project["id"], title, content, rationale, alt_list)
    summary = curator.refresh_project_summary(project_id=project["id"])
    return {"status": "recorded", "decision": decision,
            "project_summary": summary.get("summary")}


@mcp.tool()
def list_decisions(project_name: str, limit: int = 50) -> dict:
    """List decisions for a project."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    decisions = db.list_decisions(project["id"], _bounded_limit(limit, 50, 500))
    return {"count": len(decisions), "decisions": decisions}


@mcp.tool()
def record_task_snapshot(project_name: str, task_name: str,
                         status: str = "pending", completed: Any = None,
                         remaining: Any = None, notes: str = "") -> dict:
    """Record a task snapshot."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    task_name = _clean_text(task_name)
    if not task_name:
        return {"error": "task_name cannot be empty."}
    completed_list, _ = _parse_string_list(completed, "completed")
    remaining_list, _ = _parse_string_list(remaining, "remaining")
    snapshot = db.record_task_snapshot(
        project["id"], task_name, status, completed_list, remaining_list, notes)
    summary = curator.refresh_project_summary(project_id=project["id"])
    return {"status": "recorded", "snapshot": snapshot,
            "project_summary": summary.get("summary")}


@mcp.tool()
def list_task_snapshots(project_name: str, limit: int = 50) -> dict:
    """List task snapshots for a project."""
    project = _ensure_project(project_name)
    if "error" in project:
        return project
    tasks = db.list_task_snapshots(project["id"], _bounded_limit(limit, 50, 500))
    return {"count": len(tasks), "tasks": tasks}


# ── Main entry point ───────────────────────────────────────────────────

def main(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str = "/mcp",
    json_response: bool = False,
    stateless_http: bool = False,
):
    """Run either an exclusive stdio instance or a shared HTTP singleton."""
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("transport must be 'stdio' or 'streamable-http'.")
    normalized_path = "/" + path.strip("/") if path.strip("/") else "/mcp"
    endpoint = (
        f"http://{host}:{port}{normalized_path}"
        if transport == "streamable-http" else None
    )
    metadata = {
        "mode": "http-singleton" if endpoint else "stdio",
        "transport": transport,
        "endpoint": endpoint,
    }
    try:
        service_lock = acquire_service_lock(cfg, metadata=metadata)
    except ServiceLockError as exc:
        exit_for_lock_error(exc)

    _runtime.update({**metadata, "pid": os.getpid()})
    try:
        if transport == "stdio":
            run_fastmcp_stdio(mcp)
            return
        mcp.settings.host = host
        mcp.settings.port = int(port)
        mcp.settings.streamable_http_path = normalized_path
        mcp.settings.json_response = bool(json_response)
        mcp.settings.stateless_http = bool(stateless_http)
        mcp.run(transport="streamable-http")
    finally:
        service_lock.release()
