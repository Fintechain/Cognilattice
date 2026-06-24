# -*- coding: utf-8 -*-
"""Command-line entry point for Memery."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from .config import get_config
from .db import MemoryDB
from .profiles import TRAIT_KEYS


def _version() -> str:
    from . import __version__
    try:
        pkg_version = importlib.metadata.version("memery-mcp")
        return __version__ or pkg_version
    except importlib.metadata.PackageNotFoundError:
        return __version__


def _doctor() -> int:
    cfg = get_config()
    print(f"Memery {_version()}")
    print(f"Python: {sys.executable}")
    print(f"Database: {cfg.db_path}")
    print(f"Data directory: {cfg.data_dir}")
    print(f"Profile setup configured: {'yes' if cfg.profile_setup_configured else 'no'}")
    print(f"Default memory profile: {cfg.default_profile_id or '(not set)'}")

    try:
        from .backends.lancedb_backend import LanceDBStore
        vectors = LanceDBStore()
        print(f"Vector backend: lancedb ({vectors.count()} rows)")
    except Exception as exc:
        print(f"Vector backend: error: {exc}")
        return 1

    db_path = Path(cfg.db_path)
    print(f"Database exists: {'yes' if db_path.exists() else 'no'}")
    return 0


def _setup_status(db: MemoryDB | None = None) -> dict:
    from . import config as config_module

    cfg = get_config()
    owns_db = db is None
    db = db or MemoryDB()
    profile = db.get_memory_profile(cfg.default_profile_id)
    status = {
        "configured": bool(cfg.default_profile_id),
        "setup_required": cfg.setup_required() or profile is None,
        "default_profile_id": cfg.default_profile_id or None,
        "default_personality_id": cfg.default_personality_id or None,
        "default_profession_id": cfg.default_profession_id or None,
        "setup_completed_at": cfg.setup_completed_at or None,
        "default_profile": profile,
        "config_file": str(config_module.CONFIG_FILE),
    }
    if cfg.default_profile_id and profile is None:
        status["warning"] = (
            f"Configured default profile '{cfg.default_profile_id}' was not found."
        )
    if owns_db:
        db.close()
    return status


def _print_setup_status() -> int:
    status = _setup_status()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def _print_profiles() -> int:
    db = MemoryDB()
    try:
        print("Personality presets:")
        for index, item in enumerate(db.list_personality_presets(), start=1):
            print(f"{index}. {item['personality_id']} - {item['name']}")
            print(f"   {item['description']}")
        print("\nProfession presets:")
        for index, item in enumerate(db.list_profession_presets(), start=1):
            print(f"{index}. {item['profession_id']} - {item['name']}")
            print(f"   {item['description']}")
    finally:
        db.close()
    return 0


def _prompt_choice(
    items: list[dict],
    default_id: str,
    id_key: str,
    label_key: str = "name",
    title: str = "Choose a preset:",
    prompt_label: str = "Preset",
) -> dict:
    default_index = next(
        (idx for idx, item in enumerate(items, start=1)
         if item[id_key] == default_id),
        1,
    )
    print(f"\n{title}")
    for index, item in enumerate(items, start=1):
        marker = " [recommended]" if index == default_index else ""
        print(f"  {index}. {item[label_key]} ({item[id_key]}){marker}")
        if item.get("description"):
            print(f"     {item['description']}")
    while True:
        raw = input(f"{prompt_label} [{default_index}]: ").strip()
        if not raw:
            return items[default_index - 1]
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1]
        match = next((item for item in items if item[id_key] == raw), None)
        if match:
            return match
        print("Please enter a number from the list or the preset id.")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default:.2f}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number between 0 and 1.")
            continue
        if 0 <= value <= 1:
            return value
        print("Please enter a number between 0 and 1.")


def _customize_traits(profile: dict) -> dict:
    traits = dict(profile.get("traits", {}))
    print("\nTrait weights control selective memory attention.")
    for key in TRAIT_KEYS:
        traits[key] = _prompt_float(key, float(traits.get(key, 0.5) or 0.5))
    profile = {**profile, "traits": traits}
    return profile


def _save_custom_composed_profile(
    db: MemoryDB,
    profile: dict,
    personality: dict,
    profession: dict,
) -> dict:
    custom_id = input(
        f"Custom profile id [{profile['profile_id']}_custom]: "
    ).strip() or f"{profile['profile_id']}_custom"
    return db.upsert_memory_profile(
        profile_id=custom_id,
        traits=personality.get("traits", {}),
        profession=profile.get("profession", {}),
        development_plan=profile.get("development_plan", {}),
        calibration=profile.get("calibration", {}),
        name=f"{personality.get('name', '')} + {profession.get('name', '')} Custom",
        description=profile.get("description", ""),
        version=profile.get("version", "1"),
        source="custom",
    )


def _select_personality(db: MemoryDB, current: str | None) -> dict:
    personalities = db.list_personality_presets()
    recommended = current or "balanced_operator_v1"
    return _prompt_choice(
        personalities,
        recommended,
        "personality_id",
        title="Step 1/2: Choose personality. This shapes attention, execution, and memory pressure.",
        prompt_label="Personality",
    )


def _select_profession(db: MemoryDB, current: str | None) -> dict:
    professions = db.list_profession_presets()
    recommended = current or "generalist_v1"
    return _prompt_choice(
        professions,
        recommended,
        "profession_id",
        title="Step 2/2: Choose profession. This shapes expert objects, methods, evaluation, and error memory.",
        prompt_label="Profession",
    )


def _configure(args: argparse.Namespace) -> int:
    from . import config as config_module

    db = MemoryDB()
    try:
        personalities = db.list_personality_presets()
        professions = db.list_profession_presets()
        if not personalities or not professions:
            print("No personality or profession presets are available.")
            return 1
        cfg = get_config()
        if args.profile:
            selected = db.get_memory_profile(args.profile)
            if not selected:
                print(f"Memory profile '{args.profile}' was not found.")
                return 1
            personality_id = selected.get("development_plan", {}).get("personality_id") or "balanced_operator_v1"
            profession_id = selected.get("development_plan", {}).get("profession_id") or selected.get("profession", {}).get("role", "generalist")
            if args.yes:
                selected_profile = selected
                selected_personality = db.list_personality_presets()[0]
                selected_profession = db.list_profession_presets()[0]
            else:
                selected_profile = selected
                selected_personality = next((item for item in personalities if item["personality_id"] == personality_id), personalities[0])
                selected_profession = next((item for item in professions if item["profession_id"] == profession_id or item["profession"].get("role") == profession_id), professions[0])
        elif args.yes:
            selected_personality = next((item for item in personalities if item["personality_id"] == "balanced_operator_v1"), personalities[0])
            selected_profession = next((item for item in professions if item["profession_id"] == "generalist_v1"), professions[0])
            selected_profile = db.compose_memory_profile(
                selected_personality["personality_id"],
                selected_profession["profession_id"],
            )
        else:
            current_personality = cfg.default_personality_id or "balanced_operator_v1"
            current_profession = cfg.default_profession_id or "generalist_v1"
            selected_personality = _select_personality(db, current_personality)
            selected_profession = _select_profession(db, current_profession)
            selected_profile = db.compose_memory_profile(
                selected_personality["personality_id"],
                selected_profession["profession_id"],
            )
            if _prompt_yes_no("Adjust trait weights?", default=False):
                customized_personality = _customize_traits(selected_personality)
                selected_profile = _save_custom_composed_profile(
                    db, selected_profile, customized_personality, selected_profession,
                )

        cfg.mark_profile_setup(
            selected_profile["profile_id"],
            personality_id=selected_personality["personality_id"],
            profession_id=selected_profession["profession_id"],
        )
        print("\nMemery profile setup complete.")
        print(f"Personality: {selected_personality['personality_id']} - {selected_personality['name']}")
        print(f"Profession: {selected_profession['profession_id']} - {selected_profession['name']}")
        print(f"Default profile: {selected_profile['profile_id']} - {selected_profile.get('name', '')}")
        print(f"Config file: {config_module.CONFIG_FILE}")
        if args.create_context:
            existing = db.get_project_by_name(args.create_context)
            if existing:
                print(f"Context already exists: {args.create_context}")
            else:
                context = db.create_project(
                    args.create_context,
                    description=args.context_description or "",
                    context_type=args.context_type,
                    profile_id=selected_profile["profile_id"],
                )
                print(f"Created context: {context['name']} ({context['profile_id']})")
        print("\nMCP clients can call get_setup_status to detect this state.")
    finally:
        db.close()
    return 0


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(
        prog="memery",
        description="Memery MCP server and diagnostics.",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="run the MCP stdio server (default when no command is given)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"memery-mcp {_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="check the local Memery installation")
    serve = subparsers.add_parser(
        "serve",
        help="run one shared Streamable HTTP MCP service",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--path", default="/mcp")
    serve.add_argument(
        "--json-response",
        action="store_true",
        help="return JSON responses instead of opening SSE streams where possible",
    )
    serve.add_argument(
        "--stateless-http",
        action="store_true",
        help="create an independent MCP transport for each HTTP request",
    )
    subparsers.add_parser("profiles", help="list available personality/profession profiles")
    subparsers.add_parser("setup-status", help="print first-run profile setup status")
    configure = subparsers.add_parser(
        "configure",
        help="choose personality then profession and compose the default profile",
    )
    configure.add_argument(
        "--profile",
        help="profile_id to use without interactive selection",
    )
    configure.add_argument(
        "--yes",
        action="store_true",
        help="accept defaults for non-interactive setup",
    )
    configure.add_argument(
        "--create-context",
        help="optionally create an initial context bound to the selected profile",
    )
    configure.add_argument(
        "--context-type",
        default="auto",
        choices=["auto", "software", "research", "business", "learning", "general"],
        help="context type to use with --create-context",
    )
    configure.add_argument(
        "--context-description",
        default="",
        help="description to use with --create-context",
    )

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _doctor()
    if args.command == "serve":
        from .server import main as server_main
        server_main(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            path=args.path,
            json_response=args.json_response,
            stateless_http=args.stateless_http,
        )
        return None
    if args.command == "profiles":
        return _print_profiles()
    if args.command == "setup-status":
        return _print_setup_status()
    if args.command == "configure":
        return _configure(args)

    from .server import main as server_main
    server_main()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
