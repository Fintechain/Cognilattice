from __future__ import annotations

import json

from memory_server import cli
from memory_server.config import MemeryConfig
from memory_server.db import MemoryDB


def _isolate_config(monkeypatch, tmp_path, db_path=None) -> MemeryConfig:
    import memory_server.config as config_module

    config_dir = tmp_path / "home" / ".memery"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    cfg = MemeryConfig(
        data_dir=str(tmp_path / "data"),
        db_path=str(db_path or tmp_path / "memory.db"),
        cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    return cfg


def test_profile_setup_persists_default_profile(monkeypatch, tmp_path):
    cfg = _isolate_config(monkeypatch, tmp_path)

    cfg.mark_profile_setup(
        "research_scientist_v1",
        personality_id="rigorous_validator_v1",
        profession_id="research_scientist_v1",
        completed_at="2026-06-18T00:00:00Z",
    )

    import memory_server.config as config_module

    saved = json.loads(config_module.CONFIG_FILE.read_text(encoding="utf-8"))
    assert saved["profile_setup_configured"] is True
    assert saved["default_profile_id"] == "research_scientist_v1"
    assert saved["default_personality_id"] == "rigorous_validator_v1"
    assert saved["default_profession_id"] == "research_scientist_v1"
    assert saved["setup_completed_at"] == "2026-06-18T00:00:00Z"

    loaded = MemeryConfig.from_env()
    assert loaded.default_profile_id == "research_scientist_v1"
    assert loaded.default_personality_id == "rigorous_validator_v1"
    assert loaded.default_profession_id == "research_scientist_v1"
    assert loaded.profile_setup_configured is True
    assert loaded.setup_required() is False


def test_personality_profession_presets_compose_profile(tmp_path):
    db = MemoryDB(str(tmp_path / "memory.db"))

    personalities = db.list_personality_presets()
    professions = db.list_profession_presets()
    profile = db.compose_memory_profile(
        "risk_sentinel_v1",
        "software_engineer_v1",
    )

    assert len(personalities) == 36
    assert len(professions) == 216
    assert profile["profile_id"] == "risk_sentinel__software_engineer_v1"
    assert profile["traits"]["risk_sensitivity"] > 0.9
    assert profile["profession"]["role"] == "software_engineer"
    prompt = profile["development_plan"]["tendency_prompt"]
    assert "personality" in prompt["system"][0].lower()
    assert "profession" in prompt["system"][0].lower()
    assert prompt["storage"]
    db.close()


def test_configured_default_profile_is_used_for_new_projects(monkeypatch, tmp_path):
    cfg = _isolate_config(monkeypatch, tmp_path)
    db = MemoryDB(str(tmp_path / "memory.db"))
    profile = db.compose_memory_profile("rigorous_validator_v1", "research_scientist_v1")
    cfg.mark_profile_setup(
        profile["profile_id"],
        personality_id="rigorous_validator_v1",
        profession_id="research_scientist_v1",
        completed_at="2026-06-18T00:00:00Z",
    )

    project = db.create_project("default-profile-project")

    assert project["profile_id"] == "rigorous_validator__research_scientist_v1"
    assert db.get_active_profile_for_project(project["id"])["profile_id"] == profile["profile_id"]
    db.close()


def test_cli_configure_noninteractive_sets_default_profile(monkeypatch, tmp_path):
    db_path = tmp_path / "cli-memory.db"
    _isolate_config(monkeypatch, tmp_path, db_path=db_path)

    result = cli.main(["configure", "--yes"])

    assert result == 0
    db = MemoryDB(str(db_path))
    status = cli._setup_status(db)
    assert status["configured"] is True
    assert status["setup_required"] is False
    assert status["default_personality_id"] == "balanced_operator_v1"
    assert status["default_profession_id"] == "generalist_v1"
    assert status["default_profile"]["profession"]["role"] == "generalist"
    db.close()


def test_server_setup_tools_configure_default(monkeypatch, tmp_path):
    import memory_server.server as server

    cfg = _isolate_config(monkeypatch, tmp_path)
    db = MemoryDB(str(tmp_path / "server-memory.db"))
    monkeypatch.setattr(server, "cfg", cfg)
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server.curator, "db", db)

    before = server.get_setup_status()
    assert before["setup_required"] is True

    configured = server.configure_memory_defaults(
        personality_id="risk_sentinel_v1",
        profession_id="clinical_reasoner_v1",
    )

    assert configured["status"] == "configured"
    assert configured["default_personality_id"] == "risk_sentinel_v1"
    assert configured["default_profession_id"] == "clinical_reasoner_v1"
    assert server.get_setup_status()["default_profile"]["profession"]["role"] == "clinical_reasoner"
    project = db.create_project("clinical-context")
    assert project["profile_id"] == "risk_sentinel__clinical_reasoner_v1"
    db.close()
