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


def test_environment_overrides_config_file(monkeypatch, tmp_path):
    import memory_server.config as config_module

    config_dir = tmp_path / "home" / ".memery"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.json"
    config_file.write_text(
        json.dumps({
            "db_path": str(tmp_path / "file.db"),
            "data_dir": str(tmp_path / "file-data"),
            "default_profile_id": "generalist_v1",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setenv("MEMERY_DB_PATH", str(tmp_path / "env.db"))
    monkeypatch.setenv("MEMERY_DATA_DIR", str(tmp_path / "env-data"))
    monkeypatch.setenv("MEMERY_DEFAULT_PROFILE_ID", "software_engineer_v1")

    loaded = MemeryConfig.from_env()

    assert loaded.db_path == str(tmp_path / "env.db")
    assert loaded.data_dir == str(tmp_path / "env-data")
    assert loaded.default_profile_id == "software_engineer_v1"


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
    cfg = _isolate_config(monkeypatch, tmp_path)
    import memory_server.server as server

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


def test_create_memory_profile_accepts_preset_ids_and_overrides(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    import memory_server.server as server

    db = MemoryDB(str(tmp_path / "server-memory.db"))
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server.curator, "db", db)

    created = server.create_memory_profile(
        profile_id="rigorous_stats_probe_v1",
        personality_id="rigorous_validator_v1",
        profession_id="statistician_v1",
        overrides={"calibration": {"store_threshold": 0.82}},
    )

    assert created["status"] == "saved"
    profile = created["profile"]
    assert profile["profile_id"] == "rigorous_stats_probe_v1"
    assert profile["profession"]["role"] == "statistician"
    assert profile["traits"]["rigor"] > 0.9
    assert profile["calibration"]["store_threshold"] == 0.82
    db.close()


def test_create_memory_profile_reports_invalid_json_field(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    import memory_server.server as server

    db = MemoryDB(str(tmp_path / "server-memory.db"))
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server.curator, "db", db)

    result = server.create_memory_profile(
        profile_id="bad_json_probe_v1",
        traits={"rigor": 0.8},
        profession={"role": "tester"},
        calibration="not json",
    )

    assert result["error"].startswith("Invalid JSON in field 'calibration':")
    db.close()


def test_create_memory_profile_reports_invalid_override_field(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    import memory_server.server as server

    db = MemoryDB(str(tmp_path / "server-memory.db"))
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server.curator, "db", db)

    result = server.create_memory_profile(
        profile_id="bad_override_probe_v1",
        personality_id="rigorous_validator_v1",
        profession_id="statistician_v1",
        overrides={"calibration": "store threshold should be high"},
    )

    assert result["error"] == (
        "Invalid JSON in field 'overrides.calibration': expected an object, got str."
    )
    db.close()


def test_preset_lists_are_compact_pageable_and_addressable(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    import memory_server.server as server

    db = MemoryDB(str(tmp_path / "server-memory.db"))
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server.curator, "db", db)

    page = server.list_profession_presets(offset=0, limit=3)
    assert page["count"] == 3
    assert page["total"] == 216
    assert page["has_more"] is True
    assert "profession_prompt" not in page["professions"][0]

    single = server.list_profession_presets(id="statistician_v1")
    assert single["profession"]["profession_id"] == "statistician_v1"
    assert single["profession"]["role"] == "statistician"

    status = server.get_setup_status()
    assert status["available_counts"]["professions"] == 216
    assert status["available_professions_truncated"] is True
    assert len(status["available_professions"]) == 50
    db.close()


def test_delete_memory_profile_removes_custom_but_protects_builtin(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    import memory_server.server as server

    db = MemoryDB(str(tmp_path / "server-memory.db"))
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server.curator, "db", db)
    db.compose_memory_profile(
        "risk_sentinel_v1",
        "software_engineer_v1",
        profile_id="probe_tmp_v1",
    )

    deleted = server.delete_memory_profile("probe_tmp_v1")
    assert deleted == {"status": "deleted", "profile_id": "probe_tmp_v1"}
    assert db.get_memory_profile("probe_tmp_v1") is None

    protected = server.delete_memory_profile("generalist_v1")
    assert "cannot be deleted" in protected["error"]
    assert db.get_memory_profile("generalist_v1") is not None
    db.close()
