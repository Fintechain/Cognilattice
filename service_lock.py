# -*- coding: utf-8 -*-
"""Single-instance guard for the local MCP service."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import sys
import tempfile


class ServiceLockError(RuntimeError):
    """Raised when another Memery service instance already holds the lock."""


class ServiceLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata_path = self.path.with_name(self.path.name + ".json")
        self._file = None

    def acquire(self, metadata: dict | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.lockf(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            self._file = None
            existing = read_service_lock(self.path)
            detail = ""
            if existing.get("endpoint"):
                detail = f" Connect to {existing['endpoint']} instead."
            elif existing.get("mode"):
                detail = f" Running mode: {existing['mode']}."
            raise ServiceLockError(
                "Another Memery/Cognilattice MCP service is already running on this machine. "
                f"Lock file: {self.path}.{detail}"
            ) from exc
        self._file.seek(0)
        self._file.truncate()
        payload = {"pid": os.getpid(), **(metadata or {})}
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()
        self.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def release(self) -> None:
        if not self._file:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.lockf(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
            try:
                self.metadata_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def read_service_lock(path: str | Path) -> dict:
    """Read best-effort service metadata without claiming the lock."""
    path = Path(path)
    metadata_path = path.with_name(path.name + ".json")
    for candidate in (metadata_path, path):
        result = _read_service_metadata(candidate)
        if result:
            return result
    return {}


def _read_service_metadata(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        result = {}
        for line in raw.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
        return result


def get_service_lock_path(config) -> Path:
    lock_path = os.environ.get("MEMERY_SERVICE_LOCK_PATH")
    if lock_path:
        return Path(lock_path)
    return Path(config.cache_dir) / "memery-service.lock"


def acquire_service_lock(config, metadata: dict | None = None) -> ServiceLock:
    """Acquire the process-wide service lock and keep it until interpreter exit."""
    lock_path = get_service_lock_path(config)
    lock = ServiceLock(lock_path)
    try:
        lock.acquire(metadata=metadata)
    except PermissionError:
        fallback = Path(tempfile.gettempdir()) / "memery-service.lock"
        lock = ServiceLock(fallback)
        lock.acquire(metadata=metadata)
    atexit.register(lock.release)
    return lock


def exit_for_lock_error(exc: ServiceLockError) -> None:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from exc
