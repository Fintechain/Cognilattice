from __future__ import annotations

import pytest

from memory_server.service_lock import ServiceLock, ServiceLockError, read_service_lock


def test_service_lock_allows_only_one_holder(tmp_path):
    lock_path = tmp_path / "memery-service.lock"
    first = ServiceLock(lock_path)
    second = ServiceLock(lock_path)

    first.acquire()
    try:
        with pytest.raises(ServiceLockError):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_service_lock_exposes_shared_endpoint(tmp_path):
    lock_path = tmp_path / "memery-service.lock"
    first = ServiceLock(lock_path)
    second = ServiceLock(lock_path)
    endpoint = "http://127.0.0.1:8765/mcp"

    first.acquire({"mode": "http-singleton", "endpoint": endpoint})
    try:
        assert read_service_lock(lock_path)["endpoint"] == endpoint
        with pytest.raises(ServiceLockError, match="Connect to"):
            second.acquire()
    finally:
        first.release()
