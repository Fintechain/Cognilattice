from __future__ import annotations


class FakeLock:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_http_singleton_configures_streamable_transport(monkeypatch):
    import memory_server.server as server

    fake_lock = FakeLock()
    captured = {}

    def acquire(config, metadata=None):
        captured["metadata"] = metadata
        return fake_lock

    def run(transport="stdio", mount_path=None):
        captured["transport"] = transport

    monkeypatch.setattr(server, "acquire_service_lock", acquire)
    monkeypatch.setattr(server.mcp, "run", run)

    server.main(
        transport="streamable-http",
        host="127.0.0.1",
        port=9876,
        path="/memory",
        json_response=True,
    )

    assert captured["transport"] == "streamable-http"
    assert captured["metadata"]["endpoint"] == "http://127.0.0.1:9876/memory"
    assert server.mcp.settings.port == 9876
    assert server.mcp.settings.streamable_http_path == "/memory"
    assert server.mcp.settings.json_response is True
    assert fake_lock.released is True


def test_stdio_mode_remains_available(monkeypatch):
    import memory_server.server as server

    fake_lock = FakeLock()
    captured = {}

    monkeypatch.setattr(
        server,
        "acquire_service_lock",
        lambda config, metadata=None: (
            captured.update({"metadata": metadata}) or fake_lock
        ),
    )
    monkeypatch.setattr(
        server,
        "run_fastmcp_stdio",
        lambda mcp: captured.update({"stdio": True}),
    )

    server.main(transport="stdio")

    assert captured["stdio"] is True
    assert captured["metadata"]["mode"] == "stdio"
    assert captured["metadata"]["endpoint"] is None
    assert fake_lock.released is True
