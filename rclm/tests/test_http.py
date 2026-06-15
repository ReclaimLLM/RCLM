from __future__ import annotations

from rclm import _http


class _FakeContext:
    def __init__(self) -> None:
        self.loaded_ca_files: list[str] = []

    def load_verify_locations(self, *, cafile: str) -> None:
        self.loaded_ca_files.append(cafile)


def test_create_ssl_context_uses_explicit_ca_bundle(monkeypatch):
    calls = []

    def fake_create_default_context(*, cafile=None):
        calls.append(cafile)
        return _FakeContext()

    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/custom-ca.pem")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.setattr(_http.ssl, "create_default_context", fake_create_default_context)

    context = _http.create_ssl_context()

    assert calls == ["/tmp/custom-ca.pem"]
    assert context.loaded_ca_files == []


def test_create_ssl_context_adds_certifi_roots_to_default_context(monkeypatch):
    calls = []
    fake_context = _FakeContext()

    def fake_create_default_context(*, cafile=None):
        calls.append(cafile)
        return fake_context

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.setattr(_http.ssl, "create_default_context", fake_create_default_context)
    monkeypatch.setattr(_http.certifi, "where", lambda: "/tmp/certifi.pem")

    context = _http.create_ssl_context()

    assert context is fake_context
    assert calls == [None]
    assert fake_context.loaded_ca_files == ["/tmp/certifi.pem"]
