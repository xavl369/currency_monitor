"""Parsing tests for FallbackClient (src/collector/fetch_fallback.py), with
requests.get monkeypatched — no network."""

import pytest
import requests

from src.collector.fetch_fallback import FallbackClient


def test_fetch_parses_payload_and_stringifies_rate(monkeypatch, fake_response_cls):
    # Frankfurter sends the rate as a JSON number; the client must return str.
    monkeypatch.setattr(
        "src.collector.fetch_fallback.requests.get",
        lambda url, **kwargs: fake_response_cls(
            {"date": "2026-07-17", "rates": {"MXN": 18.6543}}
        ),
    )
    date_iso, rate = FallbackClient().fetch_rate()
    assert (date_iso, rate) == ("2026-07-17", "18.6543")
    assert isinstance(rate, str)


def test_request_params(monkeypatch, fake_response_cls):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return fake_response_cls({"date": "2026-07-17", "rates": {"MXN": 18.65}})

    monkeypatch.setattr("src.collector.fetch_fallback.requests.get", fake_get)

    FallbackClient().fetch_rate()

    ((url, kwargs),) = calls
    assert url == "https://api.frankfurter.app/latest"
    assert kwargs["params"] == {"from": "USD", "to": "MXN"}
    assert kwargs["timeout"] == 10


def test_http_error_propagates(monkeypatch, fake_response_cls):
    monkeypatch.setattr(
        "src.collector.fetch_fallback.requests.get",
        lambda url, **kwargs: fake_response_cls(status_error=requests.HTTPError("503")),
    )
    with pytest.raises(requests.HTTPError):
        FallbackClient().fetch_rate()


def test_missing_mxn_key_raises(monkeypatch, fake_response_cls):
    monkeypatch.setattr(
        "src.collector.fetch_fallback.requests.get",
        lambda url, **kwargs: fake_response_cls({"date": "2026-07-17", "rates": {}}),
    )
    with pytest.raises(KeyError):
        FallbackClient().fetch_rate()
