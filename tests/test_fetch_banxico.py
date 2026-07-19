"""Parsing tests for BanxicoClient (src/collector/fetch_banxico.py), with
requests.get monkeypatched — no network."""

import pytest
import requests

from src.collector.fetch_banxico import BanxicoClient


def banxico_payload(fecha, dato):
    return {"bmx": {"series": [{"datos": [{"fecha": fecha, "dato": dato}]}]}}


def test_url_property():
    assert BanxicoClient("tok").url == (
        "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43718/datos/oportuno"
    )


def test_fetch_parses_payload_and_converts_date(monkeypatch, fake_response_cls):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return fake_response_cls(banxico_payload("17/07/2026", "18.6543"))

    monkeypatch.setattr("src.collector.fetch_banxico.requests.get", fake_get)

    client = BanxicoClient("tok")
    assert client.fetch_fix_rate() == ("2026-07-17", "18.6543")

    ((url, kwargs),) = calls
    assert url == client.url
    assert kwargs["headers"] == {"Bmx-Token": "tok"}
    assert kwargs["timeout"] == 10


def test_date_conversion_is_day_first(monkeypatch, fake_response_cls):
    monkeypatch.setattr(
        "src.collector.fetch_banxico.requests.get",
        lambda url, **kwargs: fake_response_cls(banxico_payload("01/02/2026", "18.00")),
    )
    date_iso, _ = BanxicoClient("tok").fetch_fix_rate()
    assert date_iso == "2026-02-01"


def test_http_error_propagates(monkeypatch, fake_response_cls):
    monkeypatch.setattr(
        "src.collector.fetch_banxico.requests.get",
        lambda url, **kwargs: fake_response_cls(status_error=requests.HTTPError("401")),
    )
    with pytest.raises(requests.HTTPError):
        BanxicoClient("bad-token").fetch_fix_rate()


def test_unexpected_payload_shape_raises(monkeypatch, fake_response_cls):
    # Malformed payloads must escape, not be swallowed — that's what lets
    # DailyCollector's broad except trigger the fallback source.
    monkeypatch.setattr(
        "src.collector.fetch_banxico.requests.get",
        lambda url, **kwargs: fake_response_cls({}),
    )
    with pytest.raises(KeyError):
        BanxicoClient("tok").fetch_fix_rate()
